"""Point-cloud Flow Q-Learning (Park, Li and Levine, ICML 2025).

The objectives follow https://github.com/seohongpark/fql (agents/fql.py).
Observation encoders and dense action-conditioned residual critics reuse this
repository's dressing networks. This is a reference-method adaptation, not IPC
policy gradients or a new algorithm. Checkpoints are explicitly distinct from SAC.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import Critic, EncoderConfig, WangFlowActor, reuse_neighbourhoods, trunk_layers
from .obs import EXTRA_DIM, ObsSpec
from .sac import frozen_parameters, soft_update


@dataclass
class FQLConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    hidden_dim: int = 256
    trunk_blocks: int = 2
    discount: float = .995
    tau: float = .005
    learning_rate: float = 3e-4
    alpha: float = 10.0
    flow_steps: int = 10
    q_agg: str = "mean"

    def __post_init__(self):
        if isinstance(self.encoder, dict):
            self.encoder = EncoderConfig.from_dict(self.encoder)
        if (not 0 < self.discount < 1 or not 0 < self.tau <= 1
                or not np.isfinite([self.learning_rate, self.alpha]).all()
                or self.learning_rate <= 0 or self.alpha < 0 or self.flow_steps < 1
                or self.q_agg not in ("mean", "min")):
            raise ValueError("Invalid FQL configuration")


class FlowPolicy(WangFlowActor):
    """Existing tool-point encoder followed by a noise-conditioned action head.

    The behavior model takes flow time as an extra input; the one-step policy
    does not. Encoding is separated so Euler integration encodes the cloud once.
    """

    def __init__(self, spec, action_dim, cfg, *, timed):
        super().__init__(spec, action_dim, cfg.hidden_dim, cfg.encoder,
                         trunk_style="residual", trunk_blocks=cfg.trunk_blocks)
        self.timed = timed
        width = self.encoder.feature_dim + EXTRA_DIM + action_dim + int(timed)
        self.trunk = trunk_layers(width, cfg.hidden_dim, action_dim, "residual", cfg.trunk_blocks)

    def vector(self, encoded, x, time=None):
        inputs = [encoded, x]
        if self.timed:
            if time is None:
                raise ValueError("The behavior vector field needs a flow time")
            inputs.append(time)
        return self.trunk(torch.cat(inputs, dim=-1))

    def forward(self, obs, noise, time=None):
        return self.vector(self._frame_latent(obs), noise, time)


def bellman_target(reward, mask, q1, q2, *, discount, aggregation="mean"):
    """Ordinary return, with no SAC entropy term; timeouts retain mask=1."""
    q = (q1 + q2) * .5 if aggregation == "mean" else torch.minimum(q1, q2)
    return reward + discount * mask * q


class FQLAgent:
    def __init__(self, spec: ObsSpec, action_dim: int, cfg: FQLConfig, device="cuda"):
        self.spec, self.action_dim, self.cfg = spec, int(action_dim), cfg
        self.device = torch.device(device)
        self.actor = FlowPolicy(spec, action_dim, cfg, timed=False).to(self.device)
        self.behavior = FlowPolicy(spec, action_dim, cfg, timed=True).to(self.device)
        self.critic = Critic(spec, action_dim, cfg.hidden_dim, cfg.encoder,
                             action_mode="dense", trunk_style="residual",
                             trunk_blocks=cfg.trunk_blocks).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.modules = nn.ModuleDict(dict(actor=self.actor, behavior=self.behavior,
                                          critic=self.critic, critic_target=self.critic_target))
        self.optimizer = torch.optim.Adam(
            [p for p in self.modules.parameters() if p.requires_grad], lr=cfg.learning_rate,
            fused=self.device.type == "cuda")
        self.updates = 0

    def unpack(self, flat):
        pos, feat, valid, extra = self.spec.unpack_torch(flat)
        # Same valid-first padding removal as SAC, before repeated cloud encodes.
        if self.cfg.encoder.kind == "pointnet2" and all(r >= 1 for r in self.cfg.encoder.sa_ratio):
            used = valid.any(dim=0).nonzero()
            end = int(used.max()) + 1 if used.numel() else 1
            pos, feat, valid = pos[:, :end], feat[:, :end], valid[:, :end]
        return pos, feat, valid, extra

    @torch.no_grad()
    def flow_actions(self, encoded, noise):
        x = noise
        for k in range(self.cfg.flow_steps):
            time = x.new_full((len(x), 1), k / self.cfg.flow_steps)
            x = x + self.behavior.vector(encoded, x, time) / self.cfg.flow_steps
        return x.clamp(-1, 1)

    def update(self, batch):
        flat, actions, rewards, next_flat, masks = batch
        with reuse_neighbourhoods():
            obs, nxt = self.unpack(flat), self.unpack(next_flat)
            with torch.no_grad():
                noise = torch.randn_like(actions)
                next_actions = self.actor(nxt, noise).clamp(-1, 1)
                next_q1, next_q2 = self.critic_target(nxt, next_actions)
                target = bellman_target(rewards, masks, next_q1, next_q2,
                                        discount=self.cfg.discount, aggregation=self.cfg.q_agg)
            q1, q2 = self.critic(obs, actions)
            critic_loss = .5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))

            encoded = self.behavior._frame_latent(obs)
            x0, time = torch.randn_like(actions), torch.rand_like(rewards)
            xt = (1 - time) * x0 + time * actions
            velocity = self.behavior.vector(encoded, xt, time)
            flow_loss = F.mse_loss(velocity, actions - x0)

            noise = torch.randn_like(actions)
            flow_target = self.flow_actions(encoded.detach(), noise)
            raw_actions = self.actor(obs, noise)
            distill_loss = F.mse_loss(raw_actions, flow_target)
            with frozen_parameters(self.critic):
                aq1, aq2 = self.critic(obs, raw_actions.clamp(-1, 1))
                actor_q = .5 * (aq1 + aq2)
            q_loss = -actor_q.mean()
            loss = critic_loss + flow_loss + self.cfg.alpha * distill_loss + q_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            soft_update(self.critic, self.critic_target, self.cfg.tau)
        self.updates += 1
        # No scalar device-to-host copy on every update; the runner reports in batches.
        return {k: v.detach() for k, v in dict(loss=loss, critic_loss=critic_loss,
                    flow_loss=flow_loss, distill_loss=distill_loss, q_loss=q_loss,
                    data_q=.5 * (q1.mean() + q2.mean()), target_q=target.mean(),
                    actor_q=actor_q.mean(), action_mse=F.mse_loss(raw_actions.clamp(-1, 1), actions)).items()}

    @torch.no_grad()
    def act(self, observations, deterministic=False, *, behavior=False):
        obs = self.unpack(torch.as_tensor(observations, dtype=torch.float32, device=self.device))
        noise = torch.zeros((len(observations), self.action_dim), device=self.device)
        if not deterministic:
            noise.normal_()
        if behavior:
            actions = self.flow_actions(self.behavior._frame_latent(obs), noise)
        else:
            actions = self.actor(obs, noise).clamp(-1, 1)
        return actions.cpu().numpy()

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format="dressing_fql_v1", config=asdict(self.cfg),
                       point_budget=self.spec.point_budget, action_dim=self.action_dim,
                       models=self.modules.state_dict(), optimizer=self.optimizer.state_dict(),
                       updates=self.updates, metadata=metadata or {}, rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cuda", *, resume=False):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "dressing_fql_v1":
            raise ValueError("Expected an explicit FQL checkpoint, not SAC weights")
        agent = cls(ObsSpec(payload["point_budget"]), payload["action_dim"], FQLConfig(**payload["config"]), device)
        agent.modules.load_state_dict(payload["models"])
        agent.updates = payload["updates"]
        if resume:
            agent.optimizer.load_state_dict(payload["optimizer"])
            torch.set_rng_state(payload["rng"])
            if agent.device.type == "cuda" and payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state(payload["cuda_rng"], agent.device)
        return agent, payload["metadata"]
