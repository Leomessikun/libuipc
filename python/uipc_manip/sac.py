"""Scalar Soft Actor-Critic on point clouds, ported from the Newton dressing teacher.

The defaults reproduce the Wang RSS 2023 ``pointcloud_3`` launcher as recorded in
the Newton pipeline notes: actor and critic learning rate ``1e-4``, batch 64,
actor updates every fourth optimizer step, Polyak coefficients ``0.01`` for the
Q heads and ``0.05`` for the critic encoder every second update, a learned
entropy temperature with target ``-action_dim``, no gradient clipping, and the
reference discount ``0.99`` at the reference 150-step horizon. The helpers
below rescale the discount, temperature learning rate, and replay reward
scale for other horizons exactly as the Newton port does.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import Actor, Critic, EncoderConfig
from .obs import FLAG_TOOL, ObsSpec

WANG_HORIZON_STEPS = 150
WANG_DISCOUNT = 0.99
WANG_ALPHA_LR = 1.0e-4


def wang_equivalent_discount(horizon: int) -> float:
    """Discount whose planning horizon is the same fraction of ``horizon`` as ``0.99`` is of 150 steps."""
    horizon = max(1, int(horizon))
    reference_fraction = (1.0 / (1.0 - WANG_DISCOUNT)) / float(WANG_HORIZON_STEPS)
    planning_steps = max(1.0, reference_fraction * float(horizon))
    return float(1.0 - 1.0 / planning_steps)


def wang_equivalent_alpha_lr(horizon: int) -> float:
    """Temperature learning rate that spends the reference per-episode anneal over ``horizon`` steps."""
    return float(WANG_ALPHA_LR * float(WANG_HORIZON_STEPS) / float(max(1, int(horizon))))


def wang_equivalent_reward_scale(discount: float) -> float:
    """Replay reward multiplier keeping ``Q ~ r * 100`` at any discount."""
    return float((1.0 - float(discount)) / (1.0 - WANG_DISCOUNT))


def gradient_update_budget(
    *, transitions_added: int, replay_size: int, batch_size: int, updates_started: bool, updates_per_step: int
) -> tuple[int, bool]:
    """Vector-step update budget that avoids replay-prefill overshoot (Newton ``sac_gradient_update_budget``)."""
    if replay_size <= batch_size:
        return 0, bool(updates_started)
    budget = int(transitions_added) if updates_per_step <= 0 else int(updates_per_step)
    if not updates_started:
        budget = min(budget, replay_size - batch_size - 1)
    budget = max(0, budget)
    return budget, bool(updates_started or budget > 0)


@dataclass
class SACConfig:
    discount: float = WANG_DISCOUNT
    critic_tau: float = 0.01
    encoder_tau: float = 0.05
    actor_update_freq: int = 4
    critic_target_update_freq: int = 2
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    alpha_lr: float = WANG_ALPHA_LR
    actor_beta: float = 0.9
    critic_beta: float = 0.9
    alpha_beta: float = 0.5
    init_temperature: float = 0.1
    alpha_fixed: bool = False
    min_alpha: float = 0.0
    target_entropy_scale: float = 1.0
    hidden_dim: int = 1024
    batch_size: int = 64
    reward_abs_bound: float = 2.0
    grad_clip_max_norm: float = 0.0
    random_shift_scale: float = 0.0
    point_jitter_scale: float = 0.0
    actor_log_std_min: float = -10.0
    actor_log_std_max: float = 2.0
    use_extra: bool = True
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["encoder"] = self.encoder.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SACConfig":
        data = dict(data)
        data["encoder"] = EncoderConfig.from_dict(data.get("encoder", {}))
        return cls(**data)


def soft_update(src: torch.nn.Module, tgt: torch.nn.Module, tau: float) -> None:
    for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


@contextlib.contextmanager
def frozen_parameters(module: torch.nn.Module):
    saved = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in saved:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, was in saved:
            p.requires_grad_(was)


class SACAgent:
    def __init__(self, spec: ObsSpec, action_dim: int, cfg: SACConfig, device) -> None:
        self.spec = spec
        self.action_dim = int(action_dim)
        self.cfg = cfg
        self.device = torch.device(device)
        self.actor = Actor(
            spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra, cfg.actor_log_std_min, cfg.actor_log_std_max
        ).to(self.device)
        self.critic = Critic(spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra).to(self.device)
        self.critic_target = Critic(spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(cfg.init_temperature), dtype=torch.float32, device=self.device)
        self.log_alpha.requires_grad_(True)
        self.target_entropy = -float(cfg.target_entropy_scale) * float(action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=(cfg.actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr, betas=(cfg.critic_beta, 0.999)
        )
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr, betas=(cfg.alpha_beta, 0.999))
        self.updates = 0
        self.train()

    def train(self, training: bool = True) -> None:
        self.actor.train(training)
        self.critic.train(training)
        self.critic_target.train(training)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    def _unpack(self, flat: torch.Tensor, augment: bool = False):
        pos, feat, valid, extra = self.spec.unpack_torch(flat)
        if augment:
            pos = self._augment(pos, feat, valid)
        return pos, feat, valid, extra

    def _augment(self, pos: torch.Tensor, feat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.random_shift_scale > 0.0:
            shift = torch.empty((pos.shape[0], 1, 3), device=pos.device).uniform_(
                -cfg.random_shift_scale, cfg.random_shift_scale
            )
            pos = pos + shift
        if cfg.point_jitter_scale > 0.0:
            jitter = torch.empty_like(pos).uniform_(-cfg.point_jitter_scale, cfg.point_jitter_scale)
            keep = valid & (feat[:, :, FLAG_TOOL] < 0.5)
            pos = pos + jitter * keep[..., None]
        return pos

    def act(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            flat = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device).reshape(-1, self.spec.dim)
            batch = self._unpack(flat)
            if deterministic:
                mu, _, _, _ = self.actor(batch, compute_pi=False, compute_log_pi=False)
                return mu.cpu().numpy()
            _, pi, _, _ = self.actor(batch, compute_log_pi=False)
            return pi.cpu().numpy()

    # ------------------------------------------------------------------
    def _update_critic(self, obs, action, reward, next_obs, not_done) -> dict:
        with torch.no_grad():
            _, next_pi, next_log_pi, _ = self.actor(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs, next_pi)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_pi
            target_q = reward + not_done * self.cfg.discount * target_v
            bound = float(self.cfg.reward_abs_bound) / max(1.0 - float(self.cfg.discount), 1e-6)
            target_q = target_q.clamp(-bound, bound)
        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = self._clip(self.critic)
        self.critic_optimizer.step()
        return {"critic_loss": float(critic_loss.item()), "q1_mean": float(current_q1.mean().item()), "critic_grad_norm": grad_norm}

    def _update_actor_and_alpha(self, obs) -> dict:
        _, pi, log_pi, _ = self.actor(obs)
        with frozen_parameters(self.critic):
            q1, q2 = self.critic(obs, pi, detach_encoder=True)
        actor_loss = (self.alpha.detach() * log_pi - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = self._clip(self.actor)
        self.actor_optimizer.step()
        stats = {"actor_loss": float(actor_loss.item()), "entropy": float(-log_pi.mean().item()), "actor_grad_norm": grad_norm}
        if not self.cfg.alpha_fixed:
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            if self.cfg.min_alpha > 0.0:
                with torch.no_grad():
                    self.log_alpha.clamp_(min=float(np.log(self.cfg.min_alpha)))
            stats["alpha_loss"] = float(alpha_loss.item())
        stats["alpha"] = float(self.alpha.item())
        return stats

    def _clip(self, module: torch.nn.Module) -> float:
        max_norm = float(self.cfg.grad_clip_max_norm) if self.cfg.grad_clip_max_norm > 0 else float("inf")
        return float(torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_norm))

    def update(self, replay) -> dict:
        """One optimizer update following the Wang actor/target schedule."""
        obs_flat, action, reward, next_obs_flat, not_done = replay.sample(self.cfg.batch_size)
        obs = self._unpack(obs_flat, augment=True)
        next_obs = self._unpack(next_obs_flat, augment=True)
        stats = self._update_critic(obs, action, reward, next_obs, not_done)
        stats["batch_reward"] = float(reward.mean().item())
        self.updates += 1
        if self.updates % self.cfg.actor_update_freq == 0:
            stats.update(self._update_actor_and_alpha(obs))
        if self.updates % self.cfg.critic_target_update_freq == 0:
            soft_update(self.critic.Q1, self.critic_target.Q1, self.cfg.critic_tau)
            soft_update(self.critic.Q2, self.critic_target.Q2, self.cfg.critic_tau)
            soft_update(self.critic.encoder, self.critic_target.encoder, self.cfg.encoder_tau)
        return stats

    # ------------------------------------------------------------------
    def protocol(self) -> dict:
        """Settings a checkpoint must agree on before its weights can be reused."""
        return {
            "point_budget": int(self.spec.point_budget),
            "obs_dim": int(self.spec.dim),
            "action_dim": int(self.action_dim),
            "hidden_dim": int(self.cfg.hidden_dim),
            "use_extra": bool(self.cfg.use_extra),
            "encoder": self.cfg.encoder.to_dict(),
        }

    def save(self, path: str | Path, step: int, metadata: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "updates": int(self.updates),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "sac_config": self.cfg.to_dict(),
            "protocol": self.protocol(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        path.with_suffix(".json").write_text(
            json.dumps({"step": int(step), "protocol": payload["protocol"], "metadata": payload["metadata"]}, indent=2)
            + "\n"
        )
        return path

    @staticmethod
    def read_checkpoint(path: str | Path) -> dict:
        return torch.load(Path(path), map_location="cpu", weights_only=False)

    def load(self, path: str | Path, *, load_optimizers: bool = True, strict_protocol: bool = True) -> dict:
        payload = self.read_checkpoint(path)
        if strict_protocol and payload["protocol"] != self.protocol():
            raise ValueError(
                f"Checkpoint protocol {json.dumps(payload['protocol'], sort_keys=True)} does not match "
                f"{json.dumps(self.protocol(), sort_keys=True)}"
            )
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(payload["log_alpha"].to(self.device))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.log_alpha_optimizer.load_state_dict(payload["log_alpha_optimizer"])
        self.updates = int(payload.get("updates", 0))
        return payload
