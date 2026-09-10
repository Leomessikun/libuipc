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
import functools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .models import Actor, CategoricalCritic, Critic, EncoderConfig, PrivilegedCritic, WangFlowActor, reuse_neighbourhoods
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
    actor_type: str = "wang-flow"
    """``wang-flow`` reads the tool point of a segmentation encoder (the reference); ``flat`` pools globally."""
    algo: str = "sac"
    """``sac`` is the scalar reference critic; ``flashsac`` is the bounded categorical critic."""
    num_bins: int = 51
    min_v: float = -50.0
    max_v: float = 50.0
    critic_input: str = "points"
    """``points`` encodes the point cloud as the actor does (the reference); ``privileged`` is an asymmetric
    critic on the simulator's low-dimensional state, which only training reads."""
    privileged_dim: int = 0
    """Length of that state; the trainer sets it from the environment."""
    distill_weight: float = 0.0
    """Weight of Wang's teacher loss on the actor (``bc_loss_weight`` in ``SAC_AWAC.py``; the paper states
    0.01, the reference launcher 0.002). It acts once teachers are attached with :meth:`SACAgent.set_teachers`."""
    encoder_precision: str = "fp32"
    """``bf16`` runs the point encoders under bfloat16 autocast and hands their features on in fp32; the
    heads, targets, and losses stay fp32 either way. Weights are unchanged, so checkpoints load across both."""
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


def wang_distill_loss(student_mu, student_log_std, teacher_mu, teacher_log_std) -> torch.Tensor:
    """Wang RSS 2023's teacher loss, as ``SAC_AWAC.py:1025-1036`` computes it.

    The squared distance of the squashed means plus the squared distance of the
    square-rooted standard deviations, summed over the batch and the action
    dimensions rather than averaged.
    """
    return ((teacher_mu - student_mu) ** 2).sum() + ((teacher_log_std.exp().sqrt() - student_log_std.exp().sqrt()) ** 2).sum()


def soft_update(src: torch.nn.Module, tgt: torch.nn.Module, tau: float) -> None:
    for sp, tp in zip(src.parameters(), tgt.parameters(), strict=True):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def bf16_forward(forward, device_type: str):
    """``forward`` under bfloat16 autocast, returning fp32 so that only the wrapped module changes precision."""

    @functools.wraps(forward)
    def run(*args, **kwargs):
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            out = forward(*args, **kwargs)
        return out.float()

    return run


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
        if cfg.actor_type == "wang-flow":
            actor_cls = WangFlowActor
        elif cfg.actor_type == "flat":
            actor_cls = Actor
        else:
            raise ValueError(f"Unknown actor_type {cfg.actor_type!r}")
        self.actor = actor_cls(
            spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra, cfg.actor_log_std_min, cfg.actor_log_std_max
        ).to(self.device)
        if cfg.critic_input == "privileged":
            if cfg.algo != "sac" or int(cfg.privileged_dim) <= 0:
                raise ValueError("The privileged critic is the scalar 'sac' critic and needs privileged_dim > 0")
            make_critic = lambda: PrivilegedCritic(cfg.privileged_dim, action_dim, cfg.hidden_dim)  # noqa: E731
        elif cfg.critic_input != "points":
            raise ValueError(f"Unknown critic_input {cfg.critic_input!r}")
        elif cfg.algo == "sac":
            make_critic = lambda: Critic(spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra)  # noqa: E731
        elif cfg.algo == "flashsac":
            make_critic = lambda: CategoricalCritic(  # noqa: E731
                spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.num_bins, cfg.min_v, cfg.max_v, cfg.use_extra
            )
        else:
            raise ValueError(f"Unknown algo {cfg.algo!r}")
        self.critic = make_critic().to(self.device)
        self.critic_target = make_critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        if cfg.encoder_precision == "bf16":
            # Q values here run near 90 where bfloat16 resolves about 0.5, so only the point work is lowered.
            for module in (self.actor, self.critic, self.critic_target):
                if getattr(module, "encoder", None) is not None:
                    module.encoder.forward = bf16_forward(module.encoder.forward, self.device.type)
        elif cfg.encoder_precision != "fp32":
            raise ValueError(f"Unknown encoder_precision {cfg.encoder_precision!r}")
        self.log_alpha = torch.tensor(np.log(cfg.init_temperature), dtype=torch.float32, device=self.device)
        self.log_alpha.requires_grad_(True)
        self.target_entropy = -float(cfg.target_entropy_scale) * float(action_dim)
        # The fused kernel keeps Adam on the device; the default path reads two scalars per
        # parameter tensor back to the host every step, over a hundred synchronisations per update.
        fused = self.device.type == "cuda"
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr, betas=(cfg.actor_beta, 0.999), fused=fused
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr, betas=(cfg.critic_beta, 0.999), fused=fused
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=cfg.alpha_lr, betas=(cfg.alpha_beta, 0.999), fused=fused
        )
        self.updates = 0
        self.teachers: dict[int, torch.nn.Module] = {}
        # Clouds are packed valid-first and every PointNet++ stage masks by validity, so the columns
        # no cloud in a batch reaches change nothing. A sampling ratio below one would draw its
        # centres from the padded length, so the cut is taken only at ratio one.
        self._cut_padding = cfg.encoder.kind == "pointnet2" and all(float(r) >= 1.0 for r in cfg.encoder.sa_ratio)
        self.train()

    def set_teachers(self, teachers: dict[int, torch.nn.Module]) -> None:
        """Frozen teacher actors keyed by replay label, the arm-pose region, for Wang's distillation."""
        self.teachers = {}
        for label, actor in teachers.items():
            actor = actor.to(self.device).eval()
            for p in actor.parameters():
                p.requires_grad_(False)
            self.teachers[int(label)] = actor

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
        if self._cut_padding:
            used = valid.any(dim=0).nonzero()
            n = int(used.max()) + 1 if used.numel() else 1
            pos, feat, valid = pos[:, :n], feat[:, :n], valid[:, :n]
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
    def _update_critic(self, obs, action, reward, next_obs, not_done, state=None, next_state=None) -> dict:
        """``state`` and ``next_state``, when given, are what the critic reads; the actor always reads observations."""
        with torch.no_grad():
            _, next_pi, next_log_pi, _ = self.actor(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs if next_state is None else next_state, next_pi)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_pi
            target_q = reward + not_done * self.cfg.discount * target_v
            bound = float(self.cfg.reward_abs_bound) / max(1.0 - float(self.cfg.discount), 1e-6)
            target_q = target_q.clamp(-bound, bound)
        current_q1, current_q2 = self.critic(obs if state is None else state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = self._clip(self.critic)
        self.critic_optimizer.step()
        return {"critic_loss": float(critic_loss.item()), "q1_mean": float(current_q1.mean().item()), "critic_grad_norm": grad_norm}

    def _critic_scalar(self, obs, action, detach_encoder: bool = False):
        """Scalar ``(Q1, Q2)`` for either critic form."""
        out = self.critic(obs, action, detach_encoder=detach_encoder)
        return out[0], out[1]

    def _update_critic_categorical(self, obs, action, reward, next_obs, not_done) -> dict:
        """FlashSAC-style update: project the soft Bellman target onto the atom grid, cross-entropy loss."""
        with torch.no_grad():
            _, next_pi, next_log_pi, _ = self.actor(next_obs)
            tq1, tq2, tlp1, tlp2 = self.critic_target(next_obs, next_pi)
            target_min = torch.minimum(tq1, tq2)
            target_scalar = reward + not_done * self.cfg.discount * (target_min - self.alpha.detach() * next_log_pi)
            target_probs = self._project_categorical_target(target_scalar, tlp1.shape[-1])
        q1, _, log_p1, log_p2 = self.critic(obs, action)
        critic_loss = -(target_probs * log_p1).sum(-1).mean() - (target_probs * log_p2).sum(-1).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = self._clip(self.critic)
        self.critic_optimizer.step()
        return {"critic_loss": float(critic_loss.item()), "q1_mean": float(q1.mean().item()), "critic_grad_norm": grad_norm}

    def _project_categorical_target(self, target_scalar: torch.Tensor, num_bins: int) -> torch.Tensor:
        """Point-mass projection of the scalar target onto its two neighbouring atoms (Newton's C51-on-mean)."""
        min_v, max_v = float(self.cfg.min_v), float(self.cfg.max_v)
        width = (max_v - min_v) / (num_bins - 1)
        b = (target_scalar.clamp(min_v, max_v) - min_v) / width
        lower = torch.floor(b).long().clamp(0, num_bins - 1)
        upper = (lower + 1).clamp(0, num_bins - 1)
        frac = b - lower.float()
        probs = torch.zeros(target_scalar.shape[0], num_bins, device=target_scalar.device, dtype=target_scalar.dtype)
        probs.scatter_add_(1, lower, 1.0 - frac)
        probs.scatter_add_(1, upper, frac)
        return probs

    def _update_actor_and_alpha(self, obs, state=None, label=None) -> dict:
        mu, pi, log_pi, log_std = self.actor(obs)
        with frozen_parameters(self.critic):
            q1, q2 = self._critic_scalar(obs if state is None else state, pi, detach_encoder=True)
        actor_loss = (self.alpha.detach() * log_pi - torch.min(q1, q2)).mean()
        distill = None
        if self.teachers and label is not None and self.cfg.distill_weight > 0.0:
            # Each row is pulled toward its own region's teacher on the same observation; rows
            # whose region has no teacher are left out, as in the reference.
            distill = torch.zeros((), device=mu.device)
            for region, teacher in self.teachers.items():
                rows = (label == region).nonzero(as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                with torch.no_grad():
                    teacher_mu, _, _, teacher_log_std = teacher(tuple(t[rows] for t in obs), compute_pi=False, compute_log_pi=False)
                distill = distill + wang_distill_loss(mu[rows], log_std[rows], teacher_mu, teacher_log_std)
            actor_loss = actor_loss + self.cfg.distill_weight * distill
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = self._clip(self.actor)
        self.actor_optimizer.step()
        stats = {"actor_loss": float(actor_loss.item()), "entropy": float(-log_pi.mean().item()), "actor_grad_norm": grad_norm}
        if distill is not None:
            stats["distill_loss"] = float(distill.item())
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
        with reuse_neighbourhoods():
            return self._update(replay)

    def _update(self, replay) -> dict:
        batch = replay.sample(self.cfg.batch_size)
        obs_flat, action, reward, next_obs_flat, not_done = batch[:5]
        rest = list(batch[5:])
        state = next_state = None
        if self.cfg.critic_input == "privileged":
            if not getattr(replay, "priv_dim", 0):
                raise ValueError("The privileged critic needs a replay buffer that stores the privileged state")
            state, next_state = rest[:2]
        if getattr(replay, "priv_dim", 0):
            rest = rest[2:]
        label = rest[0] if getattr(replay, "labelled", False) else None
        obs = self._unpack(obs_flat, augment=True)
        next_obs = self._unpack(next_obs_flat, augment=True)
        if self.cfg.algo == "flashsac":
            stats = self._update_critic_categorical(obs, action, reward, next_obs, not_done)
        else:
            stats = self._update_critic(obs, action, reward, next_obs, not_done, state, next_state)
        stats["batch_reward"] = float(reward.mean().item())
        self.updates += 1
        if self.updates % self.cfg.actor_update_freq == 0:
            stats.update(self._update_actor_and_alpha(obs, state, label))
        if self.updates % self.cfg.critic_target_update_freq == 0:
            soft_update(self.critic.Q1, self.critic_target.Q1, self.cfg.critic_tau)
            soft_update(self.critic.Q2, self.critic_target.Q2, self.cfg.critic_tau)
            if self.critic.encoder is not None:
                soft_update(self.critic.encoder, self.critic_target.encoder, self.cfg.encoder_tau)
        return stats

    # ------------------------------------------------------------------
    def protocol(self) -> dict:
        """Settings a checkpoint must agree on before its weights can be reused."""
        protocol = {
            "point_budget": int(self.spec.point_budget),
            "obs_dim": int(self.spec.dim),
            "action_dim": int(self.action_dim),
            "hidden_dim": int(self.cfg.hidden_dim),
            "use_extra": bool(self.cfg.use_extra),
            "actor_type": str(self.cfg.actor_type),
            "algo": str(self.cfg.algo),
            "num_bins": int(self.cfg.num_bins),
            "min_v": float(self.cfg.min_v),
            "max_v": float(self.cfg.max_v),
            "encoder": self.cfg.encoder.to_dict(),
        }
        if self.cfg.critic_input != "points":
            # Only a privileged critic adds these keys, so point-critic checkpoints saved before them still load.
            protocol.update(critic_input=str(self.cfg.critic_input), privileged_dim=int(self.cfg.privileged_dim))
        return protocol

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
        if load_optimizers and SACConfig.from_dict(payload["sac_config"]).to_dict() != self.cfg.to_dict():
            raise ValueError("Training resume requires the saved SACConfig; restoring optimizers with a different "
                             "discount, reward protocol, or learning rate would silently mix experiments")
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
