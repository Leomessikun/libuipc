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

from .history import RolloutHistory
from .models import Actor, CategoricalCritic, Critic, EncoderConfig, PrivilegedCritic, WangFlowActor, _sample_head, reuse_neighbourhoods
from .obs import FLAG_TOOL, ObsSpec
from .rlt import RLTConfig

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
    temperature_count: int = 1
    """Entropy temperatures, one per replay buffer as ``SAC_AWAC.py:633`` sizes ``log_alpha``; each update
    uses the temperature of the buffer its batch was drawn from, Wang's ``alpha[alpha_idx]``. One is the
    single temperature every checkpoint saved before this field carries."""
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
    trunk_style: str = "plain"
    """Shape of every head's body. ``plain`` is Linear-ReLU-Linear-ReLU-Linear, what this port has
    always used; ``residual`` is the pre-normalised residual arrangement value networks are reported
    to need before they benefit from scale at all — without normalisation a larger multi-layer
    perceptron critic gets worse, not better."""
    trunk_blocks: int = 2
    """Residual blocks per head under ``trunk_style = residual``; ignored otherwise."""
    critic_action_mode: str = "dense"
    """Where the action enters the point-cloud critic.

    ``dense`` is the reference's Q function: the action becomes a feature of every point and the
    encoder sees it. ``latent`` encodes the cloud alone and concatenates the action to the latent
    vector, which the reference measures at about 0.11 lower upper-arm dressed ratio over its three
    pose sub-ranges, and which this port used until 2026-09-13
    (``agent_docs/performance/2026-09-12-critic-architecture-defect.md``). Checkpoints written before
    then carry no such key and load as ``latent``."""
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
    history_length: int = 1
    """Frames the policy and critic condition on, the proposal's H. ``1`` is the single-frame network
    every earlier checkpoint stores. Above one, each frame is encoded on its own and the ordered frame
    vectors, their validity and the commands between them feed the trunk (``models.FrameHistory``);
    learning then draws padded windows from sequence replay, one learning step per window, and
    collection carries a raw-frame ``RolloutHistory`` per stream. Implemented for the scalar dense
    point critic without stochastic augmentation; other combinations are refused, not approximated."""
    history_kind: str = "frames"
    """How the frames of a window are read above one. ``frames`` concatenates them in order
    (``models.FrameHistory``) and learns one step per window, its last transition. ``rlt`` runs the
    recurrent looped transformer (``rlt.RecurrentLoopedHistory``) over them. Its learning positions
    and successor context are selected by ``rlt_learning_mode``. Collection uses the same sliding
    raw-frame window for both history kinds."""
    rlt: RLTConfig = field(default_factory=RLTConfig)
    """Width, depth, decoder window, memory groups, feedback scale and tying of the ``rlt`` history."""
    rlt_learning_mode: str = "endpoint"
    """``endpoint`` learns one transition with the same sliding H-frame context used to act,
    including a separately shifted H-frame Bellman successor. ``prefix`` retains the experimental
    all-position objective: shorter prefixes and an H+1 successor differ from rollout context.
    Old RLT checkpoints without this field restore ``prefix`` explicitly."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["encoder"] = self.encoder.to_dict()
        data["rlt"] = self.rlt.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SACConfig":
        data = dict(data)
        data["encoder"] = EncoderConfig.from_dict(data.get("encoder", {}))
        data["rlt"] = RLTConfig.from_dict(data.get("rlt", {}))
        if "rlt_learning_mode" not in data and data.get("history_kind") == "rlt":
            data["rlt_learning_mode"] = "prefix"
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
        history = int(cfg.history_length)
        if history < 1:
            raise ValueError("history_length must be at least 1")
        if cfg.history_kind not in ("frames", "rlt"):
            raise ValueError(f"Unknown history_kind {cfg.history_kind!r}; expected 'frames' or 'rlt'")
        if cfg.rlt_learning_mode not in ("endpoint", "prefix"):
            raise ValueError("rlt_learning_mode must be 'endpoint' or 'prefix'")
        if history == 1 and cfg.history_kind != "frames":
            raise ValueError("history_kind applies to a window; set history_length above 1 or leave the kind at 'frames'")
        if history > 1:
            if cfg.algo != "sac" or cfg.critic_input != "points" or cfg.critic_action_mode != "dense":
                raise ValueError("A frame history is implemented for the scalar dense point critic only; the flashsac, "
                                 "privileged and latent critics refuse it until their sequence updates exist")
            if cfg.random_shift_scale > 0.0 or cfg.point_jitter_scale > 0.0:
                raise ValueError("Stochastic observation augmentation is not temporally consistent across a window; "
                                 "disable it for a frame history")
        if cfg.actor_type == "wang-flow":
            actor_cls = WangFlowActor
        elif cfg.actor_type == "flat":
            actor_cls = Actor
        else:
            raise ValueError(f"Unknown actor_type {cfg.actor_type!r}")
        self.actor = actor_cls(
            spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra, cfg.actor_log_std_min,
            cfg.actor_log_std_max, cfg.trunk_style, cfg.trunk_blocks, history_length=history,
            history_kind=cfg.history_kind, rlt=cfg.rlt,
        ).to(self.device)
        if cfg.critic_input == "privileged":
            if cfg.algo != "sac" or int(cfg.privileged_dim) <= 0:
                raise ValueError("The privileged critic is the scalar 'sac' critic and needs privileged_dim > 0")
            make_critic = lambda: PrivilegedCritic(  # noqa: E731
                cfg.privileged_dim, action_dim, cfg.hidden_dim, cfg.trunk_style, cfg.trunk_blocks
            )
        elif cfg.critic_input != "points":
            raise ValueError(f"Unknown critic_input {cfg.critic_input!r}")
        elif cfg.algo == "sac":
            make_critic = lambda: Critic(  # noqa: E731
                spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.use_extra, cfg.critic_action_mode,
                cfg.trunk_style, cfg.trunk_blocks, history_length=history, history_kind=cfg.history_kind, rlt=cfg.rlt,
            )
        elif cfg.algo == "flashsac":
            make_critic = lambda: CategoricalCritic(  # noqa: E731
                spec, action_dim, cfg.hidden_dim, cfg.encoder, cfg.num_bins, cfg.min_v, cfg.max_v,
                cfg.use_extra, cfg.critic_action_mode, cfg.trunk_style, cfg.trunk_blocks,
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
        count = int(cfg.temperature_count)
        if count < 1:
            raise ValueError(f"temperature_count must be at least 1, got {count}")
        # One temperature stays the 0-d tensor every earlier checkpoint stores.
        self.log_alpha = torch.full(
            (count,) if count > 1 else (), float(np.log(cfg.init_temperature)), dtype=torch.float32, device=self.device
        )
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
        if teachers and int(self.cfg.history_length) != 1:
            raise ValueError("Teacher distillation onto a history-aware student is not implemented; "
                             "each teacher would need a window of its own")
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

    def _alpha_at(self, index: int) -> torch.Tensor:
        """The temperature of an update whose batch came from replay buffer ``index``; the one temperature when shared."""
        return self.alpha if self.log_alpha.dim() == 0 else self.alpha[int(index)]

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

    def make_history(self, num_streams: int) -> RolloutHistory:
        """Rollout state for ``num_streams`` streams. State lives outside the network: training
        collection, each evaluation world and each teacher own one, and reset it themselves."""
        return RolloutHistory(num_streams, self.spec.dim, self.action_dim, self.cfg.history_length)

    def _unpack_window(self, obs: torch.Tensor, valid: torch.Tensor, commands: torch.Tensor):
        """``obs [B,L,D]``, ``valid [B,L]``, ``commands [B,L-1,A]`` -> the heads' ``(frames, valid, commands)``."""
        return self._unpack(obs.reshape(-1, self.spec.dim)), valid, commands

    def act(self, obs: np.ndarray, deterministic: bool, history: RolloutHistory | None = None) -> np.ndarray:
        """Act on one observation per stream. A history-aware policy reads ``history`` and records the
        decision into it; the caller resets the streams whose physics ended."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, self.spec.dim)
        to = lambda value: torch.as_tensor(value, device=self.device)  # noqa: E731
        with torch.no_grad():
            if self.actor.history is None:
                batch = self._unpack(to(obs))
            elif history is None:
                raise ValueError("A history-aware policy needs its rollout state; pass the RolloutHistory of these streams")
            else:
                frames, commands, valid = history.window(obs)
                batch = self._unpack_window(to(frames), to(valid), to(commands))
            if deterministic:
                mu, _, _, _ = self.actor(batch, compute_pi=False, compute_log_pi=False)
                action = mu.cpu().numpy()
            else:
                _, pi, _, _ = self.actor(batch, compute_log_pi=False)
                action = pi.cpu().numpy()
        if history is not None:
            history.push(obs, action)
        return action

    # ------------------------------------------------------------------
    def _update_critic(self, obs, action, reward, next_obs, not_done, state=None, next_state=None, index: int = 0) -> dict:
        """``state`` and ``next_state``, when given, are what the critic reads; the actor always reads observations.
        ``index`` is the replay buffer the batch came from, whose temperature the soft target uses."""
        with torch.no_grad():
            _, next_pi, next_log_pi, _ = self.actor(next_obs)
            target_q1, target_q2 = self.critic_target(next_obs if next_state is None else next_state, next_pi)
            target_v = torch.min(target_q1, target_q2) - self._alpha_at(index).detach() * next_log_pi
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

    def _update_critic_categorical(self, obs, action, reward, next_obs, not_done, index: int = 0) -> dict:
        """FlashSAC-style update: project the soft Bellman target onto the atom grid, cross-entropy loss."""
        with torch.no_grad():
            _, next_pi, next_log_pi, _ = self.actor(next_obs)
            tq1, tq2, tlp1, tlp2 = self.critic_target(next_obs, next_pi)
            target_min = torch.minimum(tq1, tq2)
            target_scalar = reward + not_done * self.cfg.discount * (target_min - self._alpha_at(index).detach() * next_log_pi)
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

    def _update_actor_and_alpha(self, obs, state=None, label=None, index: int = 0) -> dict:
        mu, pi, log_pi, log_std = self.actor(obs)
        with frozen_parameters(self.critic):
            # Detaching the critic's encoder saves a backward pass through it, but only the latent
            # critic can afford it: under the reference's dense critic the action is a feature of
            # every point, so the path from the action to Q runs *through* the encoder and detaching
            # would leave Q constant in pi, reducing the actor to entropy maximisation.
            detach = getattr(self.critic, "action_mode", "latent") != "dense"
            q1, q2 = self._critic_scalar(obs if state is None else state, pi, detach_encoder=detach)
        # Wang's alpha[alpha_idx]: the actor loss and the temperature loss use the batch's buffer's temperature.
        alpha = self._alpha_at(index)
        actor_loss = (alpha.detach() * log_pi - torch.min(q1, q2)).mean()
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
            alpha_loss = (alpha * (-log_pi - self.target_entropy).detach()).mean()
            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            if self.cfg.min_alpha > 0.0:
                with torch.no_grad():
                    self.log_alpha.clamp_(min=float(np.log(self.cfg.min_alpha)))
            stats["alpha_loss"] = float(alpha_loss.item())
        stats["alpha"] = float(self._alpha_at(index).item())
        if self.log_alpha.dim():
            stats[f"alpha_{int(index)}"] = stats["alpha"]
        return stats

    def _clip(self, module: torch.nn.Module) -> float:
        max_norm = float(self.cfg.grad_clip_max_norm) if self.cfg.grad_clip_max_norm > 0 else float("inf")
        return float(torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_norm))

    def update(self, replay) -> dict:
        """One optimizer update following the Wang actor/target schedule."""
        with reuse_neighbourhoods():
            return self._update(replay)

    def _sample_single(self, replay):
        batch = replay.sample(self.cfg.batch_size)
        obs_flat, action, reward, next_obs_flat, not_done = batch[:5]
        rest = list(batch[5:])
        # A replay set appends the index of the buffer its batch came from: Wang's alpha_idx.
        indexed = bool(getattr(replay, "indexed", False))
        index = int(rest.pop()) if indexed else 0
        if self.log_alpha.dim() and not indexed:
            raise ValueError("One temperature per replay buffer needs a replay set that reports which buffer each batch came from")
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
        return obs, action, reward, next_obs, not_done, state, next_state, label, index

    def _sample_windows(self, replay):
        """One learning step per window: its last transition, seen through ``history_length`` frames.

        The successor window advances the history with the command actually recorded in replay
        before the current policy proposes its next candidate, so the observed past is never
        rewritten. Windows are padded at episode openings, where a deployed policy also starts empty.
        """
        if not getattr(replay, "sequence", False):
            raise ValueError("A history-aware policy learns from sequence replay; record with --sequence-replay")
        indexed = bool(getattr(replay, "indexed", False))
        if self.log_alpha.dim() and not indexed:
            raise ValueError("One temperature per replay buffer needs a replay set that reports which buffer each batch came from")
        length = int(self.cfg.history_length)
        batch = replay.sample_sequences(length, self.cfg.batch_size, pad=True, strict_context=True)
        index = int(batch.buffer_index) if indexed else 0
        obs = self._unpack_window(batch.obs[:, :length], batch.valid, batch.actions[:, :-1])
        next_valid = torch.cat([batch.valid[:, 1:], torch.ones_like(batch.valid[:, :1])], dim=1)
        next_obs = self._unpack_window(batch.obs[:, 1:], next_valid, batch.actions[:, 1:])
        label = batch.labels[:, -1] if getattr(replay, "labelled", False) else None
        return obs, batch.actions[:, -1], batch.rewards[:, -1], next_obs, batch.not_dones[:, -1], None, None, label, index

    def _update_rlt(self, replay) -> dict:
        """One update on windows whose every recorded position is a learning step.

        The report's replay contract (5.3–5.4) inside a window: the recorded run rebuilds every state
        from the window's first frame under the current parameters, the current policy's candidate at
        position ``t`` branches from the recorded state of ``t-1`` and touches only frame ``t``'s
        encoding, and the Bellman successor of ``t`` is the recorded run at ``t+1`` — advanced with the
        command actually recorded, never with a fresh one. The window start is the declared truncation
        of the recurrence; nothing inside the window is detached. Losses are averaged over the recorded
        positions, so padded openings neither learn nor dilute.
        """
        if not getattr(replay, "sequence", False):
            raise ValueError("A history-aware policy learns from sequence replay; record with --sequence-replay")
        indexed = bool(getattr(replay, "indexed", False))
        if self.log_alpha.dim() and not indexed:
            raise ValueError("One temperature per replay buffer needs a replay set that reports which buffer each batch came from")
        length = int(self.cfg.history_length)
        batch = replay.sample_sequences(length, self.cfg.batch_size, pad=True)
        index = int(batch.buffer_index) if indexed else 0
        # The window's L transitions plus the saved successor: L+1 frames, the last always recorded.
        frames = self._unpack(batch.obs.reshape(-1, self.spec.dim))
        recorded = batch.valid.bool()
        valid = torch.cat([recorded, torch.ones_like(recorded[:, :1])], dim=1)
        # The successor's own command is not recorded; its zero is never read by a branch before it.
        commands = torch.cat([batch.actions, torch.zeros_like(batch.actions[:, :1])], dim=1)
        learn = recorded[..., None].to(batch.rewards.dtype)
        weight = 1.0 / learn.sum().clamp_min(1.0)
        alpha = self._alpha_at(index).detach()

        actor_step = (self.updates + 1) % self.cfg.actor_update_freq == 0
        with torch.set_grad_enabled(actor_step):
            mu, log_std = self.actor.head_sequence(frames, valid, batch.actions)
            _, pi, log_pi, _ = _sample_head(mu, log_std, True, True)
        with torch.no_grad():
            target_run = self.critic_target.run_sequence(frames, valid, commands)
            target_q1, target_q2 = self.critic_target.q_branch(target_run, frames, valid, pi)
            target_v = torch.min(target_q1, target_q2)[:, 1:] - alpha * log_pi[:, 1:]
            target_q = batch.rewards + batch.not_dones * self.cfg.discount * target_v
            bound = float(self.cfg.reward_abs_bound) / max(1.0 - float(self.cfg.discount), 1e-6)
            target_q = target_q.clamp(-bound, bound)
        run = self.critic.run_sequence(frames, valid, commands)
        current_q1, current_q2 = self.critic.q_sequence(run)
        current_q1, current_q2 = current_q1[:, :length], current_q2[:, :length]
        critic_loss = (((current_q1 - target_q) ** 2 + (current_q2 - target_q) ** 2) * learn).sum() * weight
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = self._clip(self.critic)
        self.critic_optimizer.step()
        stats = {
            "critic_loss": float(critic_loss.item()), "q1_mean": float((current_q1 * learn).sum().item() * weight),
            "critic_grad_norm": grad_norm, "batch_reward": float((batch.rewards * learn).sum().item() * weight),
            "learning_positions": int(learn.sum().item()),
        }
        self.updates += 1
        if self.updates % self.cfg.actor_update_freq == 0:
            with frozen_parameters(self.critic):
                # The recorded prefix under the critic just stepped; the candidate's path to Q runs
                # through frame t's encoding, as in the single-frame dense critic, so it is not detached.
                with torch.no_grad():
                    run_now = self.critic.run_sequence(frames, valid, commands)
                q1, q2 = self.critic.q_branch(run_now, frames, valid, pi)
            actor_loss = ((alpha * log_pi - torch.min(q1, q2))[:, :length] * learn).sum() * weight
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = self._clip(self.actor)
            self.actor_optimizer.step()
            entropy = -(log_pi[:, :length] * learn).sum() * weight
            stats.update({"actor_loss": float(actor_loss.item()), "entropy": float(entropy.item()), "actor_grad_norm": actor_grad_norm})
            if not self.cfg.alpha_fixed:
                live_alpha = self._alpha_at(index)
                alpha_loss = (live_alpha * ((-log_pi[:, :length] - self.target_entropy).detach() * learn)).sum() * weight
                self.log_alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.log_alpha_optimizer.step()
                if self.cfg.min_alpha > 0.0:
                    with torch.no_grad():
                        self.log_alpha.clamp_(min=float(np.log(self.cfg.min_alpha)))
                stats["alpha_loss"] = float(alpha_loss.item())
            stats["alpha"] = float(self._alpha_at(index).item())
            if self.log_alpha.dim():
                stats[f"alpha_{int(index)}"] = stats["alpha"]
        if self.updates % self.cfg.critic_target_update_freq == 0:
            soft_update(self.critic.Q1, self.critic_target.Q1, self.cfg.critic_tau)
            soft_update(self.critic.Q2, self.critic_target.Q2, self.cfg.critic_tau)
            soft_update(self.critic.encoder, self.critic_target.encoder, self.cfg.encoder_tau)
            soft_update(self.critic.history, self.critic_target.history, self.cfg.encoder_tau)
        return stats

    def _update(self, replay) -> dict:
        if (int(self.cfg.history_length) > 1 and self.cfg.history_kind == "rlt"
                and self.cfg.rlt_learning_mode == "prefix"):
            return self._update_rlt(replay)
        sample = self._sample_windows if int(self.cfg.history_length) > 1 else self._sample_single
        obs, action, reward, next_obs, not_done, state, next_state, label, index = sample(replay)
        if self.cfg.algo == "flashsac":
            stats = self._update_critic_categorical(obs, action, reward, next_obs, not_done, index)
        else:
            stats = self._update_critic(obs, action, reward, next_obs, not_done, state, next_state, index)
        stats["batch_reward"] = float(reward.mean().item())
        stats["learning_positions"] = int(action.shape[0])
        self.updates += 1
        if self.updates % self.cfg.actor_update_freq == 0:
            stats.update(self._update_actor_and_alpha(obs, state, label, index))
        if self.updates % self.cfg.critic_target_update_freq == 0:
            soft_update(self.critic.Q1, self.critic_target.Q1, self.cfg.critic_tau)
            soft_update(self.critic.Q2, self.critic_target.Q2, self.cfg.critic_tau)
            if self.critic.encoder is not None:
                soft_update(self.critic.encoder, self.critic_target.encoder, self.cfg.encoder_tau)
            if getattr(self.critic, "history", None) is not None:
                soft_update(self.critic.history, self.critic_target.history, self.cfg.encoder_tau)
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
        if self.cfg.trunk_style != "plain":
            # Absent, the key means the plain trunk, so earlier checkpoints still load.
            protocol.update(trunk_style=str(self.cfg.trunk_style), trunk_blocks=int(self.cfg.trunk_blocks))
        if self.cfg.critic_input == "points" and self.cfg.critic_action_mode != "latent":
            # Absent, the key means the old latent critic, so checkpoints from before the fix still load.
            protocol.update(critic_action_mode=str(self.cfg.critic_action_mode))
        if int(self.cfg.history_length) != 1:
            # Absent, the key means the single-frame policy, so earlier checkpoints still load; a
            # consumer that cannot carry rollout state must refuse the key before building a world.
            protocol.update(history_length=int(self.cfg.history_length))
            if self.cfg.history_kind != "frames":
                # Absent, the key means the ordered frame concatenation, so H-frame checkpoints still load.
                protocol.update(history_kind=str(self.cfg.history_kind), rlt=self.cfg.rlt.to_dict())
                if self.cfg.rlt_learning_mode != "prefix":
                    protocol.update(rlt_learning_mode=str(self.cfg.rlt_learning_mode))
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
