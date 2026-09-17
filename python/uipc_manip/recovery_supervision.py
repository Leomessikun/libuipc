"""Verified IPC actions as an auxiliary loss inside the SAC actor update.

The critic still learns ordinary soft Bellman targets from real replay
transitions. Observation/action-only teacher files are never invented into TD
transitions. No simulator, teacher or extra input is needed by the exported actor.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .distill import environment_contract, load_dataset


class RecoveryActionSupervision:
    def __init__(self, obs, actions, *, weight, active_axes, seed, device):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Recovery weight must be finite and nonnegative")
        self.obs = torch.as_tensor(obs, dtype=torch.float32, device=device).detach()
        self.actions = torch.as_tensor(actions, dtype=torch.float32, device=device).detach()
        self.active_axes = tuple(active_axes)
        if (self.obs.ndim != 2 or self.actions.ndim != 2 or not len(self.obs)
                or len(self.obs) != len(self.actions) or not self.active_axes
                or len(set(self.active_axes)) != len(self.active_axes)
                or min(self.active_axes) < 0 or max(self.active_axes) >= self.actions.shape[1]
                or not torch.isfinite(self.obs).all() or not torch.isfinite(self.actions).all()
                or self.actions.abs().max() > 1 + 1e-6):
            raise ValueError("Invalid recovery observations, actions or active axes")
        self.weight, self.seed = float(weight), int(seed)
        self.generator = torch.Generator(device=self.obs.device)

    def validate(self, agent):
        cfg = agent.cfg
        if (cfg.history_length != 1 or cfg.actor_type == "state" or cfg.algo != "sac"
                or cfg.critic_input != "points" or cfg.random_shift_scale or cfg.point_jitter_scale
                or cfg.physics_actor_weight or cfg.adjoint_weight or agent.teachers):
            raise ValueError("Recovery supervision requires single-frame, unaugmented point SAC without other teachers")
        if (self.obs.shape[1] != agent.spec.dim or self.actions.shape[1] != agent.action_dim
                or self.obs.device != next(agent.actor.parameters()).device):
            raise ValueError("Recovery layout/device differs from SAC")

    def loss(self, agent):
        if self.weight == 0:
            return self.obs.new_zeros(()), {"recovery_action_loss": 0.0, "recovery_rows": 0,
                                          "recovery_weight": 0.0}
        # A separate, update-indexed RNG preserves SAC's action noise and makes
        # sampling reproducible on resume without storing another RNG state.
        self.generator.manual_seed((self.seed + agent.updates) % (2**63 - 1))
        rows = torch.randint(len(self.obs), (agent.cfg.batch_size,), device=self.obs.device,
                             generator=self.generator)
        target = self.actions[rows]
        mu, _, _, _ = agent.actor(agent._unpack(self.obs[rows]), compute_pi=False, compute_log_pi=False)
        mse = (mu[:, self.active_axes] - target[:, self.active_axes]).square().mean()
        return self.weight * mse, {"recovery_action_loss": float(mse.detach()),
                                  "recovery_rows": len(rows), "recovery_weight": self.weight}


def load_recovery_supervision(source_dirs, *, agent, env, eval_bodies, weight, seed):
    """Admit only completed, verified continuations; reject held-out body leakage."""
    for source in source_dirs:
        source = Path(source)
        manifest = json.loads((source / "manifest.json").read_text())
        result = json.loads((source / "result.json").read_text())
        if (manifest.get("admission_rule") != "verified_full_continuation_v1"
                or manifest.get("completed") is not True or result.get("admitted") is not True):
            raise ValueError("Recovery data must pass completed full-continuation verification")
    obs, actions, _, _, manifests, summary = load_dataset(
        list(source_dirs), val_ratio=0, seed=seed, max_train_transitions=0, min_upperarm_ratio=None)
    if any(environment_contract(m["env"]) != environment_contract(env) for m in manifests):
        raise ValueError("Recovery environment contract differs from SAC pretraining")
    if any(body < 0 or garment == "unknown" for garment, body in summary["train_cells"]):
        raise ValueError("Recovery records need garment and body identity")
    if {int(body) for _, body in summary["train_cells"]} & set(eval_bodies):
        raise ValueError("Recovery training bodies overlap the evaluation split")
    axes = (0, 1, 2, 4, 5) if env.get("clip_rotation_to_yz") else tuple(range(agent.action_dim))
    supervision = RecoveryActionSupervision(obs, actions, weight=weight, active_axes=axes,
                                           seed=seed, device=agent.device)
    supervision.validate(agent)
    # The dataset loader's train fallback is not independent validation.
    summary = {k: v for k, v in summary.items() if not k.startswith("val_") and k != "split_kind"}
    return supervision, {**summary, "weight": float(weight), "active_axes": list(axes),
                         "objective": "SAC actor + verified-action MSE; unchanged SAC critic",
                         "sampling": "one independent recovery batch per scheduled actor update",
                         "preload_device": str(supervision.obs.device)}
