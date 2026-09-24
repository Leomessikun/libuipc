"""Explicit, recorded rollout interventions for checkpoint transfer experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class ControlProfile:
    name: str = "legacy"
    bridge_voxel: float | None = None
    force_source: str = "zero"
    force_scale: float = 1.
    force_clip: float = .04
    force_ema: float = 1.
    cloth_crop_below_arm_m: float | None = None
    tracking_budget_m: float | None = None
    rotation_gain: float | None = None
    grasp_strength_gain: float = 1.
    anchor_count: int | None = None
    lookahead_interval: int = 0

    def __post_init__(self):
        if not re.fullmatch(r"[a-z0-9_]+", self.name):
            raise ValueError("Profile names must contain lowercase letters, digits or underscores")
        if self.force_source not in ("zero", "body_contact", "gripper"):
            raise ValueError(f"Unknown force source: {self.force_source}")
        if not np.isfinite(self.force_scale) or self.force_scale < 0:
            raise ValueError("force_scale must be finite and nonnegative")
        if not np.isfinite(self.force_clip) or self.force_clip <= 0:
            raise ValueError("force_clip must be finite and positive")
        if not 0 < self.force_ema <= 1:
            raise ValueError("force_ema must be in (0, 1]")
        if not np.isfinite(self.grasp_strength_gain) or self.grasp_strength_gain <= 0:
            raise ValueError("grasp_strength_gain must be finite and positive")
        if self.anchor_count is not None and (not isinstance(self.anchor_count, int) or self.anchor_count < 3):
            raise ValueError("anchor_count must be an integer >=3")
        if not isinstance(self.lookahead_interval, int) or self.lookahead_interval < 0:
            raise ValueError("lookahead_interval must be an integer >=0")
        for key in ("bridge_voxel", "cloth_crop_below_arm_m", "tracking_budget_m", "rotation_gain"):
            value = getattr(self, key)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {key}")
        if self.tracking_budget_m == 0:
            raise ValueError("tracking_budget_m must be positive")


def load_profiles(path: Path | None, count: int):
    if path is None:
        return [ControlProfile()] * count
    data = json.loads(path.read_text())
    if not isinstance(data, list) or len(data) != count:
        raise ValueError("Need one profile per --variants entry, before replication")
    allowed = {f.name for f in fields(ControlProfile)}
    for value in data:
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError(f"Unknown profile fields: {value}")
    return [ControlProfile(**value) for value in data]


def profile_dict(profile):
    return asdict(profile)


def force_input(profile, body_force, gripper_force, previous):
    """Checkpoint-unit input, distinct from recorded physical forces in newtons.

    IPC body_force is the load on the body. The released PyBullet model takes
    the opposite contact reaction on cloth. These scales are an experimental
    adapter, not a calibration of PyBullet's force units to newtons.
    """
    if profile.force_source == "zero":
        return np.zeros(3, np.float32)
    raw = -np.asarray(body_force) if profile.force_source == "body_contact" else np.asarray(gripper_force)
    value = raw * profile.force_scale
    value *= min(1., profile.force_clip / max(np.linalg.norm(value), 1e-12))
    return (profile.force_ema * value + (1 - profile.force_ema) * previous).astype(np.float32)


def crop_observation(observation, spec, profile):
    """Apply the released PB cloth-height crop without hiding additional arm points."""
    if profile.cloth_crop_below_arm_m is None:
        return observation
    result = observation.copy()
    pos, feat, valid, _ = spec.unpack_numpy(result[None])
    arm = (feat[0, :, 1] > .5) & valid[0].astype(bool)
    cloth = (feat[0, :, 0] > .5) & valid[0].astype(bool)
    if arm.any():
        low = cloth & (pos[0, :, 2] < pos[0, arm, 2].mean() - profile.cloth_crop_below_arm_m)
        # unpack_numpy returns views; zero features make these points padding.
        feat[0, low] = 0.
    return result


def tracking_scale(action, *, anchor, offsets, held, max_translation, max_rotation, budget):
    """Limit proposed held-patch displacement using current cloth geometry.

    This is not a future-physics predictor: gravity and contact can still change
    tracking during the step, so the existing measured grasp check remains.
    """
    if budget is None:
        return 1.
    action = np.asarray(action, float)
    vector = action[3:] * max_rotation
    angle = np.linalg.norm(vector)
    axis = vector / max(angle, 1e-12)
    for scale in (1., .5, .25, .125, .0625, 0.):
        theta = angle * scale
        rotated = (offsets * np.cos(theta) + np.cross(axis, offsets) * np.sin(theta)
                   + (offsets @ axis)[:, None] * axis * (1 - np.cos(theta)))
        targets = anchor + action[:3] * max_translation * scale + rotated
        if np.linalg.norm(targets - held, axis=1).max() <= budget:
            return scale
    return 0.


def rigid_rotation(before, after):
    """Commanded tool rotation, from its original and current held-patch offsets."""
    u, _, vt = np.linalg.svd(np.asarray(before).T @ np.asarray(after))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    return vt.T @ correction @ u.T
