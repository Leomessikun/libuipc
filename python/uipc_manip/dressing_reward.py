"""Wang RSS 2023 dressing reward and dressed-ratio metrics, ported from the Newton teacher.

This is ``mdp/rewards.py`` of the Newton branch with ``reward_wang_reference``
set and the Wang ``pointcloud_3`` weights applied: the task term is the arm
progress of the sleeve opening measured by line-triangle intersection, the
upper arm is worth five times the forearm, a small collision penalty fires
when the cuff touches the body, and a centre-alignment term rewards keeping
the opening centred on the upper arm. No force, topology, coverage, or
strain terms enter the reward; they were Newton-era additions the FMVP
preset switches off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WangRewardConfig:
    task_w: float = 1.0
    upper_w: float = 5.0
    collision_w: float = 0.01
    collision_threshold: float = 0.01
    cloth_particle_radius: float = 0.0045
    center_align_reward_w: float = 0.02
    center_align_penalty_w: float = 0.05
    near_center_range: float = 0.03
    far_center_range: float = 0.075
    reward_min_clamp: float = -100.0
    success_upperarm_ratio: float = 0.7


@dataclass
class DressingProgress:
    reward: float
    task_reward: float
    on_forearm: bool
    on_upperarm: bool
    forearm_distance: float
    upperarm_distance: float
    forearm_ratio: float
    upperarm_ratio: float
    collision: float
    center_align: float


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else fallback


def line_triangles(origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Moller-Trumbore ray-triangle test (Newton ``_line_triangles``)."""
    eps = 1e-9
    d = _unit(np.asarray(direction, dtype=np.float64), np.array([1.0, 0.0, 0.0]))
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = b - a, c - a
    h = np.cross(d[None, :], e2)
    det = np.sum(e1 * h, axis=1)
    valid = np.abs(det) >= eps
    inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
    s = origin[None, :] - a
    u = inv * np.sum(s * h, axis=1)
    valid &= (u >= -eps) & (u <= 1.0 + eps)
    q = np.cross(s, e1)
    v = inv * np.sum(d[None, :] * q, axis=1)
    valid &= (v >= -eps) & (u + v <= 1.0 + eps)
    t = inv * np.sum(e2 * q, axis=1)
    valid &= t >= -eps
    points = origin[None, :] + t[:, None] * d[None, :]
    return valid, points


def wang_progress(
    cloth: np.ndarray,
    *,
    polygon_idx: np.ndarray,
    triangle_idx: np.ndarray,
    cuff_idx: np.ndarray,
    finger: np.ndarray,
    elbow: np.ndarray,
    shoulder: np.ndarray,
    human_points: np.ndarray,
    cfg: WangRewardConfig,
) -> DressingProgress:
    """Per-environment Wang reference reward and dressed distances."""
    polygon = cloth[polygon_idx]
    center = polygon.mean(axis=0)
    triangles = cloth[triangle_idx]

    forearm_axis = finger - elbow
    forearm_len = max(float(np.linalg.norm(forearm_axis)), 1e-6)
    forearm_dir = _unit(forearm_axis, np.array([1.0, 0.0, 0.0]))
    upper_dir = _unit(elbow - shoulder, np.array([1.0, 0.0, 0.0]))
    upper_len = max(float(np.linalg.norm(shoulder - elbow)), 1e-6)

    task = -float(np.linalg.norm(finger - center))
    forearm_distance = 0.0
    upperarm_distance = 0.0

    # Forearm: ray from the elbow toward the finger; progress is measured back from the finger.
    fa_hit, fa_pts = line_triangles(elbow, forearm_dir, triangles)
    fa_progress = np.sum((fa_pts - finger[None, :]) * (-forearm_dir)[None, :], axis=1)
    fa_valid = fa_hit & (fa_progress >= 0.0)
    on_forearm = bool(fa_valid.any())
    if on_forearm:
        # The reference takes the first valid triangle on the forearm.
        first = int(np.argmax(fa_valid))
        forearm_distance = float(min(fa_progress[first], forearm_len))
        task = forearm_distance

    # Upper arm: ray from the shoulder toward the elbow; progress measured back from the elbow.
    up_hit, up_pts = line_triangles(shoulder, upper_dir, triangles)
    up_progress = np.sum((up_pts - elbow[None, :]) * (-upper_dir)[None, :], axis=1)
    up_valid = up_hit & (up_progress >= 0.0)
    on_upperarm = bool(up_valid.any())
    upper_intersection = center
    if on_upperarm:
        # The reference takes the minimum over all valid triangles on the upper arm.
        masked = np.where(up_valid, up_progress, np.inf)
        best = int(np.argmin(masked))
        upperarm_distance = float(up_progress[best])
        upper_intersection = up_pts[best]
        forearm_distance = forearm_len
        task = forearm_len + cfg.upper_w * upperarm_distance

    cuff = cloth[cuff_idx]
    cuff_distance = float(np.min(np.linalg.norm(cuff[:, None, :] - human_points[None, :, :], axis=2)))
    collision = -1.0 if cuff_distance < cfg.cloth_particle_radius + cfg.collision_threshold else 0.0

    center_align = 0.0
    if on_upperarm:
        d_center = float(np.linalg.norm(center - upper_intersection))
        if d_center < cfg.near_center_range:
            center_align = cfg.center_align_reward_w
        elif d_center > cfg.far_center_range:
            center_align = -cfg.center_align_penalty_w

    reward = cfg.task_w * task + cfg.collision_w * collision + center_align
    reward = max(reward, cfg.reward_min_clamp)
    return DressingProgress(
        reward=float(reward),
        task_reward=float(task),
        on_forearm=on_forearm,
        on_upperarm=on_upperarm,
        forearm_distance=float(forearm_distance),
        upperarm_distance=float(upperarm_distance),
        forearm_ratio=float(np.clip(forearm_distance / forearm_len, 0.0, 1.0)),
        upperarm_ratio=float(np.clip(upperarm_distance / upper_len, 0.0, 1.0)),
        collision=float(collision),
        center_align=float(center_align),
    )


def opening_threaded(cloth: np.ndarray, opening_idx: np.ndarray, finger: np.ndarray, shoulder: np.ndarray) -> tuple[bool, float]:
    """Winding test: does the finger-to-shoulder axis pass through the opening ring?

    Projects the ring into the plane normal to the arm axis at the axis point
    nearest the ring centroid; the axis is enclosed when the projected points
    surround the origin, that is when the largest angular gap is below 180
    degrees. Returns the enclosure flag and the fraction along the axis.
    """
    p = cloth[opening_idx]
    axis = shoulder - finger
    length = float(np.linalg.norm(axis))
    if p.shape[0] < 3 or length < 1e-9:
        return False, 0.0
    axis = axis / length
    c = p.mean(axis=0)
    t_raw = float((c - finger) @ axis) / length
    if not 0.0 <= t_raw <= 1.0:
        # The ring must sit between the fingertip and the shoulder; a ring hovering in
        # front of the hand encloses the extended axis line without being on the arm.
        return False, float(np.clip(t_raw, 0.0, 1.0))
    t = t_raw
    a0 = finger + (t * length) * axis
    rel = p - a0
    rel = rel - np.outer(rel @ axis, axis)
    e1 = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = e1 - (e1 @ axis) * axis
    e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(axis, e1)
    ang = np.sort(np.arctan2(rel @ e2, rel @ e1))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2.0 * np.pi]))
    return bool(gaps.max() < np.pi), t
