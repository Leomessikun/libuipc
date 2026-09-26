"""Small CPU registration and explicit information controls for the motion pilot.

The covariance-weighted registration follows the GICP objective. This module
does not implement Sun et al.'s diffused scalar field or diffusion policy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _skew_batch(points):
    out = np.zeros((len(points), 3, 3))
    x, y, z = np.asarray(points).T
    out[:, 0, 1], out[:, 0, 2] = -z, y
    out[:, 1, 0], out[:, 1, 2] = z, -x
    out[:, 2, 0], out[:, 2, 1] = -y, x
    return out


def _covariances(points):
    _, indices = cKDTree(points).query(points, k=min(16, len(points)))
    neighbors = points[indices]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / neighbors.shape[1]
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, np.maximum(values[:, -1:] * .01, 1e-8))
    return np.einsum("nij,nj,nkj->nik", vectors, values, vectors)


def gicp(source, target, *, iterations=15, max_distance_m=.08):
    """Return (4x4 source-to-target transform, registration diagnostics)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    identity = np.eye(4)
    for p in (source, target):
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("GICP expects finite Nx3 point arrays")
    if min(len(source), len(target)) < 12:
        return identity, dict(valid=False, reason="too_few_visible_points", matches=0)
    if iterations < 1 or not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("Positive iterations and correspondence distance required")
    cov_s, cov_t = _covariances(source), _covariances(target)
    tree = cKDTree(target)
    r, t = np.eye(3), np.zeros(3)
    for _ in range(iterations):
        moved = source @ r.T + t
        distance, ids = tree.query(moved)
        mask = distance < max_distance_m
        if mask.sum() < 12:
            return identity, dict(valid=False, reason="too_few_correspondences", matches=int(mask.sum()))
        cov = cov_t[ids[mask]] + r[None] @ cov_s[mask] @ r.T[None]
        w = np.linalg.inv(cov)
        residual = moved[mask] - target[ids[mask]]
        jac = np.concatenate((np.broadcast_to(np.eye(3), (mask.sum(), 3, 3)), -_skew_batch(moved[mask])), axis=2)
        hessian = np.einsum("nai,nab,nbj->ij", jac, w, jac)
        grad = np.einsum("nai,nab,nb->i", jac, w, residual)
        delta = np.linalg.solve(hessian + np.eye(6) * max(np.trace(hessian) * 1e-10, 1e-9), -grad)
        if not np.isfinite(delta).all():
            return identity, dict(valid=False, reason="nonfinite_update", matches=int(mask.sum()))
        # Large rotations indicate a registration failure for this short-step use.
        if np.linalg.norm(delta[3:]) > .5:
            return identity, dict(valid=False, reason="excess_rotation", matches=int(mask.sum()))
        dr = Rotation.from_rotvec(delta[3:]).as_matrix()
        r, t = dr @ r, dr @ t + delta[:3]
        if np.linalg.norm(delta) < 1e-6:
            break
    distance, _ = tree.query(source @ r.T + t)
    matched = distance < max_distance_m
    if matched.sum() < 12:
        return identity, dict(valid=False, reason="too_few_final_correspondences", matches=int(matched.sum()))
    identity[:3, :3], identity[:3, 3] = r, t
    return identity, dict(valid=True, matches=int(matched.sum()),
                          rms_m=float(np.sqrt(np.mean(distance[matched] ** 2))))


def observed_arm_roi(points, tool, *, max_points=128):
    """Spatial ROI from already visible arm points; no simulator correspondence."""
    points = np.asarray(points, float)
    if max_points < 12:
        raise ValueError("Registration ROI must allow at least 12 points")
    if len(points) <= max_points:
        return points.copy()
    near = np.argsort(np.linalg.norm(points - np.asarray(tool), axis=1))[:2 * max_points]
    return points[near[np.linspace(0, len(near) - 1, max_points, dtype=int)]]


def bounded_correction(transform, point, max_displacement_m=.01):
    point = np.asarray(point, float)
    d = np.asarray(transform)[:3, :3] @ point + np.asarray(transform)[:3, 3] - point
    if max_displacement_m <= 0:
        raise ValueError("Correction bound must be positive")
    return d * min(1., max_displacement_m / max(np.linalg.norm(d), 1e-12))


def pause_masks(motion, decision_times, *, onset_s, horizon_s=.4, threshold_m=.02,
                speed=1., yoked_shift=1):
    """Return oracle, causal-present, and shifted-oracle diagnostic schedules.

    The causal schedule uses exact present/past finger positions, a stronger
    perception control than estimated velocity. The yoked schedule is an
    offline control with exactly the same pause count, not a deployable policy.
    """
    times = np.asarray(decision_times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] < 0
            or not np.isfinite([onset_s, horizon_s, threshold_m, speed]).all()
            or np.any(np.diff(times) <= 0) or horizon_s <= 0
            or threshold_m <= 0 or speed <= 0 or onset_s < 0):
        raise ValueError("Invalid pause schedule times or thresholds")
    def finger(t):
        return motion.sample(max(0., t - onset_s) * speed)[2][0]
    now = np.stack([finger(t) for t in times])
    future = np.stack([finger(t + horizon_s) for t in times])
    oracle = np.linalg.norm(future - now, axis=1) >= threshold_m
    velocity = np.zeros_like(now)
    velocity[1:] = np.diff(now, axis=0) / np.diff(times)[:, None]
    causal = np.linalg.norm(velocity * horizon_s, axis=1) >= threshold_m
    return {"oracle_pause": oracle, "causal_pause": causal,
            "yoked_pause": np.roll(oracle, int(yoked_shift))}
