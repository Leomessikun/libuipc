"""Visible segmented point cloud for the dressing observation.

The three geometric helpers are ported from the Newton dressing teacher's
``mdp/observations.py`` (Apache-2.0, The Newton Developers) so the observation
matches the reference: two virtual depth cameras derived from the arm
landmarks, z-buffer visibility of the arm and cloth together, and Open3D-style
voxel-centroid downsampling at Wang's 6.25 cm voxel. Camera pose jitter,
camera dropout, and local patch dropout reproduce the FMVP preset's
augmentations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DressingObsConfig:
    mode: str = "visible_dual"
    camera_distance_m: float = 1.0
    camera_height_m: float = 0.35
    camera_side: float = 1.0
    image_wh: int = 96
    depth_tolerance_m: float = 0.02
    fov_deg: float = 70.0
    voxel_size_m: float = 0.0625
    pose_jitter_m: float = 0.03
    camera_dropout_p: float = 0.1
    dropout_patches: int = 2
    dropout_radius_m: float = 0.06


def sample_segmented_cloud(
    arm: np.ndarray, cloth: np.ndarray, budget: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly subsample each segment without discarding the garment.

    Voxel centroids are spatially sorted, so prefix truncation hides one side
    of the scene. Allocate proportionally, retaining both nonempty segments
    when at least two points fit. A one-point budget prioritizes the cloth.
    Clouds that already fit are returned unchanged and consume no randomness.
    """
    if budget < 1:
        raise ValueError("The cloud budget must be positive")
    n_arm, n_cloth = len(arm), len(cloth)
    if n_arm + n_cloth <= budget:
        return arm, cloth
    if n_arm and n_cloth:
        keep_cloth = min(n_cloth, max(1, round(budget * n_cloth / (n_arm + n_cloth))))
        if budget > 1:
            keep_cloth = min(keep_cloth, budget - 1)
        keep_arm = min(n_arm, budget - keep_cloth)
        keep_cloth = min(n_cloth, budget - keep_arm)
    else:
        keep_arm, keep_cloth = min(n_arm, budget), min(n_cloth, budget)

    def sample(points, count):
        if count == len(points):
            return points
        return points[rng.choice(len(points), size=count, replace=False)]

    return sample(arm, keep_arm), sample(cloth, keep_cloth)


def voxel_downsample_torch(pts: torch.Tensor, voxel_size: float) -> torch.Tensor:
    if pts.numel() == 0:
        return pts.new_zeros((0, 3))
    voxel_idx = torch.floor(pts / float(voxel_size)).long()
    unique_vox, inverse = torch.unique(voxel_idx, dim=0, return_inverse=True)
    v, n = int(unique_vox.size(0)), int(pts.size(0))
    sums = pts.new_zeros((v, 3))
    sums.index_add_(0, inverse, pts)
    counts = pts.new_zeros((v,))
    counts.index_add_(0, inverse, pts.new_ones((n,)))
    return sums / counts.unsqueeze(-1).clamp_min(1.0)


def derive_dressing_cameras_torch(finger, shoulder, *, mode: str, distance_m: float, height_m: float, side: float = 1.0):
    center = 0.5 * (finger + shoulder)
    axis = shoulder - finger
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-9)
    up = torch.zeros_like(axis)
    up[..., 2] = 1.0
    side_dir = torch.linalg.cross(axis, up, dim=-1)
    side_norm = torch.linalg.vector_norm(side_dir, dim=-1, keepdim=True)
    fallback = torch.zeros_like(axis)
    fallback[..., 0] = 1.0
    side_dir = torch.where(side_norm > 1e-6, side_dir / side_norm.clamp_min(1e-9), fallback) * float(side)
    axis_h = axis.clone()
    axis_h[..., 2] = 0.0
    axis_h_norm = torch.linalg.vector_norm(axis_h, dim=-1, keepdim=True)
    axis_h = torch.where(axis_h_norm > 1e-6, axis_h / axis_h_norm.clamp_min(1e-9), side_dir)
    d, h = float(distance_m), float(height_m)
    cameras = [(center - 0.7 * d * axis_h + 0.7 * d * side_dir + h * up, center)]
    if mode == "visible_dual":
        cameras.append((center + d * side_dir + 1.5 * h * up, center))
    return cameras


def camera_visible_mask_torch(points, camera_position, camera_target, *, image_wh=96, depth_tolerance_m=0.02, fov_deg=70.0):
    squeeze = points.ndim == 2
    pts = points.unsqueeze(0) if squeeze else points
    cam = camera_position.reshape(-1, 3).to(dtype=pts.dtype, device=pts.device)
    tgt = camera_target.reshape(-1, 3).to(dtype=pts.dtype, device=pts.device)
    n = int(pts.shape[0])
    if cam.shape[0] == 1 and n > 1:
        cam, tgt = cam.expand(n, 3), tgt.expand(n, 3)
    fwd = tgt - cam
    fwd = fwd / torch.linalg.vector_norm(fwd, dim=-1, keepdim=True).clamp_min(1e-9)
    world_up = torch.zeros_like(fwd)
    world_up[..., 2] = 1.0
    world_alt = torch.zeros_like(fwd)
    world_alt[..., 1] = 1.0
    up0 = torch.where(fwd[..., 2].abs().unsqueeze(-1) > 0.99, world_alt, world_up)
    right = torch.linalg.cross(fwd, up0, dim=-1)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1e-9)
    up = torch.linalg.cross(right, fwd, dim=-1)
    rel = pts - cam.unsqueeze(1)
    depth = (rel * fwd.unsqueeze(1)).sum(dim=-1)
    x = (rel * right.unsqueeze(1)).sum(dim=-1)
    y = (rel * up.unsqueeze(1)).sum(dim=-1)
    half = math.tan(math.radians(float(fov_deg)) * 0.5)
    safe_depth = depth.clamp_min(1e-4)
    u, v = x / (safe_depth * half), y / (safe_depth * half)
    in_frustum = (depth > 1e-4) & (u.abs() < 1.0) & (v.abs() < 1.0)
    w = int(image_wh)
    px = (((u + 1.0) * 0.5) * w).long().clamp(0, w - 1)
    py = (((v + 1.0) * 0.5) * w).long().clamp(0, w - 1)
    pixel = py * w + px
    inf = depth.new_full(depth.shape, float("inf"))
    zbuf = depth.new_full((n, w * w), float("inf"))
    zbuf.scatter_reduce_(1, pixel, torch.where(in_frustum, depth, inf), reduce="amin", include_self=True)
    zmin = zbuf.gather(1, pixel)
    visible = in_frustum & (depth <= zmin + float(depth_tolerance_m))
    return visible.squeeze(0) if squeeze else visible


class DressingObservationBuilder:
    """Builds the Wang-style visible cloud for one environment on the GPU."""

    def __init__(self, cfg: DressingObsConfig, device) -> None:
        self.cfg = cfg
        self.device = torch.device(device)

    def visible_points(self, arm: np.ndarray, cloth: np.ndarray, finger: np.ndarray, shoulder: np.ndarray, rng: np.random.Generator, augment: bool):
        """Return ``(arm_points, cloth_points)`` after visibility, dropout, and voxelisation."""
        cfg = self.cfg
        dev = self.device
        arm_t = torch.as_tensor(arm, dtype=torch.float32, device=dev)
        cloth_t = torch.as_tensor(cloth, dtype=torch.float32, device=dev)
        finger_t = torch.as_tensor(finger, dtype=torch.float32, device=dev)
        shoulder_t = torch.as_tensor(shoulder, dtype=torch.float32, device=dev)
        combined = torch.cat([arm_t, cloth_t], dim=0)
        if cfg.mode == "xray":
            visible = torch.ones(combined.shape[0], dtype=torch.bool, device=dev)
        else:
            cameras = derive_dressing_cameras_torch(
                finger_t, shoulder_t, mode=cfg.mode, distance_m=cfg.camera_distance_m,
                height_m=cfg.camera_height_m, side=cfg.camera_side,
            )
            visible = torch.zeros(combined.shape[0], dtype=torch.bool, device=dev)
            active = 0
            for cam_pos, cam_tgt in cameras:
                if augment and cfg.camera_dropout_p > 0.0 and rng.random() < cfg.camera_dropout_p and len(cameras) > 1:
                    continue
                if augment and cfg.pose_jitter_m > 0.0:
                    cam_pos = cam_pos + torch.as_tensor(rng.uniform(-cfg.pose_jitter_m, cfg.pose_jitter_m, size=3), dtype=torch.float32, device=dev)
                visible |= camera_visible_mask_torch(
                    combined, cam_pos, cam_tgt, image_wh=cfg.image_wh,
                    depth_tolerance_m=cfg.depth_tolerance_m, fov_deg=cfg.fov_deg,
                )
                active += 1
            if active == 0:
                cam_pos, cam_tgt = cameras[0]
                visible = camera_visible_mask_torch(combined, cam_pos, cam_tgt, image_wh=cfg.image_wh, depth_tolerance_m=cfg.depth_tolerance_m, fov_deg=cfg.fov_deg)
        if augment and cfg.dropout_patches > 0 and cfg.dropout_radius_m > 0.0:
            candidates = torch.nonzero(visible, as_tuple=False).squeeze(1)
            if candidates.numel() > 0:
                for _ in range(int(cfg.dropout_patches)):
                    seed = combined[candidates[int(rng.integers(candidates.numel()))]]
                    visible &= torch.linalg.vector_norm(combined - seed[None, :], dim=1) > cfg.dropout_radius_m
        n_arm = int(arm_t.shape[0])
        arm_vis = voxel_downsample_torch(arm_t[visible[:n_arm]], cfg.voxel_size_m)
        cloth_vis = voxel_downsample_torch(cloth_t[visible[n_arm:]], cfg.voxel_size_m)
        return arm_vis.cpu().numpy(), cloth_vis.cpu().numpy()


# ---------------------------------------------------------------------------
# Batched observation over all environments
# ---------------------------------------------------------------------------
def visible_mask_batched(pts, valid, cam, tgt, *, image_wh, depth_tolerance_m, fov_deg):
    """``camera_visible_mask_torch`` on ``[B, N, 3]`` with a validity mask; padded rows never enter the z-buffer."""
    n = int(pts.shape[0])
    fwd = tgt - cam
    fwd = fwd / torch.linalg.vector_norm(fwd, dim=-1, keepdim=True).clamp_min(1e-9)
    world_up = torch.zeros_like(fwd)
    world_up[..., 2] = 1.0
    world_alt = torch.zeros_like(fwd)
    world_alt[..., 1] = 1.0
    up0 = torch.where(fwd[..., 2].abs().unsqueeze(-1) > 0.99, world_alt, world_up)
    right = torch.linalg.cross(fwd, up0, dim=-1)
    right = right / torch.linalg.vector_norm(right, dim=-1, keepdim=True).clamp_min(1e-9)
    up = torch.linalg.cross(right, fwd, dim=-1)
    rel = pts - cam.unsqueeze(1)
    depth = (rel * fwd.unsqueeze(1)).sum(dim=-1)
    x = (rel * right.unsqueeze(1)).sum(dim=-1)
    y = (rel * up.unsqueeze(1)).sum(dim=-1)
    half = math.tan(math.radians(float(fov_deg)) * 0.5)
    safe_depth = depth.clamp_min(1e-4)
    u, v = x / (safe_depth * half), y / (safe_depth * half)
    in_frustum = valid & (depth > 1e-4) & (u.abs() < 1.0) & (v.abs() < 1.0)
    w = int(image_wh)
    px = (((u + 1.0) * 0.5) * w).long().clamp(0, w - 1)
    py = (((v + 1.0) * 0.5) * w).long().clamp(0, w - 1)
    pixel = py * w + px
    inf = depth.new_full(depth.shape, float("inf"))
    zbuf = depth.new_full((n, w * w), float("inf"))
    zbuf.scatter_reduce_(1, pixel, torch.where(in_frustum, depth, inf), reduce="amin", include_self=True)
    zmin = zbuf.gather(1, pixel)
    return in_frustum & (depth <= zmin + float(depth_tolerance_m))


def voxel_centroids_batched(pts, select, voxel_size, n_envs):
    """Voxel centroids of the selected points of every environment, ordered ``(env, vx, vy, vz)``.

    The key packs the environment id above the offset voxel coordinates, so one
    sorted 1-D ``unique`` reproduces the per-environment ``unique(dim=0)`` order of
    :func:`voxel_downsample_torch`. Returns ``(centroids [M, 3], counts_per_env [B])``.
    """
    b, n, _ = pts.shape
    env_id = torch.arange(b, device=pts.device).unsqueeze(1).expand(b, n)
    sel_pts = pts[select]
    sel_env = env_id[select]
    vox = torch.floor(sel_pts / float(voxel_size)).long() + (1 << 15)
    key = (sel_env << 48) | (vox[:, 0] << 32) | (vox[:, 1] << 16) | vox[:, 2]
    uniq, inverse = torch.unique(key, return_inverse=True)
    m = int(uniq.shape[0])
    sums = pts.new_zeros((m, 3))
    sums.index_add_(0, inverse, sel_pts)
    counts = pts.new_zeros((m,))
    counts.index_add_(0, inverse, pts.new_ones((sel_pts.shape[0],)))
    centroids = sums / counts.unsqueeze(-1).clamp_min(1.0)
    per_env = torch.bincount(uniq >> 48, minlength=n_envs)
    return centroids, per_env


class BatchedDressingObservationBuilder:
    """The visible cloud for every environment in one pass.

    Twin of :class:`DressingObservationBuilder` with the same per-environment
    random draws in the same order, so outputs match to float precision, but the
    GPU work is issued once for the batch: one pinned upload, one z-buffer per
    camera, one patch-dropout pass, and one 1-D ``unique`` on packed keys. The
    per-environment version issues about 3,400 kernels and 440 stream
    synchronisations for 16 environments and spends its time waiting on them.
    """

    def __init__(self, cfg: DressingObsConfig, device) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self._pinned = None

    def _pad(self, arms, cloths):
        b = len(arms)
        n_arm = np.array([a.shape[0] for a in arms])
        n_cloth = np.array([c.shape[0] for c in cloths])
        n_tot = n_arm + n_cloth
        n_max = int(n_tot.max())
        if self._pinned is None or self._pinned.shape[0] != b or self._pinned.shape[1] != n_max:
            host = torch.empty((b, n_max, 3), dtype=torch.float32, device="cpu")
            self._pinned = host.pin_memory() if self.device.type == "cuda" else host
        host = self._pinned.numpy()
        for i, (a, c) in enumerate(zip(arms, cloths, strict=True)):
            host[i, : n_arm[i]] = a
            host[i, n_arm[i] : n_tot[i]] = c
            host[i, n_tot[i] :] = 0.0
        pts = self._pinned.to(self.device, non_blocking=True)
        ar = torch.arange(n_max, device=self.device).unsqueeze(0)
        is_arm = ar < torch.as_tensor(n_arm, device=self.device).unsqueeze(1)
        valid = ar < torch.as_tensor(n_tot, device=self.device).unsqueeze(1)
        return pts, is_arm, valid

    def visible_points(self, arms, cloths, fingers, shoulders, rngs, augment: bool):
        """Return a list of ``(arm_points, cloth_points)`` per environment."""
        cfg, dev = self.cfg, self.device
        pts, is_arm, valid = self._pad(arms, cloths)
        b = int(pts.shape[0])
        finger_t = torch.as_tensor(np.stack(fingers), dtype=torch.float32, device=dev)
        shoulder_t = torch.as_tensor(np.stack(shoulders), dtype=torch.float32, device=dev)
        if cfg.mode == "xray":
            visible = valid.clone()
        else:
            cameras = derive_dressing_cameras_torch(
                finger_t, shoulder_t, mode=cfg.mode, distance_m=cfg.camera_distance_m,
                height_m=cfg.camera_height_m, side=cfg.camera_side,
            )
            n_cam = len(cameras)
            active = np.ones((b, n_cam), dtype=bool)
            jitter = np.zeros((b, n_cam, 3), dtype=np.float32)
            if augment:
                # Same draws, same order, as the per-environment builder.
                for i, rng in enumerate(rngs):
                    for c in range(n_cam):
                        if cfg.camera_dropout_p > 0.0 and rng.random() < cfg.camera_dropout_p and n_cam > 1:
                            active[i, c] = False
                            continue
                        if cfg.pose_jitter_m > 0.0:
                            jitter[i, c] = rng.uniform(-cfg.pose_jitter_m, cfg.pose_jitter_m, size=3)
            active_t = torch.as_tensor(active, device=dev)
            jitter_t = torch.as_tensor(jitter, device=dev)
            visible = torch.zeros_like(valid)
            for c, (cam_pos, cam_tgt) in enumerate(cameras):
                vis_c = visible_mask_batched(
                    pts, valid, cam_pos + jitter_t[:, c], cam_tgt,
                    image_wh=cfg.image_wh, depth_tolerance_m=cfg.depth_tolerance_m, fov_deg=cfg.fov_deg,
                )
                visible |= vis_c & active_t[:, c].unsqueeze(1)
            none_active = ~active.any(axis=1)
            if none_active.any():
                cam_pos, cam_tgt = cameras[0]
                vis0 = visible_mask_batched(
                    pts, valid, cam_pos, cam_tgt,
                    image_wh=cfg.image_wh, depth_tolerance_m=cfg.depth_tolerance_m, fov_deg=cfg.fov_deg,
                )
                visible = torch.where(torch.as_tensor(none_active, device=dev).unsqueeze(1), vis0, visible)
        if augment and cfg.dropout_patches > 0 and cfg.dropout_radius_m > 0.0:
            counts = visible.sum(dim=1)
            cand = torch.nonzero(visible, as_tuple=False)
            counts_np = counts.cpu().numpy()
            starts = np.concatenate([[0], np.cumsum(counts_np)[:-1]])
            picks = np.full((b, int(cfg.dropout_patches)), -1, dtype=np.int64)
            for i, rng in enumerate(rngs):
                if counts_np[i] > 0:
                    for k in range(int(cfg.dropout_patches)):
                        picks[i, k] = starts[i] + int(rng.integers(counts_np[i]))
            picks_t = torch.as_tensor(picks, device=dev)
            has = picks_t >= 0
            seed_idx = cand[picks_t.clamp_min(0), 1]
            seeds = torch.gather(pts, 1, seed_idx.unsqueeze(-1).expand(-1, -1, 3))
            d = torch.linalg.vector_norm(pts.unsqueeze(1) - seeds.unsqueeze(2), dim=-1)
            keep = (d > cfg.dropout_radius_m) | ~has.unsqueeze(-1)
            visible &= keep.all(dim=1)
        arm_c, arm_n = voxel_centroids_batched(pts, visible & is_arm, cfg.voxel_size_m, b)
        cloth_c, cloth_n = voxel_centroids_batched(pts, visible & valid & ~is_arm, cfg.voxel_size_m, b)
        arm_np, cloth_np = arm_c.cpu().numpy(), cloth_c.cpu().numpy()
        arm_n, cloth_n = arm_n.cpu().numpy(), cloth_n.cpu().numpy()
        a_off = np.concatenate([[0], np.cumsum(arm_n)])
        c_off = np.concatenate([[0], np.cumsum(cloth_n)])
        return [(arm_np[a_off[i] : a_off[i + 1]], cloth_np[c_off[i] : c_off[i + 1]]) for i in range(b)]
