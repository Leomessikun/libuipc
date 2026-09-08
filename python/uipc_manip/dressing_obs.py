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
