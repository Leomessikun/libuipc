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
    # Camera rigs (``RIG_MODES``) only; the legacy modes ignore every field below.
    static_arm: bool = False
    """Render the arm without the garment, as Wang's pre-dressing capture; ``wang_static_arm`` always does."""
    head_height_m: float = 1.30
    """Stretch 3 head camera above the floor: ``joint_head`` sits 1.33 m up the mast in stretch_urdf SE3."""
    head_distance_m: float = 0.8
    """Horizontal distance from the arm's midpoint to the head, in front of and to the right of the person."""
    head_hfov_deg: float = 58.0
    head_vfov_deg: float = 87.0
    """RealSense D435if depth field of view, 87 x 58 degrees, mounted in portrait in Stretch's head."""
    head_image_w: int = 56
    head_image_h: int = 96
    head_range_m: tuple[float, float] = (0.3, 3.0)
    """D435if ideal depth range."""
    wrist_back_m: float = 0.19
    wrist_up_m: float = 0.045
    """Stretch Gripper 3's D405 sits about 19 cm behind the grasp centre and 4.4 cm off the gripper axis."""
    wrist_look_m: float = 0.1
    """The wrist camera aims this far past the grasp centre, along the gripper."""
    wrist_hfov_deg: float = 87.0
    wrist_vfov_deg: float = 58.0
    wrist_image_w: int = 48
    wrist_image_h: int = 28
    wrist_range_m: tuple[float, float] = (0.07, 0.5)
    """RealSense D405 ideal depth range."""
    wrist_jitter_m: float = 0.005
    """Extrinsic jitter of the wrist camera, which is bolted to the gripper and calibrated."""
    body_arm_exclusion_m: float = 0.015
    """Body points this close to an arm point are the arm itself and leave the occluders."""
    rig_splat_m: float = 0.01
    """World radius each point covers in a rig's depth buffer, so vertex-sampled meshes stay opaque up close."""
    rig_self_tolerance_m: float = 0.02
    rig_cross_tolerance_m: float = 0.005
    """Depth slack against the point's own segment and against the others; the tight one hides skin under a sleeve."""


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

    def visible_points(self, arm: np.ndarray, cloth: np.ndarray, finger: np.ndarray, shoulder: np.ndarray, rng: np.random.Generator, augment: bool, rig: "RigInputs | None" = None):
        """Return ``(arm_points, cloth_points)`` after visibility, dropout, and voxelisation."""
        if self.cfg.mode in RIG_MODES:
            if getattr(self, "_rig_builder", None) is None:
                self._rig_builder = BatchedDressingObservationBuilder(self.cfg, self.device)
            return self._rig_builder.visible_points([arm], [cloth], [finger], [shoulder], [rng], augment, rig=rig)[0]
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

    def visible_points(self, arms, cloths, fingers, shoulders, rngs, augment: bool, rig: "RigInputs | None" = None):
        """Return a list of ``(arm_points, cloth_points)`` per environment.

        The camera rigs of ``RIG_MODES`` also need ``rig``; the legacy modes ignore it.
        """
        if self.cfg.mode in RIG_MODES:
            return _rig_visible_points(self, arms, cloths, fingers, shoulders, rngs, augment, rig)
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

    def rig_visibility(self, arms, cloths, fingers, shoulders, rngs, augment: bool, rig: "RigInputs"):
        """Per-point visibility of a rig mode: ``(points, segment, valid, visible)``, each ``[B, N, ...]``.

        Each row holds the slot's arm points, then its garment points, then its body
        occluders. ``visible`` marks the arm and garment points an active camera sees after
        camera and patch dropout; body points are never visible.
        """
        return _rig_visibility(self, arms, cloths, fingers, shoulders, rngs, augment, rig)


# ---------------------------------------------------------------------------
# Camera rigs: explicit intrinsics, body occlusion, range limits
# ---------------------------------------------------------------------------
RIG_MODES = ("wang_static_arm", "stretch3_head", "stretch3_head_wrist")
"""Observation modes rendered by :meth:`BatchedDressingObservationBuilder.rig_visibility`.

``wang_static_arm`` is Wang RSS 2023's ``pointcloud_3`` on the port's front-oblique
camera: the arm is captured without the garment, so the sleeve never hides it, and the
garment is hidden by the body. The ``stretch3`` rigs put a Hello Robot Stretch 3 pan-tilt
head camera (RealSense D435if in portrait, 58 x 87 degrees, 0.3 to 3 m) in front of and
to the right of the person, aimed at the arm, and for ``stretch3_head_wrist`` its gripper
camera (RealSense D405, 87 x 58 degrees, 7 to 50 cm) rigidly on the tool; both see the arm
live, so the sleeve hides it unless ``static_arm`` is set. Every rig renders the arm, the
garment and the rest of the body into one segment-aware depth buffer; the camera-rig
record in ``agent_docs/performance`` gives the geometry and its sources.
"""
SEG_ARM, SEG_CLOTH, SEG_BODY = 0, 1, 2
MAX_SPLAT_PX = 3


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-9)


@dataclass
class RigInputs:
    """Per-slot scene data the rigs read beyond the legacy builder's arguments."""

    elbows: np.ndarray
    """``[B, 3]`` elbow landmarks."""
    lateral: np.ndarray
    """``[B, 3]`` the body's left-to-right direction; only its horizontal part is used."""
    floors: np.ndarray
    """``[B]`` floor height under each body."""
    tools: np.ndarray
    """``[B, 3]`` commanded tool points."""
    tool_rotations: np.ndarray
    """``[B, 3, 3]`` rotation of the grip since the reset."""
    bodies: list
    """Every body point per slot; occluders only, never observed."""

    @classmethod
    def from_cells(cls, cells, tools, offsets, initial_offsets) -> "RigInputs":
        """The rig inputs of an environment's cells, tool points and held-patch offsets."""
        from .dressing_privileged import grip_rotation

        lateral = []
        for cell in cells:
            marks = cell.landmarks
            if "left_shoulder" in marks and "right_shoulder" in marks:
                lateral.append(np.asarray(marks["right_shoulder"], dtype=np.float64) - np.asarray(marks["left_shoulder"], dtype=np.float64))
            elif "right_hip" in marks and "pelvis" in marks:
                # Live bodies carry only the right-side joints; the hip stays put whatever the arm does.
                lateral.append(np.asarray(marks["right_hip"], dtype=np.float64) - np.asarray(marks["pelvis"], dtype=np.float64))
            else:
                # From the body's centre toward the right shoulder.
                lateral.append(np.asarray(cell.shoulder, dtype=np.float64) - np.asarray(cell.human_points, dtype=np.float64).mean(axis=0))
        return cls(
            elbows=np.stack([np.asarray(cell.elbow, dtype=np.float64) for cell in cells]),
            lateral=np.stack(lateral),
            floors=np.array([float(np.asarray(cell.human_points)[:, 2].min()) for cell in cells]),
            tools=np.asarray(tools, dtype=np.float64).reshape(len(cells), 3),
            tool_rotations=np.stack([grip_rotation(o0, o) for o0, o in zip(initial_offsets, offsets, strict=True)]),
            bodies=[cell.human_points for cell in cells],
        )


@dataclass
class RigCamera:
    """One pinhole depth camera per slot: poses ``[B, 3]`` and intrinsics shared by the batch."""

    position: torch.Tensor
    target: torch.Tensor
    up: torch.Tensor
    hfov_deg: float
    vfov_deg: float
    width: int
    height: int
    near_m: float
    far_m: float
    jitter_m: float


def rig_cameras(cfg: DressingObsConfig, finger, elbow, shoulder, lateral, floor, tool, tool_rotation) -> list[RigCamera]:
    """The cameras of ``cfg.mode`` for every slot; each argument is a ``[B, ...]`` tensor.

    ``wang_static_arm`` keeps the port's front-oblique camera, the first of ``visible_dual``.
    The Stretch head stands ``head_distance_m`` from the arm's midpoint, in front of and to
    the right of the person, ``head_height_m`` above the floor, and pans and tilts onto the
    arm's midpoint. The wrist camera rides the tool: at the reset the gripper points along
    the forearm toward the elbow with the camera above it, and the grip rotation carries both.
    """
    up = torch.zeros_like(finger)
    up[..., 2] = 1.0
    if cfg.mode == "wang_static_arm":
        pos, tgt = derive_dressing_cameras_torch(
            finger, shoulder, mode="visible_single", distance_m=cfg.camera_distance_m,
            height_m=cfg.camera_height_m, side=cfg.camera_side,
        )[0]
        return [RigCamera(pos, tgt, up, cfg.fov_deg, cfg.fov_deg, cfg.image_wh, cfg.image_wh, 1e-4, float("inf"), cfg.pose_jitter_m)]
    if cfg.mode not in ("stretch3_head", "stretch3_head_wrist"):
        raise ValueError(f"{cfg.mode!r} is not a camera rig; expected one of {RIG_MODES}")
    right = lateral.clone()
    right[..., 2] = 0.0
    right = _unit(right)
    toward = _unit(torch.linalg.cross(up, right, dim=-1) + right)
    mid = 0.5 * (finger + shoulder)
    head = mid + float(cfg.head_distance_m) * toward
    head[..., 2] = floor + float(cfg.head_height_m)
    near, far = (float(x) for x in cfg.head_range_m)
    cameras = [RigCamera(head, mid, up, cfg.head_hfov_deg, cfg.head_vfov_deg, cfg.head_image_w, cfg.head_image_h, near, far, cfg.pose_jitter_m)]
    if cfg.mode == "stretch3_head_wrist":
        f0 = _unit(elbow - finger)
        n0 = up - (up * f0).sum(dim=-1, keepdim=True) * f0
        n0 = _unit(torch.where(torch.linalg.vector_norm(n0, dim=-1, keepdim=True) > 1e-6, n0, right))
        f = torch.einsum("bij,bj->bi", tool_rotation, f0)
        n = torch.einsum("bij,bj->bi", tool_rotation, n0)
        position = tool - float(cfg.wrist_back_m) * f + float(cfg.wrist_up_m) * n
        near, far = (float(x) for x in cfg.wrist_range_m)
        cameras.append(RigCamera(position, tool + float(cfg.wrist_look_m) * f, n, cfg.wrist_hfov_deg, cfg.wrist_vfov_deg,
                                 cfg.wrist_image_w, cfg.wrist_image_h, near, far, cfg.wrist_jitter_m))
    return cameras


def rig_visible_mask(pts, seg, valid, cam: RigCamera, *, splat_m: float, self_tolerance_m: float, cross_tolerance_m: float) -> torch.Tensor:
    """Points of ``pts [B, N, 3]`` that ``cam`` sees, given segments ``seg [B, N]`` (arm, garment, body).

    Every point splats a square of ``2r + 1`` pixels, ``r`` being ``splat_m`` over the pixel
    footprint at its depth (at most ``MAX_SPLAT_PX``), so a mesh sampled at its vertices stays
    opaque close to the camera. Each segment keeps its own depth buffer: a point is visible
    when no point of its own segment lies more than ``self_tolerance_m`` in front of it at its
    pixel and no point of another segment more than ``cross_tolerance_m``.
    """
    b, n, _ = pts.shape
    fwd = _unit(cam.target - cam.position)
    right = torch.linalg.cross(fwd, cam.up, dim=-1)
    alt = torch.zeros_like(fwd)
    alt[..., 1] = 1.0
    right = _unit(torch.where(torch.linalg.vector_norm(right, dim=-1, keepdim=True) > 1e-6, right, torch.linalg.cross(fwd, alt, dim=-1)))
    up = torch.linalg.cross(right, fwd, dim=-1)
    rel = pts - cam.position.unsqueeze(1)
    depth = (rel * fwd.unsqueeze(1)).sum(dim=-1)
    x = (rel * right.unsqueeze(1)).sum(dim=-1)
    y = (rel * up.unsqueeze(1)).sum(dim=-1)
    tan_h = math.tan(math.radians(float(cam.hfov_deg)) * 0.5)
    tan_v = math.tan(math.radians(float(cam.vfov_deg)) * 0.5)
    safe = depth.clamp_min(1e-4)
    u, v = x / (safe * tan_h), y / (safe * tan_v)
    w, h = int(cam.width), int(cam.height)
    seen = valid & (depth > float(cam.near_m)) & (depth < float(cam.far_m)) & (u.abs() < 1.0) & (v.abs() < 1.0)
    px = (((u + 1.0) * 0.5) * w).long().clamp(0, w - 1)
    py = (((v + 1.0) * 0.5) * h).long().clamp(0, h - 1)
    radius = (float(splat_m) / (2.0 * safe * tan_h / w)).floor().long().clamp(0, MAX_SPLAT_PX)
    radius = torch.where(seen, radius, torch.zeros_like(radius))
    inf = float("inf")
    value = torch.where(seen, depth, torch.full_like(depth, inf))
    segment = seg.clamp(0, 2)
    zbuf = depth.new_full((b, 3 * h * w), inf)
    reach = int(radius.max()) if n else 0
    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            qx, qy = px + dx, py + dy
            ok = seen & (radius >= max(abs(dx), abs(dy))) & (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
            index = segment * (h * w) + qy.clamp(0, h - 1) * w + qx.clamp(0, w - 1)
            zbuf.scatter_reduce_(1, index, torch.where(ok, value, torch.full_like(value, inf)), reduce="amin", include_self=True)
    pixel = py * w + px
    per_segment = torch.stack([zbuf.gather(1, s * (h * w) + pixel) for s in range(3)], dim=1)
    own = per_segment.gather(1, segment.unsqueeze(1)).squeeze(1)
    others = per_segment.scatter(1, segment.unsqueeze(1), inf).min(dim=1).values
    return seen & (depth <= own + float(self_tolerance_m)) & (depth <= others + float(cross_tolerance_m))


def body_occluders(body: np.ndarray, arm: np.ndarray, exclusion_m: float) -> np.ndarray:
    """Body points farther than ``exclusion_m`` from every arm point; the arm is its own segment."""
    body = np.asarray(body, dtype=np.float64)
    if len(body) == 0 or len(arm) == 0:
        return body.astype(np.float32)
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(np.asarray(arm, dtype=np.float64)).query(body, k=1)
    return body[distance > float(exclusion_m)].astype(np.float32)


def _patch_dropout(pts, visible, rngs, cfg: DressingObsConfig):
    """The legacy builder's local patch dropout, over the visible points."""
    b = int(pts.shape[0])
    counts_np = visible.sum(dim=1).cpu().numpy()
    if not counts_np.any():
        return visible
    cand = torch.nonzero(visible, as_tuple=False)
    starts = np.concatenate([[0], np.cumsum(counts_np)[:-1]])
    picks = np.full((b, int(cfg.dropout_patches)), -1, dtype=np.int64)
    for i, rng in enumerate(rngs):
        if counts_np[i] > 0:
            for k in range(int(cfg.dropout_patches)):
                picks[i, k] = starts[i] + int(rng.integers(counts_np[i]))
    picks_t = torch.as_tensor(picks, device=pts.device)
    has = picks_t >= 0
    seeds = torch.gather(pts, 1, cand[picks_t.clamp_min(0), 1].unsqueeze(-1).expand(-1, -1, 3))
    d = torch.linalg.vector_norm(pts.unsqueeze(1) - seeds.unsqueeze(2), dim=-1)
    return visible & ((d > cfg.dropout_radius_m) | ~has.unsqueeze(-1)).all(dim=1)


def _rig_visibility(builder, arms, cloths, fingers, shoulders, rngs, augment: bool, rig: "RigInputs | None"):
    cfg, dev = builder.cfg, builder.device
    if rig is None:
        raise ValueError(f"Observation mode {cfg.mode!r} needs RigInputs (see RigInputs.from_cells)")
    cache = builder.__dict__.setdefault("_occluders", {})
    bodies = []
    for body, arm in zip(rig.bodies, arms, strict=True):
        key = (id(body), id(arm), len(body), len(arm))
        if key not in cache:
            cache[key] = body_occluders(body, arm, cfg.body_arm_exclusion_m)
        bodies.append(cache[key])
    b = len(arms)
    counts = np.array([[len(a), len(c), len(o)] for a, c, o in zip(arms, cloths, bodies, strict=True)], dtype=np.int64)
    n_max = max(int(counts.sum(axis=1).max()), 1)
    host = np.zeros((b, n_max, 3), dtype=np.float32)
    seg_np = np.full((b, n_max), SEG_BODY, dtype=np.int64)
    valid_np = np.zeros((b, n_max), dtype=bool)
    for i, (a, c, o) in enumerate(zip(arms, cloths, bodies, strict=True)):
        na, nc, no = (int(x) for x in counts[i])
        host[i, :na], host[i, na : na + nc], host[i, na + nc : na + nc + no] = a, c, o
        seg_np[i, :na], seg_np[i, na : na + nc] = SEG_ARM, SEG_CLOTH
        valid_np[i, : na + nc + no] = True
    pts = torch.as_tensor(host, device=dev)
    seg = torch.as_tensor(seg_np, device=dev)
    valid = torch.as_tensor(valid_np, device=dev)

    def t(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=dev)

    cameras = rig_cameras(cfg, t(np.stack(fingers)), t(rig.elbows), t(np.stack(shoulders)), t(rig.lateral), t(rig.floors), t(rig.tools), t(rig.tool_rotations))
    n_cam = len(cameras)
    active = np.ones((b, n_cam), dtype=bool)
    jitter = np.zeros((b, n_cam, 3), dtype=np.float32)
    if augment:
        for i, rng in enumerate(rngs):
            for c, cam in enumerate(cameras):
                if cfg.camera_dropout_p > 0.0 and n_cam > 1 and rng.random() < cfg.camera_dropout_p:
                    active[i, c] = False
                    continue
                if cam.jitter_m > 0.0:
                    jitter[i, c] = rng.uniform(-cam.jitter_m, cam.jitter_m, size=3)
    # With every camera dropped, the first one looks anyway, unjittered.
    active[~active.any(axis=1), 0] = True
    active_t = torch.as_tensor(active, device=dev)
    jitter_t = torch.as_tensor(jitter, device=dev)
    static = bool(cfg.static_arm) or cfg.mode == "wang_static_arm"
    is_arm, is_cloth = seg == SEG_ARM, seg == SEG_CLOTH
    tol = {"splat_m": cfg.rig_splat_m, "self_tolerance_m": cfg.rig_self_tolerance_m, "cross_tolerance_m": cfg.rig_cross_tolerance_m}
    visible = torch.zeros_like(valid)
    for c, cam in enumerate(cameras):
        cam = RigCamera(cam.position + jitter_t[:, c], cam.target, cam.up, cam.hfov_deg, cam.vfov_deg, cam.width, cam.height, cam.near_m, cam.far_m, cam.jitter_m)
        live = rig_visible_mask(pts, seg, valid, cam, **tol)
        if static:
            bare = rig_visible_mask(pts, seg, valid & ~is_cloth, cam, **tol)
            seen = (bare & is_arm) | (live & is_cloth)
        else:
            seen = live & (is_arm | is_cloth)
        visible |= seen & active_t[:, c].unsqueeze(1)
    if augment and cfg.dropout_patches > 0 and cfg.dropout_radius_m > 0.0:
        visible = _patch_dropout(pts, visible, rngs, cfg)
    return pts, seg, valid, visible


def _rig_visible_points(builder, arms, cloths, fingers, shoulders, rngs, augment: bool, rig: "RigInputs | None"):
    pts, seg, _, visible = _rig_visibility(builder, arms, cloths, fingers, shoulders, rngs, augment, rig)
    b = int(pts.shape[0])
    arm_c, arm_n = voxel_centroids_batched(pts, visible & (seg == SEG_ARM), builder.cfg.voxel_size_m, b)
    cloth_c, cloth_n = voxel_centroids_batched(pts, visible & (seg == SEG_CLOTH), builder.cfg.voxel_size_m, b)
    arm_np, cloth_np = arm_c.cpu().numpy(), cloth_c.cpu().numpy()
    a_off = np.concatenate([[0], np.cumsum(arm_n.cpu().numpy())])
    c_off = np.concatenate([[0], np.cumsum(cloth_n.cpu().numpy())])
    return [(arm_np[a_off[i] : a_off[i + 1]], cloth_np[c_off[i] : c_off[i + 1]]) for i in range(b)]
