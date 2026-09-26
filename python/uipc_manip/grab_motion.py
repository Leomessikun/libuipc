"""Retarget recorded GRAB right-arm rotations onto a fixed dressing body.

Only the collar, shoulder, elbow and wrist are transferred. The recipient's
shape, seated torso, legs, fist, scale and world origin stay fixed. This is
motion retargeting, not a replay of the complete GRAB interaction or its object.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ARM_POSE_INDICES = np.array([13, 16, 18, 20])  # SMPL-X joints 14, 17, 19, 21
LANDMARK_NAMES = ("right_finger", "right_wrist", "right_elbow", "right_shoulder")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def retarget_poses(source, source_fps, base_pose, times, *, start_s=0., amplitude=1., ramp_s=.3):
    """Interpolate rotations on SO(3); preserve the recipient pose at time zero."""
    source = np.asarray(source, dtype=np.float64)
    base = np.asarray(base_pose, dtype=np.float64).reshape(21, 3)
    times = np.asarray(times, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 63 or len(source) < 2:
        raise ValueError("GRAB body_pose must have shape (T >= 2, 63)")
    if not np.isfinite(source).all() or not np.isfinite(base).all():
        raise ValueError("Non-finite body pose")
    if (not np.isfinite([source_fps, start_s, amplitude, ramp_s]).all()
            or source_fps <= 0 or start_s < 0 or not 0 <= amplitude <= 1 or ramp_s < 0):
        raise ValueError("Need positive fps, nonnegative start/ramp, amplitude in [0, 1]")
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Times must start at zero and increase strictly")
    source_times = np.arange(len(source)) / source_fps
    if start_s + times[-1] > source_times[-1] + 1e-9:
        raise ValueError("Requested clip extends beyond the source; select a shorter duration")
    result = np.broadcast_to(base, (len(times), 21, 3)).copy()
    blend = np.ones(len(times)) if ramp_s == 0 else np.clip(times / ramp_s, 0, 1)
    if ramp_s:
        blend = blend * blend * (3 - 2 * blend)
    for j in ARM_POSE_INDICES:
        interp = Slerp(source_times, Rotation.from_rotvec(source.reshape(-1, 21, 3)[:, j]))
        origin = interp([start_s])[0]
        delta = origin.inv() * interp(np.minimum(start_s + times, source_times[-1]))
        scaled = Rotation.from_rotvec(delta.as_rotvec() * (amplitude * blend[:, None]))
        result[:, j] = (Rotation.from_rotvec(base[j]) * scaled).as_rotvec()
    return result.reshape(-1, 63).astype(np.float32)


@dataclass
class BodyMotion:
    times: np.ndarray
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    arm_indices: np.ndarray
    landmarks: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float64)
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.joints = np.asarray(self.joints, dtype=np.float64)
        self.landmarks = np.asarray(self.landmarks, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        self.arm_indices = np.asarray(self.arm_indices, dtype=np.int64)
        n = len(self.times)
        if n < 2 or self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Motion needs >=2 increasing timestamps starting at zero")
        if self.vertices.ndim != 3 or self.vertices.shape[0] != n or self.vertices.shape[2] != 3:
            raise ValueError("Motion vertices must have shape (T,V,3)")
        if self.joints.ndim != 3 or self.joints.shape[0] != n or self.joints.shape[2] != 3:
            raise ValueError("Motion joints must have shape (T,J,3)")
        if self.landmarks.shape != (n, len(LANDMARK_NAMES), 3):
            raise ValueError("Motion landmarks must be finger,wrist,elbow,shoulder per frame")
        if not all(np.isfinite(v).all() for v in (self.times, self.vertices, self.joints, self.landmarks)):
            raise ValueError("Motion contains non-finite coordinates")
        nv = self.vertices.shape[1]
        if (self.faces.ndim != 2 or self.faces.shape[1] != 3 or not self.faces.size
                or self.faces.min() < 0 or self.faces.max() >= nv):
            raise ValueError("Motion faces are outside the vertex array")
        if (self.arm_indices.ndim != 1 or not len(self.arm_indices)
                or self.arm_indices.min() < 0 or self.arm_indices.max() >= nv
                or len(np.unique(self.arm_indices)) != len(self.arm_indices)):
            raise ValueError("Invalid arm vertex indices")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k: data[k].copy() for k in
                         ("times", "vertices", "joints", "faces", "arm_indices", "landmarks")},
                       metadata=json.loads(str(data["metadata"])))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally replacing another experiment.
        with path.open("xb") as handle:
            np.savez_compressed(handle, times=self.times, vertices=self.vertices.astype(np.float32),
                                joints=self.joints.astype(np.float32), faces=self.faces,
                                arm_indices=self.arm_indices, landmarks=self.landmarks.astype(np.float32),
                                metadata=json.dumps(self.metadata, sort_keys=True))

    def sample(self, time_s):
        if not np.isfinite(time_s):
            raise ValueError("Motion time must be finite")
        t = float(np.clip(time_s, 0., self.times[-1]))
        lo = min(max(int(np.searchsorted(self.times, t, side="right")) - 1, 0), len(self.times) - 2)
        w = (t - self.times[lo]) / (self.times[lo + 1] - self.times[lo])
        return tuple((1 - w) * x[lo] + w * x[lo + 1]
                     for x in (self.vertices, self.joints, self.landmarks))


def convert_grab(path, *, body_id=14046, start_s=0., duration_s=3., fps=30.,
                 amplitude=.35, ramp_s=.3, body_config=None):
    """CPU conversion of a trusted local GRAB NPZ to our existing SMPL-X recipient."""
    import torch
    from .dressing_body import (BodyConfig, _cpu_default_device, _model,
                                _reward_line_landmarks, body_parameters, generate_body)
    from dataclasses import replace

    if not np.isfinite([duration_s, fps]).all() or duration_s <= 0 or fps <= 0:
        raise ValueError("Duration and output fps must be positive")
    path = Path(path).resolve()
    # Official GRAB stores nested parameter dictionaries as pickled object arrays.
    with np.load(path, allow_pickle=True) as data:
        source = np.asarray(data["body"].item()["params"]["body_pose"])
        source_fps = float(data["framerate"])
        source_info = {k: str(data[k]) for k in ("sbj_id", "gender", "motion_intent", "obj_name")}
    cfg = replace(body_config or BodyConfig(), device="cpu")
    rng, gender, betas, base_pose = body_parameters(body_id, cfg)
    count = int(np.floor(duration_s * fps + 1e-9))
    if count < 1:
        raise ValueError("Clip must contain at least two frames")
    times = np.arange(count + 1, dtype=np.float64) / fps
    poses = retarget_poses(source, source_fps, base_pose, times, start_s=start_s,
                           amplitude=amplitude, ramp_s=ramp_s)
    reference = generate_body(body_id, cfg)
    model = _model(str(cfg.model_dir), gender, cfg.num_betas).to("cpu")
    rot = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    hand = torch.full((1, 6), float(cfg.hand_pca), device="cpu")
    vertices, joints = [], []
    with _cpu_default_device(), torch.no_grad():
        beta_t = torch.as_tensor(betas, device="cpu").unsqueeze(0)
        scale = 1.
        if cfg.height_range is not None:
            rest = model(betas=beta_t, body_pose=torch.zeros((1, 63)), right_hand_pose=hand)
            height = float(np.ptp(rest.vertices.numpy()[0] @ rot.T, axis=0)[2])
            scale = float(rng.uniform(*cfg.height_range)) / height
        for pose in poses:
            out = model(betas=beta_t, body_pose=torch.from_numpy(pose).unsqueeze(0), right_hand_pose=hand)
            vertices.append(out.vertices.numpy()[0].astype(np.float64) @ rot.T * scale)
            joints.append(out.joints.numpy()[0].astype(np.float64) @ rot.T * scale)
    vertices, joints = np.stack(vertices), np.stack(joints)
    # One fixed ground offset, rather than independently grounding each frame.
    ground = float(vertices[0, :, 2].min())
    vertices[:, :, 2] -= ground
    joints[:, :, 2] -= ground
    initial_error = float(np.max(np.abs(vertices[0] - reference.vertices)))
    if initial_error > 1e-5:
        raise RuntimeError(f"Retargeted frame zero differs from recipient by {initial_error:g} m")
    landmarks = np.stack([np.stack([lm[k] for k in LANDMARK_NAMES]) for lm in
                          (_reward_line_landmarks(v, j) for v, j in zip(vertices, joints, strict=True))])
    max_speed = float(np.max(np.linalg.norm(np.diff(vertices, axis=0), axis=-1)
                             / np.diff(times)[:, None]))
    metadata = dict(schema_version=1, source=str(path), source_sha256=sha256(path),
                    source_info=source_info, source_fps=source_fps, source_start_s=start_s,
                    body_id=int(body_id), body_config=cfg.to_dict(), fps=fps,
                    duration_s=float(times[-1]), amplitude=amplitude, ramp_s=ramp_s,
                    transferred_smplx_joints=(ARM_POSE_INDICES + 1).tolist(),
                    root_motion="fixed seated recipient", source_object="not simulated",
                    coordinate_frame="metres, +Z up", initial_error_m=initial_error,
                    max_vertex_speed_m_s=max_speed, landmark_names=list(LANDMARK_NAMES))
    return BodyMotion(times, vertices, joints, reference.faces, reference.arm_indices, landmarks, metadata)
