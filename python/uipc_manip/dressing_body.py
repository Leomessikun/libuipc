"""Generate a dressing body from SMPL-X, with no dependence on saved states.

Wang's `gen_human_mesh.py` samples an SMPL-X body per trial: ten shape
coefficients from U(-2, 5), a seated pose with the right arm extended forward
and its shoulder and elbow angles randomised, the right hand pinned to a relaxed
fist, and a standing height from U(1.5, 1.9). This module does the same, so the
body axis of the task is generated here rather than read from the Newton cache.

The right-arm collider is derived from SMPL-X's own skinning weights: every
vertex whose dominant joint lies on the right collar-to-fingertip chain. That
keeps the submesh a property of the model rather than of a saved index list, and
it follows the body through every shape and pose sample.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

SMPLX_VERTEX_COUNT = 10475
SMPLX_BODY_JOINTS = 21

_DEFAULT_MODEL_DIR = Path(
    os.environ.get("UIPC_MANIP_SMPLX_DIR", "/home/ge47gax/kun/newton-fmvp/models_smplx_v1_1/models")
)

# SMPL-X joint indices (smplx.joint_names.JOINT_NAMES order).
JOINT_INDEX = {
    "pelvis": 0,
    "right_hip": 2,
    "right_knee": 5,
    "right_ankle": 8,
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
    "right_index1": 40,
}
# Body-pose entries are joints 1..21, so a joint's rotation starts at (index - 1) * 3.
_POSE = {name: (index - 1) * 3 for name, index in
         {"left_hip": 1, "right_hip": 2, "left_knee": 4, "right_knee": 5,
          "left_shoulder": 16, "right_shoulder": 17, "left_elbow": 18, "right_elbow": 19}.items()}

# Arm-surface landmarks Newton reads off the fixed SMPL-X topology: the upper
# and lower vertex of the wrist, elbow, and shoulder sections. Averaging the
# pair puts the landmark on the limb axis rather than on the skin.
REWARD_LINE_UPPER = (7462, 7039, 5995)
REWARD_LINE_LOWER = (7497, 7260, 7176)
_RIGHT_HAND_JOINTS = (40, 55)

POSE_REGION_EDGES = {
    "shoulder_z": (-20.0, -8.0, 18.0, 30.0),
    "elbow_y": (70.0, 82.0, 98.0, 110.0),
    "elbow_z": (-20.0, -3.0, 14.0, 30.0),
}
"""Wang RSS 2023's split of the right-arm pose range into 27 regions (Appendix B.3), in
degrees: each of the three randomised angles is cut into three intervals. ``elbow_y`` is
Wang's inwards-outwards elbow angle, offset by the 90 degrees this seated pose holds the
forearm at. Region ``9 i + 3 j + k`` takes interval ``i`` of the shoulder, ``j`` of
``elbow_y`` and ``k`` of ``elbow_z``; Wang's own region numbering is not recorded."""

REGION_BODY_BASE = 1000
"""Body ids from ``1000 (r + 1)`` to ``1000 (r + 1) + 999`` sample their arm pose inside region
``r``; smaller ids sample the whole range, so the ids used before regions existed keep their bodies."""


def pose_region(seed: int) -> int | None:
    """The Wang arm-pose region a body id asks for, or None for the whole range."""
    seed = int(seed)
    if seed < REGION_BODY_BASE:
        return None
    region = seed // REGION_BODY_BASE - 1
    if not 0 <= region < 27:
        raise ValueError(f"Body id {seed} names region {region}; region ids run from 0 to 26")
    return region


def region_intervals(region: int) -> tuple[tuple[float, float], ...]:
    """``(shoulder_z, elbow_y, elbow_z)`` intervals of one region, in degrees."""
    if not 0 <= int(region) < 27:
        raise ValueError(f"Region {region} is outside 0-26")
    picks = (int(region) // 9, int(region) // 3 % 3, int(region) % 3)
    return tuple((edges[i], edges[i + 1]) for edges, i in zip(POSE_REGION_EDGES.values(), picks, strict=True))


@dataclass(frozen=True)
class BodyConfig:
    """The body distribution Wang's pipeline samples from."""

    model_dir: Path = field(default_factory=lambda: _DEFAULT_MODEL_DIR)
    gender: str = "neutral"
    """``neutral``, ``male``, ``female``, or ``random``."""
    num_betas: int = 10
    beta_range: tuple[float, float] = (-2.0, 5.0)
    height_range: tuple[float, float] | None = (1.5, 1.9)
    pose_mode: str = "dressing"
    """``dressing`` seats the body and randomises the right arm; ``dressing-simple``
    narrows that randomisation; ``dressing-standing`` keeps the body upright."""
    hand_pca: float = -1.0
    """Right-hand PCA coefficients, pinned as in the reference so the fist stays relaxed."""
    device: str = "cuda"
    """Device the SMPL-X forward runs on. Construction is always on the CPU because the
    model registers buffers and converts some of them to numpy while it builds, which
    fails under a device default; the evaluation then moves to this device."""

    def to_dict(self) -> dict:
        return {
            "model_dir": str(self.model_dir), "gender": self.gender, "num_betas": int(self.num_betas), "device": self.device,
            "beta_range": list(self.beta_range), "height_range": list(self.height_range) if self.height_range else None,
            "pose_mode": self.pose_mode, "hand_pca": float(self.hand_pca),
        }


@dataclass(frozen=True)
class Body:
    """One generated body: full mesh, right-arm collider, and dressing landmarks."""

    vertices: np.ndarray
    faces: np.ndarray
    joints: np.ndarray
    arm_indices: np.ndarray
    arm_points: np.ndarray
    arm_faces: np.ndarray
    landmarks: dict[str, np.ndarray]
    seed: int
    gender: str
    height_m: float

    @property
    def finger(self) -> np.ndarray:
        return self.landmarks["right_finger"]

    @property
    def elbow(self) -> np.ndarray:
        return self.landmarks["right_elbow"]

    @property
    def shoulder(self) -> np.ndarray:
        return self.landmarks["right_shoulder"]


def sample_body_pose(rng: np.random.Generator, mode: str, region: int | None = None) -> np.ndarray:
    """The seated, right-arm-forward dressing pose of ``gen_human_mesh.py``, optionally inside one region."""
    pose = np.zeros(SMPLX_BODY_JOINTS * 3, dtype=np.float32)
    mode = str(mode).lower()
    if mode in {"", "none", "rest", "zero"}:
        return pose

    def set_deg(joint: str, axis: int, degrees: float) -> None:
        pose[_POSE[joint] + axis] = np.deg2rad(float(degrees))

    if mode != "dressing-standing":
        for side in ("left", "right"):
            set_deg(f"{side}_hip", 0, -90.0)
            set_deg(f"{side}_knee", 0, 70.0)
    set_deg("left_shoulder", 2, -45.0)
    set_deg("left_elbow", 1, -90.0)
    if mode == "dressing-simple":
        shoulder_z, elbow_y, elbow_z = (-10.0, 10.0), (80.0, 100.0), (-10.0, 10.0)
    else:
        shoulder_z, elbow_y, elbow_z = (-20.0, 30.0), (70.0, 110.0), (-20.0, 30.0)
    if region is not None:
        # Same three draws in the same order, so only the intervals change.
        shoulder_z, elbow_y, elbow_z = region_intervals(region)
    set_deg("right_shoulder", 2, rng.uniform(*shoulder_z))
    set_deg("right_elbow", 1, rng.uniform(*elbow_y))
    set_deg("right_elbow", 2, rng.uniform(*elbow_z))
    return pose


@contextmanager
def _cpu_default_device():
    """Run with the CPU as torch's default device, restoring whatever was set.

    Genesis calls ``torch.set_default_device`` globally. SMPL-X both registers
    buffers and converts some of them to numpy while it builds, so a model created
    under a device default ends up half on the GPU and fails on the first
    conversion. Body generation is a handful of small products run once per body,
    so pinning it to the CPU costs nothing.
    """
    import torch

    previous = torch.get_default_device() if hasattr(torch, "get_default_device") else None
    torch.set_default_device("cpu")
    try:
        yield
    finally:
        if previous is not None:
            torch.set_default_device(previous)


@lru_cache(maxsize=8)
def _model(model_dir: str, gender: str, num_betas: int):
    import smplx

    with _cpu_default_device():
        return smplx.create(
            model_dir, model_type="smplx", gender=gender, num_betas=int(num_betas),
            use_pca=True, num_pca_comps=6, batch_size=1,
        )


@lru_cache(maxsize=8)
def right_arm_vertex_indices(model_dir: str, gender: str, num_betas: int) -> tuple[int, ...]:
    """Vertices whose dominant skinning joint is on the right collar-to-fingertip chain."""
    from smplx.joint_names import JOINT_NAMES

    model = _model(model_dir, gender, num_betas)
    weights = model.lbs_weights.detach().cpu().numpy()
    chain = {
        index
        for index, name in enumerate(JOINT_NAMES[: weights.shape[1]])
        if name.startswith("right_")
        and any(part in name for part in ("collar", "shoulder", "elbow", "wrist", "index", "middle", "pinky", "ring", "thumb"))
    }
    return tuple(int(i) for i in np.flatnonzero(np.isin(weights.argmax(axis=1), sorted(chain))))


def submesh(vertices: np.ndarray, faces: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Faces whose three vertices are all selected, renumbered onto the selection."""
    keep = np.zeros(len(vertices), dtype=bool)
    keep[np.asarray(indices, dtype=np.int64)] = True
    rows = np.asarray(faces, dtype=np.int64)
    selected = rows[keep[rows].all(axis=1)]
    remap = np.full(len(vertices), -1, dtype=np.int64)
    order = np.flatnonzero(keep)
    remap[order] = np.arange(len(order))
    return np.asarray(vertices, dtype=np.float64)[order], remap[selected].astype(np.int32)


def right_hand_tip(joints: np.ndarray) -> np.ndarray:
    """Fingertip of the articulated right hand: the mean of the four most distal hand joints."""
    joints = np.asarray(joints, dtype=np.float64)
    wrist, elbow = joints[JOINT_INDEX["right_wrist"]], joints[JOINT_INDEX["right_elbow"]]
    outward = wrist - elbow
    length = float(np.linalg.norm(outward))
    if length < 1.0e-9 or joints.shape[0] <= _RIGHT_HAND_JOINTS[0]:
        return wrist + np.array([0.10, 0.0, 0.0])
    outward = outward / length
    hand = joints[_RIGHT_HAND_JOINTS[0] : min(_RIGHT_HAND_JOINTS[1], joints.shape[0])]
    hand = hand[np.isfinite(hand).all(axis=1)]
    if len(hand) == 0:
        return wrist + outward * 0.10
    tip = hand[np.argsort((hand - wrist[None, :]) @ outward)[-min(4, len(hand)) :]].mean(axis=0)
    return tip if float(np.linalg.norm(tip - wrist)) >= 0.025 else wrist + outward * 0.10


def _reward_line_landmarks(vertices: np.ndarray, joints: np.ndarray) -> dict[str, np.ndarray]:
    """Wrist, finger, elbow, and shoulder on the limb axis, as the reward line needs them.

    Each section's landmark is the midpoint of its upper and lower surface
    vertex, so the line runs through the limb rather than along the skin; the
    finger keeps the joint-derived hand offset from the wrist.
    """
    line = 0.5 * (np.asarray(vertices)[list(REWARD_LINE_UPPER)] + np.asarray(vertices)[list(REWARD_LINE_LOWER)])
    hand_offset = right_hand_tip(joints) - np.asarray(joints)[JOINT_INDEX["right_wrist"]]
    return {
        "right_wrist": line[0],
        "right_finger": line[0] + hand_offset,
        "right_elbow": line[1],
        "right_shoulder": line[2],
    }


def body_parameters(seed: int, cfg: BodyConfig | None = None):
    """``(rng, gender, betas, pose)`` for a body id, drawn in :func:`generate_body`'s order.

    The generator is returned part way through, because the body's height is drawn from
    it later.
    """
    cfg = cfg or BodyConfig()
    rng = np.random.default_rng(int(seed))
    gender = cfg.gender
    if gender == "random":
        gender = str(rng.choice(["male", "female"]))
    betas = rng.uniform(*cfg.beta_range, size=cfg.num_betas).astype(np.float32)
    pose = sample_body_pose(rng, cfg.pose_mode, pose_region(seed))
    return rng, gender, betas, pose


def generate_body(seed: int, cfg: BodyConfig | None = None) -> Body:
    """Sample one SMPL-X dressing body, upright in the +Z-up simulation frame."""
    import torch

    cfg = cfg or BodyConfig()
    rng, gender, betas, pose = body_parameters(seed, cfg)
    model = _model(str(cfg.model_dir), gender, cfg.num_betas)
    device = cfg.device if (cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    hand = np.full(6, float(cfg.hand_pca), dtype=np.float32)

    # SMPL-X is +Y up; a +90 degree turn about X sends +Y to +Z, as the reference does.
    angle = np.deg2rad(90.0)
    rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]])

    def evaluate(body_pose: np.ndarray):
        out = model(
            betas=torch.as_tensor(betas, device=device).unsqueeze(0),
            body_pose=torch.as_tensor(body_pose, device=device).unsqueeze(0),
            right_hand_pose=torch.as_tensor(hand, device=device).view(1, -1),
            return_verts=True,
        )
        return (out.vertices.detach().cpu().numpy()[0].astype(np.float64) @ rot.T,
                out.joints.detach().cpu().numpy()[0].astype(np.float64) @ rot.T)

    scale = 1.0
    if cfg.height_range is not None:
        rest_vertices, _ = evaluate(np.zeros_like(pose))
        rest_height = float(np.ptp(rest_vertices[:, 2]))
        target = float(rng.uniform(*cfg.height_range))
        scale = target / rest_height if rest_height > 1.0e-9 else 1.0
    vertices, joints = evaluate(pose)
    vertices, joints = vertices * scale, joints * scale
    ground = float(vertices[:, 2].min())
    vertices[:, 2] -= ground
    joints[:, 2] -= ground
    if vertices.shape[0] != SMPLX_VERTEX_COUNT:
        raise RuntimeError(f"SMPL-X returned {vertices.shape[0]} vertices, expected {SMPLX_VERTEX_COUNT}")

    faces = np.asarray(getattr(model, "faces_tensor", model.faces).cpu() if hasattr(getattr(model, "faces_tensor", model.faces), "cpu") else model.faces, dtype=np.int32)
    arm_indices = np.asarray(right_arm_vertex_indices(str(cfg.model_dir), gender, cfg.num_betas), dtype=np.int64)
    arm_points, arm_faces = submesh(vertices, faces, arm_indices)
    landmarks = {name: joints[index].copy() for name, index in JOINT_INDEX.items()}
    landmarks.update(_reward_line_landmarks(vertices, joints))
    return Body(
        vertices=vertices,
        faces=faces,
        joints=joints,
        arm_indices=arm_indices,
        arm_points=arm_points,
        arm_faces=arm_faces,
        landmarks=landmarks,
        seed=int(seed),
        gender=gender,
        height_m=float(np.ptp(vertices[:, 2])),
    )
