"""Assemble dressing cells from canonical garment drapes and any cached body.

The Newton bake produced 23 pre-worn (garment, human) states by simulating the
garment onto one body; six of them interpenetrate the arm badly enough that
libuipc refuses them, and every accepted one needs the arm eroded 6 mm. That
pipeline also fixes which garments exist for which body.

This module builds the same :class:`~uipc_manip.dressing_assets.DressingCell`
without any pre-baked state. :mod:`uipc_manip.dressing_bake` drapes the raw
garment mesh online in libuipc; placing that drape on a body is then a rigid
transform, ported from Newton's ``runtime_align``: build a socket frame from the
arm landmarks, put the opening one clearance step *outside* the fingertip along
the arm axis, and map the canonical socket onto it. The garment therefore starts
in free air in front of the hand, never inside the arm, so any garment composes
with any body and no arm erosion is required.

Bodies come from the cached states, which carry the SMPL-X arm submesh, the full
body point cloud, and the joint landmarks. The garment axis is free, so the
23-cell bake becomes every (garment, body) pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dressing_assets import DressingCache, DressingCell
from .dressing_bake import PULL_SCHEDULES, BakeConfig, bake_in_subprocess, load_index_tables

OFFLINE_DRAPE_DIR = (
    "/home/ge47gax/kun/newton-fmvp/exts/newton_isaaclab_tasks/newton_isaaclab_tasks/dressing/data/canonical_drape"
)
OFFLINE_MESH_DIR = "/home/ge47gax/kun/newton-fmvp/garments"


@dataclass(frozen=True)
class LiveCellConfig:
    """How the garment is baked and how far in front of the hand it spawns."""

    bake: BakeConfig = field(default_factory=BakeConfig)
    clearance_m: float = 0.10
    """Distance the opening centre sits outside the fingertip along the arm axis.
    Newton's runtime alignment uses the same quantity so the arm has to travel
    before it reaches the opening plane; 0.10 m keeps every cached body at least
    2 mm clear of the garment, twice the contact activation distance."""
    scales: dict[str, float] | None = None
    """Per-garment mesh scale; ``None`` uses each garment's bake default."""

    def to_dict(self) -> dict:
        return {"bake": self.bake.to_dict(), "clearance_m": float(self.clearance_m), "scales": dict(self.scales or {})}


def _unit(v: np.ndarray, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return np.asarray(fallback, dtype=np.float64) if n < 1.0e-12 else v / n


def build_target_socket_frame(
    finger: np.ndarray, shoulder: np.ndarray, *, clearance: float, world_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """World socket frame for an arm: +Z along finger to shoulder, origin outside the fingertip.

    Ported from Newton ``runtime_align.build_target_socket_frame``.
    """
    finger = np.asarray(finger, dtype=np.float64).reshape(3)
    shoulder = np.asarray(shoulder, dtype=np.float64).reshape(3)
    arm_axis = _unit(shoulder - finger)
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up - arm_axis * float(np.dot(up, arm_axis))
    y_axis = _unit(up, fallback=(0.0, 0.0, 1.0))
    x_axis = _unit(np.cross(y_axis, arm_axis), fallback=(0.0, 1.0, 0.0))
    y_axis = _unit(np.cross(arm_axis, x_axis), fallback=(0.0, 0.0, 1.0))
    T = np.eye(4, dtype=np.float64)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_axis, y_axis, arm_axis
    T[:3, 3] = finger - arm_axis * float(clearance)
    return T


def canonical_to_world_transform(
    socket_to_canonical: np.ndarray, finger: np.ndarray, shoulder: np.ndarray, *, clearance: float
) -> np.ndarray:
    """``T_target @ inv(T_source)``: map canonical drape points onto this arm."""
    source = np.asarray(socket_to_canonical, dtype=np.float64).reshape(4, 4)
    target = build_target_socket_frame(finger, shoulder, clearance=clearance)
    rotation, translation = source[:3, :3], source[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return target @ inverse


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transform, dtype=np.float64)
    return pts @ t[:3, :3].T + t[:3, 3][None, :]


@dataclass(frozen=True)
class CanonicalDrape:
    """One garment hanging open in its canonical frame, with cuff semantics."""

    garment: str
    scale: float
    cloth: np.ndarray
    faces: np.ndarray
    anchor: np.ndarray
    grasp_idx: np.ndarray
    picker_idx: np.ndarray
    opening_idx: np.ndarray
    alignment_idx: np.ndarray
    socket_to_canonical: np.ndarray
    opening_radius_mean: float
    source: str = "online"
    """``online`` when libuipc baked it here, ``offline`` for Newton's pre-baked drape."""

    def place(self, cell_landmarks: dict[str, np.ndarray], *, clearance: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the garment vertices and picker position placed on this arm."""
        transform = canonical_to_world_transform(
            self.socket_to_canonical, cell_landmarks["right_finger"], cell_landmarks["right_shoulder"], clearance=clearance
        )
        return apply_transform(self.cloth, transform), apply_transform(self.anchor[None, :], transform)[0]


def available_garments(cfg: LiveCellConfig | None = None) -> list[str]:
    """Garments whose raw mesh, pull schedule, and index tables are all present."""
    cfg = cfg or LiveCellConfig()
    tables = load_index_tables(cfg.bake.index_module)
    found = []
    for name in sorted(PULL_SCHEDULES):
        from .dressing_bake import RAW_MESH_FILENAME

        mesh = Path(cfg.bake.garment_dir) / RAW_MESH_FILENAME.get(name, f"{name}.obj")
        if mesh.exists() and name in tables.grasping_particle_indices and name in tables.shoulder_polygon_particle_indices:
            found.append(name)
    return found


def load_drape(garment: str, cfg: LiveCellConfig | None = None, *, reuse: bool = True) -> CanonicalDrape:
    """Drape one garment in libuipc, falling back to Newton's offline drape.

    Three of the five raw meshes carry triangles that genuinely cross in the
    canonical rest pose, which libuipc refuses and neither separating the layers
    nor dropping the faces repairs. Those garments keep the offline drape, which
    is the same geometry the previous pipeline used; the rest are baked here.
    """
    cfg = cfg or LiveCellConfig()
    scale = (cfg.scales or {}).get(garment)
    del reuse  # The bake is always cached by a content hash of its inputs.
    try:
        baked = bake_in_subprocess(garment, scale=scale, cfg=cfg.bake)
        source = "online"
    except RuntimeError as exc:
        baked = load_offline_drape(garment, cfg)
        if baked is None:
            raise RuntimeError(f"{garment}: no online bake and no offline drape") from exc
        source = "offline"
    return CanonicalDrape(
        garment=garment,
        scale=float(baked["scale"]),
        cloth=baked["cloth"],
        faces=baked["faces"],
        anchor=baked["anchor"],
        grasp_idx=baked["grasp_idx"],
        picker_idx=baked["picker_idx"],
        opening_idx=baked["opening_idx"],
        alignment_idx=baked["alignment_idx"],
        socket_to_canonical=baked["socket_to_canonical"],
        opening_radius_mean=float(baked["opening_radius_mean_m"]),
        source=source,
    )


def load_offline_drape(garment: str, cfg: LiveCellConfig | None = None) -> dict | None:
    """Newton's pre-baked canonical drape for one garment, if it is on disk.

    The npz holds the draped vertices plus one appended kinematic anchor; faces
    come from the exported drape mesh, whose topology it shares.
    """
    from .dressing_bake import cuff_semantics

    cfg = cfg or LiveCellConfig()
    drape_dir = Path(OFFLINE_DRAPE_DIR)
    mesh_dir = Path(OFFLINE_MESH_DIR)
    tables = load_index_tables(cfg.bake.index_module)
    for path in sorted(drape_dir.glob(f"{garment}__s*.npz")):
        scale = float(path.stem.split("__s")[1])
        mesh = mesh_dir / f"{garment}_final.obj"
        if not mesh.exists():
            continue
        from .assets import load_obj

        vertices, faces = load_obj(mesh)
        with np.load(path, allow_pickle=True) as data:
            cloth = np.asarray(data["particle_q"], dtype=np.float64)[: len(vertices)]
            grasp = np.asarray(data["right_cuff_grasp_indices"], dtype=np.int64).reshape(-1)
            opening = np.asarray(data["right_cuff_opening_indices"], dtype=np.int64).reshape(-1)
            picker = np.asarray(data["right_cuff_picker_indices"], dtype=np.int64).reshape(-1)
        alignment = np.asarray(tables.alignment_line_indices[garment], dtype=np.int64).reshape(-1)
        semantics = cuff_semantics(cloth, opening, alignment)
        return {
            "garment": garment,
            "scale": scale,
            "cloth": cloth,
            "faces": np.asarray(faces, dtype=np.int32),
            "anchor": cloth[grasp].mean(axis=0),
            "grasp_idx": grasp,
            "picker_idx": picker[picker < len(vertices)],
            "opening_idx": opening,
            "alignment_idx": alignment,
            **semantics,
        }
    return None


class LiveCellFactory:
    """Build dressing cells by draping any garment online and placing it on any cached body."""

    def __init__(self, cfg: LiveCellConfig | None = None, cache: DressingCache | None = None) -> None:
        self.cfg = cfg or LiveCellConfig()
        self.cache = cache or DressingCache()
        self._drapes: dict[str, CanonicalDrape] = {}
        self.garments = available_garments(self.cfg)
        if not self.garments:
            raise FileNotFoundError(f"No bakeable garment found under {self.cfg.bake.garment_dir}")

    def humans(self) -> list[int]:
        return self.cache.humans()

    def cells(self) -> list[tuple[str, int]]:
        """Every (garment, human) pair this factory can build, against the bake's 23."""
        return [(g, h) for h in self.humans() for g in self.garments]

    def drape(self, garment: str) -> CanonicalDrape:
        if garment not in self._drapes:
            self._drapes[garment] = load_drape(garment, self.cfg)
        return self._drapes[garment]

    def _body(self, human: int):
        """Any cached cell of this human carries the same body; take the first."""
        for garment, h in self.cache.cells:
            if h == int(human):
                return self.cache.load(garment, h)
        raise KeyError(f"Human {human} is not in the body cache; available: {self.cache.humans()}")

    def build(self, garment: str, human: int) -> DressingCell:
        drape = self.drape(garment)
        body = self._body(human)
        cloth, picker_pos = drape.place(body.landmarks, clearance=self.cfg.clearance_m)
        return DressingCell(
            garment=garment,
            human=int(human),
            cloth=cloth,
            faces=drape.faces,
            grasp_idx=drape.grasp_idx,
            picker_idx=drape.picker_idx,
            opening_idx=drape.opening_idx,
            alignment_idx=drape.alignment_idx,
            picker_pos=picker_pos,
            arm_points=body.arm_points,
            arm_faces=body.arm_faces,
            human_points=body.human_points,
            landmarks=body.landmarks,
            # The bake's waypoints belong to its own pre-worn state; a live cell
            # pulls straight from the spawn point to the shoulder.
            pull_waypoints=np.stack([picker_pos, body.landmarks["right_shoulder"]]),
            opening_radius_mean=drape.opening_radius_mean,
        )

    def clearances(self, garment: str, human: int) -> dict[str, float]:
        """Smallest distances from the placed garment to the arm and to the fingertip."""
        cell = self.build(garment, human)
        arm = np.linalg.norm(cell.cloth[:, None, :] - cell.arm_points[None, :, :], axis=-1).min()
        opening = np.linalg.norm(cell.cloth[cell.opening_idx] - cell.finger[None, :], axis=-1).min()
        return {"cloth_to_arm_m": float(arm), "opening_to_finger_m": float(opening)}
