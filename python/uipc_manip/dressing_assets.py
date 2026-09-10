"""Pre-worn dressing cells baked for the Newton dressing teacher.

The Newton branch trains from a SoftGym-style cache: for every accepted
(garment, human) cell the cache holds the garment already slipped over the
hand, the human's right-arm collision mesh, the held cuff patch, the sleeve
opening polygon, the arm landmarks, and a scripted hand-to-shoulder pull. This
module reads that cache and prepares what the IPC environment needs from it.
The cache itself is not copied into this repository; its location is given by
``DressingCacheConfig`` or the ``UIPC_MANIP_DRESSING_CACHE`` environment
variable.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path("/home/ge47gax/kun/ppf-contact-solver/tools/dressing_bake/out/hand_cached_states.pkl")
DEFAULT_SEMANTICS_DIR = Path(
    "/home/ge47gax/kun/newton-fmvp/exts/newton_isaaclab_tasks/newton_isaaclab_tasks/dressing/data/canonical_drape"
)
CACHED_GARMENTS = ("tshirt_26", "tshirt_392", "tshirt_68", "hospital_gown")
"""Every garment the bake cache carries. Not all of them build for every body; the
environment's ``DEFAULT_GARMENTS`` is the subset that does."""


@dataclass
class DressingCacheConfig:
    cache_path: Path = field(default_factory=lambda: Path(os.environ.get("UIPC_MANIP_DRESSING_CACHE", DEFAULT_CACHE)))
    semantics_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("UIPC_MANIP_DRESSING_SEMANTICS", DEFAULT_SEMANTICS_DIR))
    )


@dataclass
class DressingCell:
    """One pre-worn (garment, human) configuration."""

    garment: str
    human: int
    cloth: np.ndarray
    faces: np.ndarray
    grasp_idx: np.ndarray
    picker_idx: np.ndarray
    opening_idx: np.ndarray
    alignment_idx: np.ndarray
    picker_pos: np.ndarray
    arm_points: np.ndarray
    arm_faces: np.ndarray
    human_points: np.ndarray
    landmarks: dict[str, np.ndarray]
    pull_waypoints: np.ndarray
    opening_radius_mean: float

    @property
    def name(self) -> str:
        return f"{self.garment}__human_{self.human}"

    @property
    def finger(self) -> np.ndarray:
        return self.landmarks["right_finger"]

    @property
    def elbow(self) -> np.ndarray:
        return self.landmarks["right_elbow"]

    @property
    def shoulder(self) -> np.ndarray:
        return self.landmarks["right_shoulder"]

    @property
    def cuff_idx(self) -> np.ndarray:
        """Grasp patch plus opening ring, the Newton ``cuff_particle_indices``."""
        return np.array(list(dict.fromkeys([*self.grasp_idx.tolist(), *self.opening_idx.tolist()])), dtype=np.int64)

    def polygon_triangles(self) -> np.ndarray:
        """Fan triangulation of the six-point opening polygon (Newton ``polygon_triangle_indices``)."""
        p = self.opening_idx
        return np.array([(p[0], p[i], p[i + 1]) for i in range(1, len(p) - 1)], dtype=np.int64)

    def anchor_indices(self, count: int) -> np.ndarray:
        """The picker vertices plus the grasp vertices nearest the picker, ``count`` in total.

        Newton's ``picker_patch`` anchor with ``max_anchor_particles`` pins a
        small local patch; pinning the whole 150-vertex grasp region turns the
        cuff into a rigid cap that the arm cannot enter.
        """
        picker = [int(i) for i in self.picker_idx]
        rest = [int(i) for i in self.grasp_idx if int(i) not in picker]
        order = np.argsort(np.linalg.norm(self.cloth[rest] - self.picker_pos[None, :], axis=1))
        chosen = picker + [rest[i] for i in order[: max(0, int(count) - len(picker))]]
        return np.asarray(chosen[: int(count)], dtype=np.int64)


class DressingCache:
    def __init__(self, cfg: DressingCacheConfig | None = None) -> None:
        self.cfg = cfg or DressingCacheConfig()
        path = Path(self.cfg.cache_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dressing cache {path} not found; set UIPC_MANIP_DRESSING_CACHE to the hand_cached_states.pkl path"
            )
        with path.open("rb") as handle:
            self._configs, self._states = pickle.load(handle)
        self.cells: list[tuple[str, int]] = [
            (str(c["garment"]), int(str(c["human_id"]).split("_")[-1])) for c in self._configs
        ]

    def humans(self) -> list[int]:
        return sorted({h for _, h in self.cells})

    def garments_for(self, human: int) -> list[str]:
        return [g for g, h in self.cells if h == int(human)]

    def _semantics(self, config: dict) -> dict:
        npz_name = Path(str(config["garment_npz"])).name
        path = Path(self.cfg.semantics_dir) / npz_name.replace(".npz", "_semantics.json")
        if not path.exists():
            raise FileNotFoundError(f"Garment semantics {path} not found; set UIPC_MANIP_DRESSING_SEMANTICS")
        return json.loads(path.read_text())["semantic_labels"]["right_cuff_opening"]

    def load(self, garment: str, human: int) -> DressingCell:
        try:
            index = self.cells.index((str(garment), int(human)))
        except ValueError as exc:
            raise KeyError(f"Cell ({garment}, human_{human}) is not in the cache; available: {self.cells}") from exc
        config, state = self._configs[index], self._states[index]
        semantics = self._semantics(config)
        cloth = np.asarray(state["particle_q"], dtype=np.float64)
        landmarks = {k: np.asarray(v, dtype=np.float64) for k, v in state["human_joint_positions"].items()}
        return DressingCell(
            garment=str(garment),
            human=int(human),
            cloth=cloth,
            faces=np.asarray(state["cloth_faces"], dtype=np.int32),
            grasp_idx=np.asarray(state["grasp_indices"], dtype=np.int64).reshape(-1),
            picker_idx=np.asarray(state["picker_indices"], dtype=np.int64).reshape(-1),
            opening_idx=np.asarray(state["opening_indices"], dtype=np.int64).reshape(-1),
            alignment_idx=np.asarray(semantics["alignment_line_indices"], dtype=np.int64).reshape(2),
            picker_pos=np.asarray(state["picker_pos"], dtype=np.float64).reshape(3),
            arm_points=np.asarray(state["right_arm_points"], dtype=np.float64),
            arm_faces=np.asarray(state["right_arm_faces"], dtype=np.int32),
            human_points=np.asarray(state["human_mesh_points"], dtype=np.float64),
            landmarks=landmarks,
            pull_waypoints=np.asarray(state["pull_waypoints"], dtype=np.float64).reshape(-1, 3),
            opening_radius_mean=float(config.get("opening_radius_mean_m", semantics.get("opening_radius_mean_m", 0.0))),
        )


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals; orientation follows the face winding."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def erode_arm_mesh(verts: np.ndarray, faces: np.ndarray, depth: float, axis_a: np.ndarray, axis_b: np.ndarray):
    """Shrink the arm surface inward by ``depth`` metres along outward normals.

    The cached pre-worn garments were accepted with centimetre-scale
    interpenetration against the human, which libuipc's build-time checks
    refuse. Eroding the collider by a few millimetres is how the Genesis
    sim2sim path made those states intersection free. Outwardness is decided
    against the finger-to-shoulder axis so a flipped winding cannot inflate
    the arm instead.
    """
    v = np.asarray(verts, dtype=np.float64)
    n = vertex_normals(v, faces)
    a, b = np.asarray(axis_a, dtype=np.float64), np.asarray(axis_b, dtype=np.float64)
    d = b - a
    t = np.clip(((v - a) @ d) / max(float(d @ d), 1e-12), 0.0, 1.0)
    radial = v - (a + t[:, None] * d)
    if float(np.mean(np.sum(n * radial, axis=1) > 0.0)) < 0.5:
        n = -n
    return v - float(depth) * n


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for x, y, z in np.asarray(verts, dtype=np.float64):
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in np.asarray(faces, dtype=np.int64):
            handle.write(f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}\n")
    return path
