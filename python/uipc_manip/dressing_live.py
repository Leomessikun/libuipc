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
the forearm axis, and map the canonical socket onto it. The garment therefore starts
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
from .dressing_body import BodyConfig, generate_body

OFFLINE_DRAPE_DIR = (
    "/home/ge47gax/kun/newton-fmvp/exts/newton_isaaclab_tasks/newton_isaaclab_tasks/dressing/data/canonical_drape"
)
OFFLINE_MESH_DIR = "/home/ge47gax/kun/newton-fmvp/garments"


@dataclass(frozen=True)
class LiveCellConfig:
    """How the garment is baked and how far in front of the hand it spawns."""

    bake: BakeConfig = field(default_factory=BakeConfig)
    pre_insertion: bool = True
    """Spawn the garment before insertion, its opening facing the fingertips and its
    body hanging clear of the arm, instead of the reference's pre-worn placement,
    which lays the sleeve through the arm and which libuipc refuses."""
    clearance_m: float = 0.09
    """Distance the opening centre starts outside the fingertip along the insertion axis.
    Wang's cached states hold the opening 8.8 to 9.4 cm outside the fingertip, coaxial
    with the forearm. This is the starting value: a cell whose garment would pass within
    ``min_arm_gap_m`` of the arm moves further out in 1 cm steps, see
    :meth:`LiveCellFactory.clearance_for`. The previous fixed 0.20 m cost 14 cm of extra
    travel against Newton's 6 cm."""
    axis_landmark: str = "right_elbow"
    """Landmark the insertion axis runs toward from the fingertip. The forearm is the axis
    the hand travels along first; the fingertip-to-shoulder chord it replaced is 31 to 38
    degrees off the forearm, which started the opening about one opening radius off that
    axis (10.3 to 12.4 cm, measured at bodies 0 and 3), and the scripted expert then
    threaded only where its approach happened to re-centre the opening. ``right_shoulder``
    restores the chord."""
    min_arm_gap_m: float = 0.003
    """Smallest surface distance a placed cell may start with, with no garment edge through an
    arm triangle or the reverse (see :func:`garment_arm_gap`), which is what libuipc's build
    checks. On the forearm axis it agrees with the build on every cell tried: tshirt_26 bodies
    0 and 3 legal at 0.09 and 0.12 m, tshirt_4 body 0 refused at 0.09 m for an edge through an
    arm triangle and legal at 0.12 m. The 15 mm vertex-to-vertex gap it replaces under-read
    contact by up to half an edge and pushed tshirt_26 out to 0.17 to 0.20 m, costing travel."""
    max_clearance_m: float = 0.50
    """Furthest the opening is moved out while looking for ``min_arm_gap_m``."""
    reach_tolerance_m: float = 0.05
    """How much further past the opening an online drape may reach than Newton's offline drape
    of the same garment before the offline one is used. The online bake cannot hang a long
    sleeve: on tshirt_392 and tshirt_4 the sleeve beyond the (armhole) opening stays a drooping
    cantilever reaching 0.44 m, against 0.15 and 0.25 m in the reference where it hangs down the
    body, and pre-insertion placement lays that sleeve along the forearm. A 6 s settle leaves it
    at 0.44 m, so this is the sleeve's statics under libuipc's strain-limited shell, not time."""
    hang_as_baked: bool = False
    """Re-roll the canonical socket so the garment keeps its baked hang (see
    :func:`gravity_aligned_socket`). Off by default because it costs the expert the one
    garment it could dress: on tshirt_26, body 0, 150 decisions of six steps at a 1e4 hold,
    the measured socket reaches forearm 1.0 and upper arm 0.32 with a 4.2 mm held-cuff error,
    the re-rolled one forearm 0.0 with 22.7 mm. tshirt_392's frame is upside down under the
    measured socket, but that is not what kept it off the arm; its offline drape clears the
    arm by 111 to 128 mm at either roll."""
    scales: dict[str, float] | None = None
    """Per-garment mesh scale; ``None`` uses each garment's bake default."""
    body: BodyConfig = field(default_factory=BodyConfig)
    bodies: str = "smplx"
    """``smplx`` generates each body here from shape and pose samples; ``cache`` reads
    the eight bodies of the Newton state file instead."""

    def to_dict(self) -> dict:
        return {
            "bake": self.bake.to_dict(), "clearance_m": float(self.clearance_m),
            "scales": dict(self.scales or {}), "bodies": self.bodies, "body": self.body.to_dict(),
            "pre_insertion": bool(self.pre_insertion), "hang_as_baked": bool(self.hang_as_baked),
            "axis_landmark": str(self.axis_landmark),
            "min_arm_gap_m": float(self.min_arm_gap_m),
            "max_clearance_m": float(self.max_clearance_m), "reach_tolerance_m": float(self.reach_tolerance_m),
        }


def _unit(v: np.ndarray, fallback: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return np.asarray(fallback, dtype=np.float64) if n < 1.0e-12 else v / n


def build_target_socket_frame(
    finger: np.ndarray,
    axis_end: np.ndarray,
    *,
    clearance: float,
    world_up: tuple[float, float, float] = (0.0, 0.0, 1.0),
    pre_insertion: bool = True,
) -> np.ndarray:
    """World socket frame for an arm, with the opening a clearance step outside the fingertip.

    The arm axis runs from ``finger`` toward ``axis_end``: the elbow for the forearm
    axis, the shoulder for Newton's chord. The canonical socket's +Z runs from the
    cuff into the sleeve, so aligning it with that axis lays the sleeve along the arm: that is a
    *pre-worn* state, and it is what Newton's ``runtime_align`` builds, because
    VBD tolerates the interpenetration that follows. libuipc refuses it, and a
    training episode should start before insertion anyway.

    With ``pre_insertion`` the sleeve is turned to run the other way, so the
    opening faces the fingertips and the garment hangs off the end of the hand in
    free space. The tool then has to carry the opening over the hand, which is
    the motion the task is about.
    """
    finger = np.asarray(finger, dtype=np.float64).reshape(3)
    axis_end = np.asarray(axis_end, dtype=np.float64).reshape(3)
    arm_axis = _unit(axis_end - finger)
    insertion = -arm_axis if pre_insertion else arm_axis
    up = np.asarray(world_up, dtype=np.float64).reshape(3)
    up = up - insertion * float(np.dot(up, insertion))
    y_axis = _unit(up, fallback=(0.0, 0.0, 1.0))
    x_axis = _unit(np.cross(y_axis, insertion), fallback=(0.0, 1.0, 0.0))
    y_axis = _unit(np.cross(insertion, x_axis), fallback=(0.0, 0.0, 1.0))
    T = np.eye(4, dtype=np.float64)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_axis, y_axis, insertion
    T[:3, 3] = finger - arm_axis * float(clearance)
    return T


def gravity_aligned_socket(
    socket_to_canonical: np.ndarray, up: tuple[float, float, float] = (0.0, 0.0, 1.0)
) -> np.ndarray:
    """Re-roll a canonical socket about its insertion axis so its +Y is the bake's up.

    The measured socket frame takes its +X from the first opening-polygon vertex,
    so its roll about the insertion axis is arbitrary per garment, while the target
    frame's +Y is world up. Mapping one onto the other then turns each garment by
    whatever angle its first ring vertex happens to sit at: tshirt_392's sits 90
    degrees from up, which turns that drape upside down, and tshirt_26 and tshirt_68
    are turned 50 and 64 degrees. Building the source
    +Y from the canonical up, as the target builds its own from world up, makes every
    garment hang the way it hung in the bake. Origin and insertion axis are kept.
    Opt-in through ``LiveCellConfig.hang_as_baked``; the scripted expert dresses
    tshirt_26 only under the measured socket.
    """
    T = np.asarray(socket_to_canonical, dtype=np.float64).reshape(4, 4).copy()
    z_axis = _unit(T[:3, 2], fallback=(0.0, 0.0, 1.0))
    up_v = np.asarray(up, dtype=np.float64).reshape(3)
    y_axis = _unit(up_v - z_axis * float(np.dot(up_v, z_axis)), fallback=tuple(T[:3, 1]))
    x_axis = _unit(np.cross(y_axis, z_axis), fallback=tuple(T[:3, 0]))
    y_axis = _unit(np.cross(z_axis, x_axis), fallback=(0.0, 0.0, 1.0))
    T[:3, 0], T[:3, 1], T[:3, 2] = x_axis, y_axis, z_axis
    return T


def canonical_to_world_transform(
    socket_to_canonical: np.ndarray,
    finger: np.ndarray,
    axis_end: np.ndarray,
    *,
    clearance: float,
    pre_insertion: bool = True,
    hang_as_baked: bool = False,
) -> np.ndarray:
    """``T_target @ inv(T_source)``: map canonical drape points onto this arm.

    With ``hang_as_baked`` the source socket is first re-rolled by
    :func:`gravity_aligned_socket`, so the canonical up lands on world up.
    """
    source = np.asarray(socket_to_canonical, dtype=np.float64).reshape(4, 4)
    if hang_as_baked:
        source = gravity_aligned_socket(source)
    target = build_target_socket_frame(finger, axis_end, clearance=clearance, pre_insertion=pre_insertion)
    rotation, translation = source[:3, :3], source[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return target @ inverse


def reach_past_opening(cloth: np.ndarray, socket_to_canonical: np.ndarray) -> float:
    """How far a drape extends outward past its opening along the insertion axis [m].

    Pre-insertion placement turns that side toward the hand, so this is the length of
    garment that would lie over the arm at zero clearance.
    """
    T = np.asarray(socket_to_canonical, dtype=np.float64).reshape(4, 4)
    along = (np.asarray(cloth, dtype=np.float64).reshape(-1, 3) - T[:3, 3]) @ T[:3, 2]
    return float(max(0.0, -along.min()))


class NoClearPlacement(RuntimeError):
    """No clearance up to ``LiveCellConfig.max_clearance_m`` places this garment clear of this arm.

    Raised instead of forcing an illegal cell into the world; a cell plan should drop the
    (garment, body) pair it names."""


def garment_arm_gap(cloth: np.ndarray, faces: np.ndarray, arm_points: np.ndarray, arm_faces: np.ndarray) -> float:
    """Smallest surface distance between a placed garment and the arm [m]; 0 when they cross.

    Mirrors what libuipc's build refuses: any garment edge through an arm triangle or arm
    edge through a garment triangle, then the true point-to-surface distance both ways. A
    vertex-to-vertex distance misses an edge passing between two vertices. Distances beyond
    5 cm are reported as 5 cm.
    """
    import trimesh
    from scipy.spatial import cKDTree

    from .dressing_bake import mesh_edges, segments_through_triangles

    cloth = np.asarray(cloth, dtype=np.float64)
    arm = np.asarray(arm_points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    arm_faces = np.asarray(arm_faces, dtype=np.int64).reshape(-1, 3)
    if len(segments_through_triangles(cloth, mesh_edges(faces), arm, arm_faces)[0]) or len(
        segments_through_triangles(arm, mesh_edges(arm_faces), cloth, faces)[0]
    ):
        return 0.0
    gaps = [0.05]
    for points, other, other_faces in ((cloth, arm, arm_faces), (arm, cloth, faces)):
        if not len(other_faces):
            gaps.append(float(cKDTree(other).query(points)[0].min()))
            continue
        # Candidates within 5 cm of a triangle, judged from its centroid widened by the
        # largest circumradius, so large triangles are not missed between their vertices.
        tri = other[other_faces]
        centre = tri.mean(axis=1)
        reach = 0.05 + float(np.linalg.norm(tri - centre[:, None], axis=2).max())
        near = points[cKDTree(centre).query(points)[0] < reach]
        if len(near):
            surface = trimesh.Trimesh(other, other_faces, process=False)
            gaps.append(float(trimesh.proximity.closest_point(surface, near)[1].min()))
    return float(min(gaps))


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transform, dtype=np.float64)
    return pts @ t[:3, :3].T + t[:3, 3][None, :]


@dataclass(frozen=True)
class SimpleBody:
    """The four things a dressing cell needs from a body, whatever produced it."""

    arm_points: np.ndarray
    arm_faces: np.ndarray
    human_points: np.ndarray
    landmarks: dict
    source: str


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

    def place(
        self,
        cell_landmarks: dict[str, np.ndarray],
        *,
        clearance: float,
        pre_insertion: bool = True,
        hang_as_baked: bool = False,
        axis_landmark: str = "right_elbow",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the garment vertices and picker position placed for this arm."""
        transform = canonical_to_world_transform(
            self.socket_to_canonical, cell_landmarks["right_finger"], cell_landmarks[axis_landmark],
            clearance=clearance, pre_insertion=pre_insertion, hang_as_baked=hang_as_baked,
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

    The bake untangles the raw meshes' crossing triangles and opens their near
    pairs, so every tshirt bakes. Newton's drape is used instead when the bake
    fails, or when the online drape reaches past the opening more than
    ``reach_tolerance_m`` further than Newton's does: a long sleeve that the bake
    leaves sticking out would lie along the forearm at placement.
    """
    cfg = cfg or LiveCellConfig()
    scale = (cfg.scales or {}).get(garment)
    del reuse  # The bake is always cached by a content hash of its inputs.
    try:
        baked = bake_in_subprocess(garment, scale=scale, cfg=cfg.bake)
        source = "online"
        reference = load_offline_drape(garment, cfg)
        if (
            reference is not None
            and abs(float(reference["scale"]) - float(baked["scale"])) < 1.0e-6
            and reach_past_opening(baked["cloth"], baked["socket_to_canonical"])
            > reach_past_opening(reference["cloth"], reference["socket_to_canonical"]) + cfg.reach_tolerance_m
        ):
            baked, source = reference, "offline"
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
    from .dressing_bake import cuff_semantics, untangle_crossings

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
        # Newton's drape of tshirt_68 carries a fold whose layers cross (seven edge-triangle
        # pairs on eleven vertices), which libuipc refuses; untangle it, as the bake does.
        cloth, _ = untangle_crossings(cloth, faces, margin=cfg.bake.separation_step_m)
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
    """Build dressing cells by draping a garment online and placing it on a body.

    With ``bodies="smplx"`` neither half comes from a saved state: the garment is
    draped in libuipc from its raw mesh and the body is sampled from SMPL-X, so a
    "human" is just a seed and the cell count is unbounded.
    """

    def __init__(self, cfg: LiveCellConfig | None = None, cache: DressingCache | None = None) -> None:
        self.cfg = cfg or LiveCellConfig()
        self._cache = cache
        self._drapes: dict[str, CanonicalDrape] = {}
        self._bodies: dict[int, object] = {}
        self._clearance: dict[tuple[str, int], float] = {}
        self.garments = available_garments(self.cfg)
        if not self.garments:
            raise FileNotFoundError(f"No bakeable garment found under {self.cfg.bake.garment_dir}")

    @property
    def cache(self) -> DressingCache:
        if self._cache is None:
            self._cache = DressingCache()
        return self._cache

    def humans(self, count: int = 8) -> list[int]:
        """Body seeds. Generated bodies are unbounded, so this is just the first ``count``."""
        return list(range(int(count))) if self.cfg.bodies == "smplx" else self.cache.humans()

    def cells(self, humans: int = 8) -> list[tuple[str, int]]:
        """Every (garment, body) pair this factory can build, against the bake's 23."""
        return [(g, h) for h in self.humans(humans) for g in self.garments]

    def drape(self, garment: str) -> CanonicalDrape:
        if garment not in self._drapes:
            self._drapes[garment] = load_drape(garment, self.cfg)
        return self._drapes[garment]

    def body(self, human: int):
        """The body for this seed: generated from SMPL-X, or read from the cache."""
        key = int(human)
        if key not in self._bodies:
            self._bodies[key] = self._generate(key) if self.cfg.bodies == "smplx" else self._cached_body(key)
        return self._bodies[key]

    def _generate(self, seed: int):
        body = generate_body(seed, self.cfg.body)
        return SimpleBody(body.arm_points, body.arm_faces, body.vertices, body.landmarks, f"smplx_seed_{seed}")

    def _cached_body(self, human: int):
        """Any cached cell of this human carries the same body; take the first."""
        for garment, h in self.cache.cells:
            if h == int(human):
                cell = self.cache.load(garment, h)
                return SimpleBody(cell.arm_points, cell.arm_faces, cell.human_points, cell.landmarks, f"cache_human_{human}")
        raise KeyError(f"Human {human} is not in the body cache; available: {self.cache.humans()}")

    def clearance_for(self, garment: str, human: int) -> float:
        """Clearance for this cell: ``clearance_m``, moved out in 1 cm steps until clear of the arm.

        The garment must start at least ``min_arm_gap_m`` from the arm's surface with no
        edge crossing either way (:func:`garment_arm_gap`); a sleeve longer than the
        clearance otherwise lies over the hand on some bodies. Every step is checked on
        its own, because legality is not monotonic in the clearance: on bent arms
        tshirt_26's body panel can reach the elbow at 0.14 m and not at 0.12 m. Raises
        :class:`NoClearPlacement` when nothing up to ``max_clearance_m`` passes.
        """
        key = (str(garment), int(human))
        if key not in self._clearance:
            drape, body = self.drape(garment), self.body(human)
            clearance = float(self.cfg.clearance_m)
            while True:
                if clearance > float(self.cfg.max_clearance_m) + 1.0e-9:
                    raise NoClearPlacement(
                        f"{garment} on body {human}: no clearance from {self.cfg.clearance_m:.2f} to "
                        f"{self.cfg.max_clearance_m:.2f} m leaves {self.cfg.min_arm_gap_m * 1000:.0f} mm to the arm"
                    )
                cloth, _ = drape.place(
                    body.landmarks, clearance=clearance, pre_insertion=self.cfg.pre_insertion,
                    hang_as_baked=self.cfg.hang_as_baked, axis_landmark=self.cfg.axis_landmark,
                )
                if garment_arm_gap(cloth, drape.faces, body.arm_points, body.arm_faces) >= float(self.cfg.min_arm_gap_m):
                    break
                clearance = round(clearance + 0.01, 6)
            self._clearance[key] = clearance
        return self._clearance[key]

    def build(self, garment: str, human: int) -> DressingCell:
        drape = self.drape(garment)
        body = self.body(human)
        cloth, picker_pos = drape.place(
            body.landmarks, clearance=self.clearance_for(garment, human), pre_insertion=self.cfg.pre_insertion,
            hang_as_baked=self.cfg.hang_as_baked, axis_landmark=self.cfg.axis_landmark,
        )
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
        return {
            "cloth_to_arm_m": float(arm), "opening_to_finger_m": float(opening),
            "surface_gap_m": garment_arm_gap(cell.cloth, cell.faces, cell.arm_points, cell.arm_faces),
            "clearance_m": self.clearance_for(garment, human), "source": self.drape(garment).source,
        }
