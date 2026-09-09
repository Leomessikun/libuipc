"""Bake a garment's canonical open drape online, in libuipc, from the raw mesh.

Newton's dressing states were baked offline: `ppf-contact-solver` pulled the
cuff patch open, let gravity settle the body, and wrote a `CanonicalDrape` npz
that this port then read. That step exists because Newton's own VBD cloth
collapses a closed tshirt's cuff opening. libuipc is an IPC solver of the same
class as `ppf-contact-solver`, so the bake does not need to be offline here: the
same pull-and-settle runs in a standalone libuipc scene at environment build
time, from the raw garment mesh, and the drape lands in equilibrium under the
*same* material and solver the episode will use.

The recipe follows `bake_canonical_drape_ppf.py`: apply Newton's canonical
pre-transform (scale, zyx Euler, ground the mesh, Y-up to Z-up), pin the union
of the grasp patch and the six opening-polygon vertices, translate that pin set
along a garment-specific axis, then hold while the fabric drains under gravity.
Semantics (opening centre, insertion axis, socket frame) are then measured off
the result, as `cuff_semantics.compute_right_cuff_semantics` does.

The per-garment index tables live in the `ppf-contact-solver` bake tools; point
`UIPC_MANIP_GARMENT_INDEX_MODULE` at `garment_idx_utils.py` to override.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np

from .assets import load_obj

_DEFAULT_INDEX_MODULE = Path(
    os.environ.get(
        "UIPC_MANIP_GARMENT_INDEX_MODULE",
        "/home/ge47gax/kun/ppf-contact-solver/tools/dressing_bake/garment_idx_utils.py",
    )
)
_DEFAULT_GARMENT_DIR = Path(
    os.environ.get("UIPC_MANIP_RAW_GARMENT_DIR", "/home/ge47gax/kun/ppf-contact-solver/garments")
)

# `bake_canonical_drape_ppf.OBJ_FILENAME`: the raw, unscaled source meshes. The
# `*_final.obj` files beside the Newton checkout are the *baked* drapes exported
# as OBJ, so baking must start from these instead.
RAW_MESH_FILENAME: dict[str, str] = {
    "hospital_gown": "fullgown.obj",
    "tshirt_4": "Tshirt-4.obj",
    "tshirt_26": "Tshirt-26.obj",
    "tshirt_68": "Tshirt-68.obj",
    "tshirt_392": "Tshirt-392.obj",
}


@dataclass(frozen=True)
class PullSchedule:
    """How far and along which canonical axis the pinned cuff is drawn open."""

    direction: tuple[float, float, float]
    distance_m: float
    pull_seconds: float
    settle_seconds: float


# `bake_canonical_drape_ppf._PULL_DEFAULTS`: tshirts open along -Y, the gown along +X.
PULL_SCHEDULES: dict[str, PullSchedule] = {
    "hospital_gown": PullSchedule((1.0, 0.0, 0.0), 0.40, 2.0, 2.0),
    "tshirt_4": PullSchedule((0.0, -1.0, 0.0), 0.15, 1.5, 1.5),
    "tshirt_26": PullSchedule((0.0, -1.0, 0.0), 0.15, 1.5, 1.5),
    "tshirt_68": PullSchedule((0.0, -1.0, 0.0), 0.15, 1.5, 1.5),
    "tshirt_392": PullSchedule((0.0, -1.0, 0.0), 0.15, 1.5, 1.5),
}


@dataclass(frozen=True)
class BakeConfig:
    """Solver and material settings for the online drape bake.

    The defaults are the ones measured to reproduce the offline bake's drape:
    with a 1/100 shear ratio, bending 0.1 and a 6 kPa stretch modulus the free
    fabric sags 0.105 m against the reference's 0.112 m and the opening holds at
    9.92 cm against 9.91 cm. ``DressingConfig``'s episode cloth is stiffer on all
    three counts, which is recorded but not yet changed.
    """

    garment_dir: Path = field(default_factory=lambda: _DEFAULT_GARMENT_DIR)
    index_module: Path = field(default_factory=lambda: _DEFAULT_INDEX_MODULE)
    workspace: Path = field(default_factory=lambda: Path("output/uipc_manip/drape_bake"))
    dt: float = 1.0 / 60.0
    cloth_youngs: float = 6.0e3
    cloth_poisson: float = 0.49
    cloth_density: float = 3333.0
    cloth_thickness: float = 1.5e-4
    cloth_strain_rate: float = 100.0
    cloth_bending_stiffness: float = 0.1
    cloth_shear_ratio: float | None = 0.01
    """Shear Young's modulus as a fraction of the stretch modulus. libuipc's shear term is
    ``E/(2(1+nu))`` and carries no thickness factor, so sharing one modulus with stretch
    (``E*2r/(1-nu^2)``) makes shear roughly ``1/(2r)`` times stiffer and the sheet unshearable.
    The constitution's own guidance is about 1/100; ``None`` keeps the single-modulus form."""
    pin_strength: float = 1.0e4
    d_hat: float = 1.0e-3
    friction: float = 0.3
    contact_resistance: float = 1.0e7
    newton_tolerance: float = 0.1
    newton_translation_tolerance: float = 1.0
    linesearch_iterations: int = 8
    linear_system_tolerance: float = 1.0e-3
    fem_preconditioner: str = "mas"
    sanity_check: bool = True
    max_clean_rounds: int = 6
    min_bake_thickness: float = 1.0e-6
    separation_step_m: float = 1.0e-3
    """First displacement applied to vertices whose triangles cross at rest; it
    doubles each repair round."""
    """Floor for the automatic collision-radius reduction. Some raw meshes have two
    parts passing within tens of micrometres; the radius is a modelling choice for
    a garment-only bake, and IPC remains penetration-free at any radius."""
    """Rounds of dropping self-intersecting triangles from the canonical rest mesh.
    The raw meshes ship with a few tangled triangles in hem folds; the offline
    bake dropped them the same way, and they carry no cuff dynamics."""
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)
    """libuipc's own default points along -Y; the canonical frame here is Z-up."""

    def to_dict(self) -> dict:
        return {
            k: (str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v)
            for k, v in self.__dict__.items()
        }


def load_index_tables(path: Path | None = None) -> ModuleType:
    """Import the per-garment particle index tables as a module."""
    path = Path(path or _DEFAULT_INDEX_MODULE)
    if not path.exists():
        raise FileNotFoundError(
            f"Garment index tables {path} not found; set UIPC_MANIP_GARMENT_INDEX_MODULE to garment_idx_utils.py"
        )
    spec = importlib.util.spec_from_file_location("uipc_manip_garment_indices", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_euler(garment: str) -> tuple[float, float, float]:
    return (0.0, -90.0, 180.0) if "hospital_gown" in garment else (0.0, 0.0, -90.0)


def canonical_offset(garment: str) -> np.ndarray:
    return np.array([0.0, 0.0, 0.6]) if "hospital_gown" in garment else np.array([0.4, 0.4, 0.35])


def _euler_zyx(angles_deg: tuple[float, float, float]) -> np.ndarray:
    """Rotation for scipy's ``from_euler("zyx", angles, degrees=True)``, without scipy."""
    z, y, x = (np.deg2rad(a) for a in angles_deg)

    def rot(axis: int, a: float) -> np.ndarray:
        c, s = np.cos(a), np.sin(a)
        m = np.eye(3)
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        m[i, i] = m[j, j] = c
        m[i, j], m[j, i] = -s, s
        return m

    # Intrinsic z-y-x: R = Rz @ Ry @ Rx.
    return rot(2, z) @ rot(1, y) @ rot(0, x)


def canonical_transform(vertices: np.ndarray, garment: str, scale: float) -> np.ndarray:
    """Newton's ``_canonical_garment_vertices``: scale, rotate, ground, Y-up to Z-up."""
    v = np.asarray(vertices, dtype=np.float64) * float(scale)
    centre = v.mean(axis=0)
    v = (v - centre) @ _euler_zyx(canonical_euler(garment)).T + centre
    centre = v.mean(axis=0).copy()
    centre[1] = float(v[:, 1].min())
    v = v - centre + canonical_offset(garment)
    return np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1.0e-12 else _unit(fallback, np.array([1.0, 0.0, 0.0]))


def socket_frame(centre: np.ndarray, z_axis: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Right-handed socket-to-world frame: +Z the insertion axis, origin the opening centre."""
    z_axis = _unit(z_axis, np.array([0.0, 0.0, 1.0]))
    x_axis = None
    for point in np.asarray(ring, dtype=np.float64):
        candidate = point - centre
        candidate = candidate - np.dot(candidate, z_axis) * z_axis
        if np.linalg.norm(candidate) > 1.0e-8:
            x_axis = _unit(candidate, np.array([1.0, 0.0, 0.0]))
            break
    if x_axis is None:
        hint = np.array([0.0, 1.0, 0.0]) if abs(z_axis[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        x_axis = _unit(hint - np.dot(hint, z_axis) * z_axis, np.array([1.0, 0.0, 0.0]))
    y_axis = _unit(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0]))
    x_axis = _unit(np.cross(y_axis, z_axis), np.array([1.0, 0.0, 0.0]))
    T = np.eye(4, dtype=np.float64)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x_axis, y_axis, z_axis, centre
    return T


def cuff_semantics(vertices: np.ndarray, opening_idx: np.ndarray, alignment_idx: np.ndarray) -> dict:
    """Opening centre, insertion axis, socket frame and radii, measured off a drape."""
    v = np.asarray(vertices, dtype=np.float64)
    ring = v[np.asarray(opening_idx, dtype=np.int64)]
    centre = ring.mean(axis=0)
    centred = ring - centre
    normal = _unit(np.linalg.svd(centred)[2][2], np.array([0.0, 0.0, 1.0]))
    align = np.asarray(alignment_idx, dtype=np.int64).reshape(-1)
    # ``alignment_line_indices`` is [shoulder_end, hand_end]; the insertion axis
    # runs from the cuff into the sleeve.
    axis = _unit(v[align[0]] - v[align[1]], normal) if align.size == 2 else _unit(v.mean(axis=0) - centre, normal)
    radial = centred - np.outer(centred @ axis, axis)
    radii = np.linalg.norm(radial, axis=1)
    return {
        "opening_center": centre,
        "insertion_axis": axis,
        "socket_to_canonical": socket_frame(centre, axis, ring),
        "opening_radius_mean_m": float(radii.mean()),
        "opening_radius_max_m": float(radii.max()),
    }


_REPORT_KINDS = (
    ("intersected_mesh", "self-intersecting"),
    ("close_mesh", "closer than the summed collision radius"),
)


def _obj_vertices(path: Path) -> np.ndarray:
    """Vertices of an OBJ, tolerating a mesh that carries edges but no triangles."""
    points = [
        [float(x) for x in parts[1:4]]
        for parts in (line.split() for line in Path(path).read_text().splitlines())
        if parts and parts[0] == "v"
    ]
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def separate_reported_vertices(
    vertices: np.ndarray, faces: np.ndarray, reported_points: np.ndarray, step: float
) -> np.ndarray:
    """Push the flagged vertices apart along their own normals.

    Two layers of fabric crossing in a hem fold have opposing surface normals,
    so displacing each flagged vertex along its own normal separates them while
    leaving the topology, and therefore every semantic index, untouched.
    Dropping the faces instead leaves orphan vertices that libuipc's volume
    check then rejects without a report of its own.
    """
    from .dressing_assets import vertex_normals

    verts = np.asarray(vertices, dtype=np.float64).copy()
    flagged = _match_reported(verts, reported_points)
    normals = vertex_normals(verts, np.asarray(faces, dtype=np.int64))
    index = np.array(sorted(flagged), dtype=np.int64)
    verts[index] += normals[index] * float(step)
    return verts


def _match_reported(vertices: np.ndarray, reported_points: np.ndarray) -> set[int]:
    """Source vertex indices for the points the sanity check reported."""
    reported = np.asarray(reported_points, dtype=np.float64).reshape(-1, 3)
    lookup = {tuple(np.round(p, 6)): i for i, p in enumerate(np.asarray(vertices, dtype=np.float64))}
    flagged = {lookup[key] for key in (tuple(np.round(p, 6)) for p in reported) if key in lookup}
    if not flagged:
        raise RuntimeError("Could not map any reported point back to the source mesh")
    return flagged


def drop_reported_faces(vertices: np.ndarray, faces: np.ndarray, reported_points: np.ndarray) -> np.ndarray:
    """Remove every source face touching a point the sanity check reported.

    The checker hands back the offending primitives as a mesh whose vertices are
    copies of the rest positions, so a rounded-position lookup maps them to
    source vertices. Dropping the faces around them is what the offline bake did
    with the few tangled triangles the raw meshes ship with; they carry no cuff
    dynamics.
    """
    flagged = _match_reported(vertices, reported_points)
    rows = np.asarray(faces, dtype=np.int64)
    keep = ~np.isin(rows, list(flagged)).any(axis=1)
    if not keep.any() or keep.all():
        raise RuntimeError(f"Dropping the reported primitives would remove {'every' if not keep.any() else 'no'} face")
    return np.asarray(faces, dtype=np.int32)[keep]


def bake_key(garment: str, scale: float, cfg: BakeConfig) -> str:
    """Content hash over everything that changes the drape, for the on-disk cache."""
    payload = {"garment": garment, "scale": float(scale), "schedule": PULL_SCHEDULES[garment].__dict__, "cfg": cfg.to_dict()}
    payload["cfg"].pop("workspace", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def bake_drape(garment: str, *, scale: float | None = None, cfg: BakeConfig | None = None, reuse: bool = True) -> dict:
    """Pull the cuff open and settle the garment in libuipc; return the drape and its semantics.

    Runs a standalone libuipc world with the garment alone: no arm, no Genesis.
    The pinned set is the grasp patch plus the six opening-polygon vertices, as
    in the offline bake, which is what keeps a closed tshirt's opening from
    collapsing.
    """
    import uipc
    from uipc import Logger, builtin, view
    from uipc.constitution import (
        DiscreteShellBending,
        ElasticModuli2D,
        SoftPositionConstraint,
        StrainLimitingBaraffWitkinShell,
    )
    from uipc.core import Engine, Scene, World
    from uipc.geometry import label_surface, trimesh

    cfg = cfg or BakeConfig()
    tables = load_index_tables(cfg.index_module)
    if garment not in PULL_SCHEDULES:
        raise KeyError(f"No pull schedule for {garment!r}; known: {sorted(PULL_SCHEDULES)}")
    scale = float(tables.cloth_scales[garment] if scale is None else scale)
    mesh_path = Path(cfg.garment_dir) / RAW_MESH_FILENAME.get(garment, f"{garment}.obj")
    if not mesh_path.exists():
        raise FileNotFoundError(f"Raw garment mesh {mesh_path} not found; set UIPC_MANIP_RAW_GARMENT_DIR")

    workspace = Path(cfg.workspace)
    cached = workspace / f"{garment}__s{scale:.4f}__{bake_key(garment, scale, cfg)}.npz"
    grasp_idx = np.asarray(tables.grasping_particle_indices[garment], dtype=np.int64).reshape(-1)
    opening_idx = np.asarray(tables.shoulder_polygon_particle_indices[garment], dtype=np.int64).reshape(-1)
    picker_idx = np.asarray(tables.picker_picking_particle_indices.get(garment) or [], dtype=np.int64).reshape(-1)
    alignment_idx = np.asarray(tables.alignment_line_indices.get(garment) or [], dtype=np.int64).reshape(-1)
    vertices, faces = load_obj(mesh_path)
    if reuse and cached.exists():
        with np.load(cached) as data:
            drape = np.asarray(data["cloth"], dtype=np.float64)
        return _drape_payload(garment, scale, drape, faces, grasp_idx, opening_idx, picker_idx, alignment_idx, cached, 0.0)

    rest = canonical_transform(vertices, garment, scale)
    schedule = PULL_SCHEDULES[garment]
    pin_idx = np.array(sorted(set(grasp_idx.tolist()) | set(opening_idx.tolist())), dtype=np.int64)
    direction = _unit(np.asarray(schedule.direction, dtype=np.float64), np.array([0.0, -1.0, 0.0]))
    pull_steps = max(1, int(round(schedule.pull_seconds / cfg.dt)))
    settle_steps = max(1, int(round(schedule.settle_seconds / cfg.dt)))

    workspace.mkdir(parents=True, exist_ok=True)
    Logger.set_level(Logger.Level.Error)
    faces = np.asarray(faces, dtype=np.int32)
    dropped = 0
    thickness = float(cfg.cloth_thickness)
    for attempt in range(int(cfg.max_clean_rounds) + 1):
        world_dir = workspace / f"world_{garment}_{attempt}"
        try:
            built = _build_bake_world(world_dir, rest, faces, pin_idx, cfg, garment, thickness=thickness)
            world, slot, state = built["world"], built["slot"], built["state"]
            break
        except _RestIllegal as exc:
            if attempt == int(cfg.max_clean_rounds):
                raise RuntimeError(
                    f"{garment}: the canonical rest mesh is still illegal after {attempt} repair rounds ({exc})"
                ) from exc
            if exc.kind == "self-intersecting":
                # Separate the crossing layers instead of cutting them out; the
                # cut leaves orphan vertices the volume check then rejects.
                separation = cfg.separation_step_m * (2.0**attempt)
                rest = separate_reported_vertices(rest, faces, exc.points, separation)
                dropped += 1
            else:
                # Two parts of the garment merely pass close by. The collision
                # radius is a modelling choice, so shrink it rather than cut the
                # mesh; IPC stays penetration-free at any radius.
                if thickness <= cfg.min_bake_thickness:
                    raise RuntimeError(
                        f"{garment}: still too close at the minimum bake collision radius {thickness:g} m"
                    ) from exc
                thickness = max(cfg.min_bake_thickness, thickness / 4.0)

    t0 = time.time()
    step = (direction * schedule.distance_m) / pull_steps
    for k in range(pull_steps + settle_steps):
        if k < pull_steps:
            state["targets"] = state["targets"] + step[None, :]
        world.advance()
        world.retrieve()
        if not world.is_valid():
            raise RuntimeError(f"{garment}: the drape bake diverged at step {k}")
    drape = np.asarray(uipc.view(slot.geometry().positions()), dtype=np.float64).reshape(-1, 3).copy()
    seconds = time.time() - t0
    if not np.isfinite(drape).all():
        raise RuntimeError(f"{garment}: the drape bake produced non-finite positions")

    np.savez_compressed(
        cached,
        cloth=drape,
        faces=faces,
        grasp_idx=grasp_idx,
        opening_idx=opening_idx,
        picker_idx=picker_idx,
        alignment_idx=alignment_idx,
        config=json.dumps({"garment": garment, "scale": scale, "seconds": seconds, "repair_rounds": dropped, **cfg.to_dict()}),
    )
    return _drape_payload(garment, scale, drape, faces, grasp_idx, opening_idx, picker_idx, alignment_idx, cached, seconds)


class _RestIllegal(RuntimeError):
    """The canonical rest mesh is illegal, carrying the offending points.

    ``kind`` separates the two causes, which need opposite repairs: triangles
    that actually cross have to be dropped, while primitives that merely pass
    within the summed collision radius only need a thinner radius.
    """

    def __init__(self, points: np.ndarray, kind: str, summary: str) -> None:
        super().__init__(f"rest mesh is {kind}: {summary}")
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.kind = kind


def _build_bake_world(world_dir: Path, rest, faces, pin_idx, cfg: "BakeConfig", garment: str, *, thickness: float):
    """Build a one-garment libuipc world; raise :class:`_RestIntersects` if the rest state is illegal."""
    import uipc
    from uipc import Logger, builtin, view
    from uipc.constitution import (
        DiscreteShellBending,
        ElasticModuli2D,
        SoftPositionConstraint,
        StrainLimitingBaraffWitkinShell,
    )
    from uipc.core import Engine, Scene, World
    from uipc.geometry import label_surface, trimesh

    engine = Engine("cuda", str(world_dir))
    world = World(engine)
    config = Scene.default_config()
    config["dt"] = cfg.dt
    config["gravity"] = [[g] for g in cfg.gravity]
    config["contact"]["d_hat"] = cfg.d_hat
    config["sanity_check"]["enable"] = int(bool(cfg.sanity_check))  # the config is JSON: a Python bool disables the check
    config["newton"]["velocity_tol"] = cfg.newton_tolerance
    config["newton"]["transrate_tol"] = cfg.newton_translation_tolerance
    config["line_search"]["max_iter"] = cfg.linesearch_iterations
    config["linear_system"]["tol_rate"] = cfg.linear_system_tolerance
    config["linear_system"]["fem_preconditioner"] = cfg.fem_preconditioner
    scene = Scene(config)
    scene.contact_tabular().default_model(cfg.friction, cfg.contact_resistance)
    contact = scene.contact_tabular().default_element()

    mesh = trimesh(rest, np.asarray(faces, dtype=np.int32))
    label_surface(mesh)
    stretch = ElasticModuli2D.youngs_poisson(cfg.cloth_youngs, cfg.cloth_poisson)
    if cfg.cloth_shear_ratio is None:
        StrainLimitingBaraffWitkinShell().apply_to(
            mesh, stretch, cfg.cloth_density, thickness, cfg.cloth_strain_rate
        )
    else:
        shear = ElasticModuli2D.youngs_poisson(cfg.cloth_youngs * float(cfg.cloth_shear_ratio), cfg.cloth_poisson)
        StrainLimitingBaraffWitkinShell().apply_to(
            mesh, stretch, shear, cfg.cloth_density, thickness, cfg.cloth_strain_rate
        )
    DiscreteShellBending().apply_to(mesh, cfg.cloth_bending_stiffness)
    SoftPositionConstraint().apply_to(mesh, cfg.pin_strength)
    contact.apply_to(mesh)
    obj = scene.objects().create(f"drape_{garment}")

    state = {"targets": rest[pin_idx].copy()}

    def animate(info):
        geo = info.geo_slots()[0].geometry()
        flags = view(geo.vertices().find(builtin.is_constrained)).reshape(-1)
        flags[:] = 0
        flags[pin_idx] = 1
        view(geo.vertices().find(builtin.aim_position)).reshape(-1, 3)[pin_idx] = state["targets"]

    scene.animator().insert(obj, animate)
    slot = obj.geometries().create(mesh)[0]
    world.init(scene)
    if not world.is_valid():
        # Prefer the checker's own report objects; they are empty on some builds,
        # so fall back to the meshes it writes under this world's directory,
        # polling because the flush lands after ``init`` returns.
        for message in world.sanity_checker().errors().values():
            geometries = message.geometries()
            for key, kind in _REPORT_KINDS:
                if key in geometries:
                    points = np.asarray(uipc.view(geometries[key].positions()), dtype=np.float64).reshape(-1, 3)
                    raise _RestIllegal(points, kind, str(message.message()).splitlines()[0])
        for _ in range(50):
            for key, kind in _REPORT_KINDS:
                hits = sorted(Path(world_dir).glob(f"**/{key}.obj"))
                if hits:
                    raise _RestIllegal(_obj_vertices(hits[-1]), kind, f"see {hits[-1]}")
            time.sleep(0.1)
        raise RuntimeError(f"{garment}: the canonical rest mesh is not a valid libuipc scene")
    # Engine, world, scene and object must outlive this call or the backend expires.
    return {"engine": engine, "world": world, "scene": scene, "object": obj, "slot": slot, "state": state}


def _drape_payload(garment, scale, drape, faces, grasp_idx, opening_idx, picker_idx, alignment_idx, path, seconds) -> dict:
    semantics = cuff_semantics(drape, opening_idx, alignment_idx)
    return {
        "garment": garment,
        "scale": float(scale),
        "cloth": drape,
        "faces": np.asarray(faces, dtype=np.int32),
        "grasp_idx": grasp_idx,
        "opening_idx": opening_idx,
        "picker_idx": picker_idx,
        "alignment_idx": alignment_idx,
        "anchor": drape[grasp_idx].mean(axis=0),
        "path": Path(path),
        "bake_seconds": float(seconds),
        **semantics,
    }


def bake_in_subprocess(garment: str, *, scale: float | None = None, cfg: BakeConfig | None = None) -> dict:
    """Bake one garment in a fresh interpreter, then load the cached result.

    libuipc's sanity checker keeps state across worlds in one process, so a
    second garment's repair round sees the previous garment's checks and fails
    without a report. The offline tool ran one container per garment for the
    same reason. Baking is a one-off per garment, and the result is cached by a
    content hash of its inputs, so the process cost is paid once.
    """
    import subprocess
    import sys

    cfg = cfg or BakeConfig()
    tables = load_index_tables(cfg.index_module)
    scale = float(tables.cloth_scales[garment] if scale is None else scale)
    cached = Path(cfg.workspace) / f"{garment}__s{scale:.4f}__{bake_key(garment, scale, cfg)}.npz"
    if not cached.exists():
        # The child changes directory, so every path it is handed must be absolute.
        payload_cfg = cfg.to_dict()
        for key in ("garment_dir", "index_module", "workspace"):
            payload_cfg[key] = str(Path(payload_cfg[key]).resolve())
        payload = json.dumps({"garment": garment, "scale": scale, "cfg": payload_cfg})
        package_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(package_root), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        result = subprocess.run(
            [sys.executable, "-m", "uipc_manip.dressing_bake", payload],
            capture_output=True,
            text=True,
            cwd=str(package_root),
            env=env,
        )
        if result.returncode != 0 or not cached.exists():
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-4:]
            raise RuntimeError(f"{garment}: the drape bake subprocess failed\n" + "\n".join(tail))
    return bake_drape(garment, scale=scale, cfg=cfg, reuse=True)


def main(argv: list[str] | None = None) -> None:
    """Bake one garment described by a JSON payload; used by :func:`bake_in_subprocess`."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m uipc_manip.dressing_bake '{\"garment\": ..., \"scale\": ..., \"cfg\": {...}}'")
    request = json.loads(argv[0])
    fields = {f.name for f in BakeConfig.__dataclass_fields__.values()}
    raw = {k: v for k, v in (request.get("cfg") or {}).items() if k in fields}
    for key in ("garment_dir", "index_module", "workspace"):
        if key in raw:
            raw[key] = Path(raw[key])
    if "gravity" in raw:
        raw["gravity"] = tuple(raw["gravity"])
    baked = bake_drape(request["garment"], scale=request.get("scale"), cfg=BakeConfig(**raw), reuse=True)
    print(json.dumps({"garment": baked["garment"], "path": str(baked["path"]), "seconds": baked["bake_seconds"]}))


if __name__ == "__main__":
    main()
