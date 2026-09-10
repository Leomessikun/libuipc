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


def _obj_primitives(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Edges (``l``) and triangles (``f``) of a checker report OBJ, zero-based."""
    edges, tris = [], []
    for parts in (line.split() for line in Path(path).read_text().splitlines()):
        if parts and parts[0] == "l" and len(parts) >= 3:
            edges.append([int(x.split("/")[0]) - 1 for x in parts[1:3]])
        elif parts and parts[0] == "f" and len(parts) >= 4:
            tris.append([int(x.split("/")[0]) - 1 for x in parts[1:4]])
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2), np.asarray(tris, dtype=np.int64).reshape(-1, 3)


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


def mesh_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle mesh, each row sorted."""
    f = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def segments_through_triangles(
    points_a: np.ndarray, edges_a: np.ndarray, points_b: np.ndarray, faces_b: np.ndarray, *, same_mesh: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``edges_a`` whose segment passes through a triangle of ``faces_b``, with those rows.

    With ``same_mesh`` both index one vertex array and pairs sharing a vertex are skipped.
    """
    from scipy.spatial import cKDTree

    va, vb = np.asarray(points_a, dtype=np.float64), np.asarray(points_b, dtype=np.float64)
    e = np.asarray(edges_a, dtype=np.int64).reshape(-1, 2)
    f = np.asarray(faces_b, dtype=np.int64).reshape(-1, 3)
    if len(e) == 0 or len(f) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    tri = vb[f]
    centre = tri.mean(axis=1)
    radius = np.linalg.norm(tri - centre[:, None], axis=2).max(axis=1)
    a, b = va[e[:, 0]], va[e[:, 1]]
    half = 0.5 * np.linalg.norm(b - a, axis=1)
    near = cKDTree(centre).query_ball_point(0.5 * (a + b), float(radius.max() + half.max()))
    ei = np.repeat(np.arange(len(e)), [len(n) for n in near])
    ti = np.fromiter((t for n in near for t in n), dtype=np.int64, count=len(ei))
    if same_mesh:
        shared = (f[ti] == e[ei, :1]).any(axis=1) | (f[ti] == e[ei, 1:]).any(axis=1)
        ei, ti = ei[~shared], ti[~shared]
    # Moller-Trumbore on the segment a + t (b - a), t in [0, 1].
    p0 = vb[f[ti, 0]]
    e1, e2 = vb[f[ti, 1]] - p0, vb[f[ti, 2]] - p0
    d = va[e[ei, 1]] - va[e[ei, 0]]
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1.0e-20
    inv = 1.0 / np.where(ok, det, 1.0)
    s = va[e[ei, 0]] - p0
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, e1)
    w = np.einsum("ij,ij->i", d, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0.0) & (w >= 0.0) & (u + w <= 1.0) & (t >= 0.0) & (t <= 1.0)
    return ei[hit], ti[hit]


def crossing_pairs(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every edge that passes through a triangle it does not share a vertex with.

    Returns the crossing edges (vertex pairs) and the rows of ``faces`` they cross.
    This is the CPU counterpart of libuipc's self-intersection check, which reports
    the primitives but not which edge crosses which triangle.
    """
    e = mesh_edges(faces)
    ei, ti = segments_through_triangles(vertices, e, vertices, faces, same_mesh=True)
    return e[ei], ti


def untangle_crossings(
    vertices: np.ndarray, faces: np.ndarray, *, margin: float = 1.0e-3, max_rounds: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Move every vertex that pokes through a triangle back to its own side of it.

    For each crossing edge the endpoint nearer the triangle's plane is the one that
    went through; it moves along *the triangle's* normal to the side of the edge's
    other endpoint, ``margin`` clear of the plane. Pushing the flagged vertices along
    their own normals instead fails whenever the crossing layers' normals are not
    opposed, as in a folded hem, which is why that repair never cleared tshirt_68.
    Topology, and so every semantic index, is untouched. Returns the vertices and
    the indices that moved.
    """
    v = np.asarray(vertices, dtype=np.float64).copy()
    f = np.asarray(faces, dtype=np.int64)
    moved: set[int] = set()
    for _ in range(int(max_rounds)):
        edges, tris = crossing_pairs(v, f)
        if len(tris) == 0:
            return v, np.array(sorted(moved), dtype=np.int64)
        corner = v[f[tris]]
        normal = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        da = np.einsum("ij,ij->i", v[edges[:, 0]] - corner[:, 0], normal)
        db = np.einsum("ij,ij->i", v[edges[:, 1]] - corner[:, 0], normal)
        first = np.abs(da) <= np.abs(db)
        poker = np.where(first, edges[:, 0], edges[:, 1])
        depth, other = np.where(first, da, db), np.where(first, db, da)
        step = normal * (np.where(other >= 0.0, 1.0, -1.0) * float(margin) - depth)[:, None]
        # One move per vertex per round, the largest any of its crossings asks for.
        done: set[int] = set()
        for k in np.argsort(-np.linalg.norm(step, axis=1)):
            vid = int(poker[k])
            if vid not in done:
                done.add(vid)
                v[vid] += step[k]
                moved.add(vid)
    raise RuntimeError(f"{len(tris)} edge-triangle crossings remain after {max_rounds} untangling rounds")


def _closest_on_segments(p0, p1, q0, q1) -> tuple[np.ndarray, np.ndarray]:
    """Closest points between segments ``p0p1`` and ``q0q1`` (Ericson 5.1.9)."""
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    denom = a * e - b * b
    s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) if denom > 1.0e-30 else 0.0
    t = (b * s + f) / e if e > 1.0e-30 else 0.0
    if t < 0.0:
        t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0)) if a > 1.0e-30 else 0.0
    elif t > 1.0:
        t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0)) if a > 1.0e-30 else 0.0
    return p0 + s * d1, q0 + t * d2


def _closest_on_triangle(p, a, b, c) -> np.ndarray:
    """Closest point to ``p`` on triangle ``abc``: its plane projection if inside, else an edge."""
    n = np.cross(b - a, c - a)
    candidates = []
    if float(n @ n) > 1.0e-30:
        q = p - float((p - a) @ n) / float(n @ n) * n
        if all(float(np.cross(v1 - v0, q - v0) @ n) >= 0.0 for v0, v1 in ((a, b), (b, c), (c, a))):
            candidates.append(q)
    for v0, v1 in ((a, b), (b, c), (c, a)):
        seg = v1 - v0
        t = float(np.clip((p - v0) @ seg / max(float(seg @ seg), 1.0e-30), 0.0, 1.0))
        candidates.append(v0 + t * seg)
    return min(candidates, key=lambda x: float(np.linalg.norm(p - x)))


def separate_close_primitives(
    vertices: np.ndarray, points: np.ndarray, edges: np.ndarray, triangles: np.ndarray, gap: float
) -> np.ndarray | None:
    """Push each pair of primitives the distance check reported apart to ``gap``.

    The checker reports the offending primitives as ``points`` (copies of the rest
    positions) with ``edges`` and ``triangles`` indexing them; each pair of reported edges (and each reported point against each
    reported triangle) that sits closer than ``gap`` is opened along its closest-point
    direction, half the shortfall on each side. This keeps the collision radius the
    episode will use: shrinking the radius instead, the previous repair, let
    tshirt_392 bake at 5 um, where the pair's slack froze the fabric and the "drape"
    was the rest pose carried 0.15 m along the pull. Returns ``None`` when nothing in
    the report can be paired.
    """
    verts = np.asarray(vertices, dtype=np.float64).copy()
    lookup = {tuple(np.round(p, 6)): i for i, p in enumerate(verts)}
    source = np.array([lookup.get(tuple(np.round(p, 6)), -1) for p in np.asarray(points).reshape(-1, 3)], dtype=np.int64)
    edges = [tuple(int(i) for i in source[e]) for e in np.asarray(edges, dtype=np.int64).reshape(-1, 2) if (source[e] >= 0).all()]
    tris = [tuple(int(i) for i in source[t]) for t in np.asarray(triangles, dtype=np.int64).reshape(-1, 3) if (source[t] >= 0).all()]
    in_primitive = {i for prim in edges + tris for i in prim}
    lone = [int(i) for i in source if i >= 0 and int(i) not in in_primitive]
    moves: list[tuple[tuple[int, ...], np.ndarray]] = []

    def push(side_a, side_b, pa, pb, fallback):
        d = float(np.linalg.norm(pa - pb))
        if d < gap:
            direction = _unit(pa - pb, fallback) if d > 1.0e-12 else _unit(fallback, np.array([0.0, 0.0, 1.0]))
            moves.append((side_a, direction * (gap - d) / 2.0))
            moves.append((side_b, -direction * (gap - d) / 2.0))

    for k, e in enumerate(edges):
        for g in edges[k + 1 :]:
            if not set(e) & set(g):
                pa, pb = _closest_on_segments(verts[e[0]], verts[e[1]], verts[g[0]], verts[g[1]])
                push(e, g, pa, pb, np.cross(verts[e[1]] - verts[e[0]], verts[g[1]] - verts[g[0]]))
    for p in lone:
        for t in tris:
            if p not in t:
                q = _closest_on_triangle(verts[p], *verts[list(t)])
                push((p,), t, verts[p], q, np.cross(verts[t[1]] - verts[t[0]], verts[t[2]] - verts[t[0]]))
    if not moves:
        return None
    for index, delta in moves:
        verts[list(index)] += delta
    return verts


_BAKE_REVISION = 2
"""Bumped whenever the rest-mesh repair changes what a bake produces, so stale drapes re-bake.
Revision 2 separates crossing and close primitives instead of shrinking the collision radius."""


def bake_key(garment: str, scale: float, cfg: BakeConfig) -> str:
    """Content hash over everything that changes the drape, for the on-disk cache."""
    payload = {"garment": garment, "scale": float(scale), "schedule": PULL_SCHEDULES[garment].__dict__, "cfg": cfg.to_dict()}
    payload["revision"] = _BAKE_REVISION
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
    repaired: set[int] = set()
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
            dropped += 1
            if exc.kind == "self-intersecting":
                # Separate the crossing layers instead of cutting them out; the
                # cut leaves orphan vertices the volume check then rejects.
                untangled, moved = untangle_crossings(rest, faces, margin=cfg.separation_step_m)
                if moved.size:
                    rest = untangled
                    repaired.update(moved.tolist())
                else:
                    # The CPU test sees no crossing where the checker does: fall back
                    # to pushing the reported vertices along their own normals.
                    rest = separate_reported_vertices(rest, faces, exc.points, cfg.separation_step_m * (2.0**attempt))
            else:
                # Two parts of the garment merely pass close by. Open the reported
                # pairs at the episode's collision radius; shrink the radius only if
                # the report cannot be paired, since a sliver of slack freezes the bake.
                separated = separate_close_primitives(
                    rest, exc.points, exc.edges, exc.triangles, 2.0 * thickness + cfg.separation_step_m
                )
                if separated is not None:
                    repaired.update(np.flatnonzero(np.any(separated != rest, axis=1)).tolist())
                    rest = separated
                    continue
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
        config=json.dumps(
            {
                "garment": garment, "scale": scale, "seconds": seconds, "repair_rounds": dropped,
                "repaired_vertices": sorted(repaired), "bake_thickness": thickness, **cfg.to_dict(),
            }
        ),
    )
    return _drape_payload(garment, scale, drape, faces, grasp_idx, opening_idx, picker_idx, alignment_idx, cached, seconds)


class _RestIllegal(RuntimeError):
    """The canonical rest mesh is illegal, carrying the offending points.

    ``kind`` separates the two causes, which need opposite repairs: triangles
    that actually cross have to be dropped, while primitives that merely pass
    within the summed collision radius only need a thinner radius.
    """

    def __init__(self, points: np.ndarray, kind: str, summary: str, edges=None, triangles=None) -> None:
        super().__init__(f"rest mesh is {kind}: {summary}")
        self.points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self.kind = kind
        # Report-local indices into ``points``; empty when the report carried none.
        self.edges = np.asarray(edges if edges is not None else [], dtype=np.int64).reshape(-1, 2)
        self.triangles = np.asarray(triangles if triangles is not None else [], dtype=np.int64).reshape(-1, 3)


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
                    report = geometries[key]
                    points = np.asarray(uipc.view(report.positions()), dtype=np.float64).reshape(-1, 3)
                    edges = np.asarray(uipc.view(report.edges().topo()), dtype=np.int64).reshape(-1, 2)
                    tris = np.asarray(uipc.view(report.triangles().topo()), dtype=np.int64).reshape(-1, 3)
                    raise _RestIllegal(points, kind, str(message.message()).splitlines()[0], edges, tris)
        for _ in range(50):
            for key, kind in _REPORT_KINDS:
                hits = sorted(Path(world_dir).glob(f"**/{key}.obj"))
                if hits:
                    raise _RestIllegal(_obj_vertices(hits[-1]), kind, f"see {hits[-1]}", *_obj_primitives(hits[-1]))
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
