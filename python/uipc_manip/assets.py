"""Deformable asset builders that produce libuipc geometries.

Every builder returns a :class:`Deformable` whose ``mesh`` already carries its
constitutions, so the environment only has to attach contact and constraint
elements. Material values match the validated Genesis 1.1.2 IPC examples
(``ipc_robot_deformables.py``) so the assets stay within a regime that the
pinned ``pyuipc`` solver is known to handle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"


@dataclass
class Deformable:
    """A deformable body ready to be inserted into an IPC scene."""

    kind: str
    mesh: object
    rest: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    radius: float
    provenance: dict = field(default_factory=dict)

    @property
    def vertex_count(self) -> int:
        return int(self.rest.shape[0])


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and fan-triangulated faces from a Wavefront OBJ file."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open() as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == "f":
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                for k in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[k], indices[k + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def build_cloth(
    *,
    size: float = 0.25,
    center: tuple[float, float, float] = (0.55, 0.0, 0.01),
    thickness: float = 0.002,
    youngs: float = 5e4,
    poisson: float = 0.35,
    mass_density: float = 200.0,
    bending_youngs: float = 5e4,
    source: Path = ASSET_DIR / "grid20x20.obj",
) -> Deformable:
    """Square sheet lying flat in the XY plane, centred at ``center``.

    The source grid is a unit square in XY. It is scaled to ``size`` metres and
    lifted to ``center[2]``, which must leave the sheet above the table by more
    than its collision radius plus the contact ``d_hat`` so IPC starts
    intersection free.
    """
    from uipc.constitution import DiscreteShellBending, ElasticModuli2D, NeoHookeanShell
    from uipc.geometry import label_surface
    from uipc.geometry import trimesh as ipc_trimesh

    vertices, faces = load_obj(source)
    vertices = vertices - vertices.mean(axis=0)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    axes = np.argsort(extent)[::-1][:2]
    flat = np.zeros_like(vertices)
    flat[:, 0] = vertices[:, axes[0]]
    flat[:, 1] = vertices[:, axes[1]]
    scale = float(size) / float(extent[axes].max())
    rest = flat * scale + np.asarray(center, dtype=np.float64)
    mesh = ipc_trimesh(rest, faces)
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(youngs, poisson), mass_density, thickness)
    DiscreteShellBending().apply_to(mesh, bending_youngs, poisson)
    provenance = {
        "source": str(source.name),
        "size_m": float(size),
        "collision_radius_m": float(thickness),
        "material": "NeoHookeanShell + DiscreteShellBending",
        "youngs_pa": float(youngs),
        "poisson": float(poisson),
        "mass_density": float(mass_density),
    }
    return Deformable(
        kind="cloth",
        mesh=mesh,
        rest=rest,
        faces=faces,
        edges=np.empty((0, 2), dtype=np.int32),
        radius=float(thickness),
        provenance=provenance,
    )


def build_cable(
    *,
    length: float = 0.30,
    count: int = 26,
    radius: float = 0.005,
    start: tuple[float, float, float] = (0.40, 0.0, 0.01),
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    stretch_kappa: float = 2e6,
    mass_density: float = 1000.0,
    bending_kappa: float = 1e5,
) -> Deformable:
    """Straight elastic rod starting at ``start`` and pointing along ``direction``."""
    from uipc.constitution import HookeanSpring, KirchhoffRodBending
    from uipc.geometry import label_surface, linemesh

    axis = np.asarray(direction, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    t = np.linspace(0.0, float(length), int(count))
    rest = np.asarray(start, dtype=np.float64)[None, :] + t[:, None] * axis[None, :]
    edges = np.column_stack((np.arange(count - 1), np.arange(1, count))).astype(np.int32)
    mesh = linemesh(rest, edges)
    label_surface(mesh)
    HookeanSpring().apply_to(mesh, stretch_kappa, mass_density, radius)
    KirchhoffRodBending().apply_to(mesh, bending_kappa)
    provenance = {
        "source": "Genesis cable.xml dimensions (count, radius); straight IPC rod",
        "length_m": float(length),
        "count": int(count),
        "radius_m": float(radius),
        "material": "HookeanSpring + KirchhoffRodBending",
        "stretch_kappa": float(stretch_kappa),
        "bending_kappa": float(bending_kappa),
    }
    return Deformable(
        kind="cable",
        mesh=mesh,
        rest=rest,
        faces=np.empty((0, 3), dtype=np.int32),
        edges=edges,
        radius=float(radius),
        provenance=provenance,
    )


def build_garment(
    path: Path,
    *,
    scale: float = 1.0,
    center: tuple[float, float, float] = (0.55, 0.0, 0.05),
    thickness: float = 0.002,
    youngs: float = 5e4,
    poisson: float = 0.35,
    mass_density: float = 200.0,
    bending_youngs: float = 5e4,
) -> Deformable:
    """Arbitrary triangle-mesh garment (for example the Newton t-shirt OBJ files).

    The mesh is centred at ``center`` after scaling. Garments are heavier than
    the square sheet, so expect slower IPC steps; they are provided for
    experiments and are not part of the validated task set.
    """
    from uipc.constitution import DiscreteShellBending, ElasticModuli2D, NeoHookeanShell
    from uipc.geometry import label_surface
    from uipc.geometry import trimesh as ipc_trimesh

    vertices, faces = load_obj(Path(path))
    rest = (vertices - vertices.mean(axis=0)) * float(scale) + np.asarray(center, dtype=np.float64)
    mesh = ipc_trimesh(rest, faces)
    label_surface(mesh)
    NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(youngs, poisson), mass_density, thickness)
    DiscreteShellBending().apply_to(mesh, bending_youngs, poisson)
    return Deformable(
        kind="cloth",
        mesh=mesh,
        rest=rest,
        faces=faces,
        edges=np.empty((0, 2), dtype=np.int32),
        radius=float(thickness),
        provenance={"source": str(path), "scale": float(scale), "material": "NeoHookeanShell + DiscreteShellBending"},
    )


def asset_sources() -> dict:
    return json.loads((ASSET_DIR / "sources.json").read_text())
