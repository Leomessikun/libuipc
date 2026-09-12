"""Contact forces in newtons, read out of a running libuipc scene.

``uipc.core.ContactSystemFeature`` exports the contact gradient of the incremental
potential, so a force is minus that gradient divided by the time step squared. That
conversion was measured rather than assumed: ten particles of known mass resting on a
ground half-plane, where the total normal force must equal their weight, give a ratio of
1.000000 at both dt 1/60 and dt 1/120
(``agent_docs/performance/2026-09-12-contact-force-calibration.md``).

The export names ten primitive types, five geometric cases each split into a normal term
(``+N``) and a friction term (``+F``):

    EE+N EE+F   edge-edge
    PE+N PE+F   point-edge
    PP+N PP+F   point-point
    PT+N PT+F   point-triangle
    PH+N PH+F   point-halfplane

Each call fills a geometry with one entry per contact doublet: ``i``, a global vertex
index, and ``grad``, a 3-vector. A vertex appears once per contact it takes part in, so
the entries of a vertex must be summed. Normal and friction are therefore separable per
vertex, which a single wrist force reading is not.

Three limits of the export, each verified in the source rather than assumed:

* **An unavailable exporter is not zero force.** Inspect the registered channels with
  :func:`contact_export_status`. ``vertex_forces`` rejects a missing feature or absent
  normal channels; registered channels containing no contacts legitimately return zero.
* **The gradient and the energy describe different configurations.** The gradient is
  assembled at the top of a Newton iteration and the energies are rewritten during line
  search, so a force read after ``advance`` belongs to the last iterate rather than to the
  frame's final state. At equilibrium the difference vanishes, which is why the calibration
  on a resting stack is exact; during a violent contact it need not.
* **The overloads taking a constitution are dead.** They look an exporter up by a
  ``"#<uid>"`` name that is never registered, and warn and return nothing.
"""

from __future__ import annotations

import operator

import numpy as np

NORMAL_SUFFIX = "+N"
FRICTION_SUFFIX = "+F"


def geometry_vertex_block(geometry) -> tuple[int, int]:
    """The ``(first_vertex, vertex_count)`` of a geometry in the solver's global index space.

    The backend stamps ``builtin.global_vertex_offset`` on each geometry's meta, so a body's
    own rows can be selected without assuming the order in which the scene was built.
    Only single-instance geometries are supported: rigid instances expand into multiple
    global vertex blocks, so silently returning the first block would miss their forces.
    """
    from uipc import builtin, view

    offset = geometry.meta().find(builtin.global_vertex_offset)
    if offset is None:
        raise ValueError("geometry carries no global_vertex_offset; read it after world.init")
    if geometry.instances().size() != 1:
        raise ValueError("contact force attribution requires a single-instance geometry")
    first = int(np.asarray(view(offset)).reshape(-1)[0])
    if first < 0:
        raise ValueError("geometry has an invalid global_vertex_offset")
    return first, int(geometry.vertices().size())


def find_contact_feature(world):
    """The scene's :class:`ContactSystemFeature`, or None when the backend has none."""
    from uipc.core import ContactSystemFeature

    return world.features().find(ContactSystemFeature)


def contact_export_status(feature, *, frame_stats: dict | None = None) -> dict:
    """Describe channel availability and the limitations of the assembled readout.

    Registered friction channels with zero gradients differ from an unavailable friction
    exporter. Iteration counts diagnose early stopping but cannot certify final-state force
    accuracy: the exporter returns cached assembly buffers, not a fresh evaluation.
    """
    channels = tuple(feature.contact_primitive_types()) if feature is not None else ()
    normal = tuple(name for name in channels if name.endswith(NORMAL_SUFFIX))
    friction = tuple(name for name in channels if name.endswith(FRICTION_SUFFIX))
    return {
        "normal_channels": normal,
        "friction_channels": friction,
        "normal_available": bool(normal),
        "friction_available": bool(friction),
        "configuration": "last_assembled_iterate",
        "frame": (frame_stats or {}).get("frame"),
        "newton_iterations": (frame_stats or {}).get("newton_iterations"),
        "converged": (frame_stats or {}).get("converged"),
    }


def vertex_forces(feature, dt: float, vertex_count: int, *, first_vertex: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Normal and friction force in newtons on each vertex of one contiguous index block.

    ``first_vertex`` and ``vertex_count`` select the block, so a single body's force can be
    read out of a scene holding several: the remaining contacts, including a cloth's own
    self-contact, are skipped rather than summed into the total. Returns two
    ``(vertex_count, 3)`` arrays, normal and friction.
    """
    from uipc import view
    from uipc.geometry import Geometry

    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    vertex_count, first_vertex = operator.index(vertex_count), operator.index(first_vertex)
    if vertex_count < 0:
        raise ValueError("vertex_count must not be negative")
    if first_vertex < 0:
        raise ValueError("first_vertex must not be negative")
    status = contact_export_status(feature)
    if not status["normal_available"]:
        raise RuntimeError("No normal contact exporter is available; this is not a zero-force measurement")
    scale = 1.0 / (dt ** 2)
    normal = np.zeros((vertex_count, 3), dtype=np.float64)
    friction = np.zeros((vertex_count, 3), dtype=np.float64)
    stop = first_vertex + vertex_count
    for prim in feature.contact_primitive_types():
        if not prim.endswith((NORMAL_SUFFIX, FRICTION_SUFFIX)):
            raise ValueError(f"Unrecognized contact force channel {prim!r}")
        geometry = Geometry()
        feature.contact_gradient(prim, geometry)
        count = geometry.instances().size()
        if count == 0:
            continue
        index = np.asarray(view(geometry.instances().find("i")), dtype=np.int64).reshape(count)
        gradient = np.asarray(view(geometry.instances().find("grad")), dtype=np.float64).reshape(count, 3)
        if not np.isfinite(gradient).all() or np.any(index < 0):
            raise ValueError(f"Invalid contact gradient or index in channel {prim}")
        keep = (index >= first_vertex) & (index < stop)
        if not keep.any():
            continue
        target = friction if prim.endswith(FRICTION_SUFFIX) else normal
        # A force is the negated gradient of the potential, and the exported gradient
        # belongs to the incremental potential, whose contact term carries a dt^2.
        np.add.at(target, index[keep] - first_vertex, -gradient[keep] * scale)
    return normal, friction


def force_summary(normal: np.ndarray, friction: np.ndarray) -> dict:
    """Summarize assembled nodal forces, not pressure or a calibrated skin traction field.

    The net and the summed magnitudes differ by exactly what matters for a garment gripping
    a limb: a sleeve squeezing an arm carries large opposing local forces whose vector sum
    largely cancels. Nodal peaks and summed magnitudes depend on mesh discretization;
    neither is a clinical safety threshold or a substitute for a surface-area measure.
    """
    normal, friction = np.asarray(normal), np.asarray(friction)
    if normal.ndim != 2 or normal.shape[1] != 3 or friction.shape != normal.shape:
        raise ValueError("normal and friction must have matching (vertex_count, 3) shapes")
    if not np.isfinite(normal).all() or not np.isfinite(friction).all():
        raise ValueError("force fields must be finite")
    normal_magnitude = np.linalg.norm(np.atleast_2d(normal), axis=1)
    friction_magnitude = np.linalg.norm(np.atleast_2d(friction), axis=1)
    return {
        "vertices_in_contact": int((normal_magnitude > 0.0).sum() + (friction_magnitude > 0.0).sum() - ((normal_magnitude > 0.0) & (friction_magnitude > 0.0)).sum()),
        "net_normal_n": float(np.linalg.norm(np.atleast_2d(normal).sum(axis=0))),
        "net_friction_n": float(np.linalg.norm(np.atleast_2d(friction).sum(axis=0))),
        "summed_normal_n": float(normal_magnitude.sum()),
        "summed_friction_n": float(friction_magnitude.sum()),
        "peak_vertex_normal_n": float(normal_magnitude.max(initial=0.0)),
        "peak_vertex_friction_n": float(friction_magnitude.max(initial=0.0)),
    }
