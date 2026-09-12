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
"""

from __future__ import annotations

import numpy as np

NORMAL_SUFFIX = "+N"
FRICTION_SUFFIX = "+F"


def find_contact_feature(world):
    """The scene's :class:`ContactSystemFeature`, or None when the backend has none."""
    from uipc.core import ContactSystemFeature

    return world.features().find(ContactSystemFeature)


def vertex_forces(feature, dt: float, vertex_count: int, *, first_vertex: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Normal and friction force in newtons on each vertex of one contiguous index block.

    ``first_vertex`` and ``vertex_count`` select the block, so a single body's force can be
    read out of a scene holding several: the remaining contacts, including a cloth's own
    self-contact, are skipped rather than summed into the total. Returns two
    ``(vertex_count, 3)`` arrays, normal and friction.
    """
    from uipc import view
    from uipc.geometry import Geometry

    if vertex_count < 0:
        raise ValueError("vertex_count must not be negative")
    scale = 1.0 / (float(dt) ** 2)
    normal = np.zeros((vertex_count, 3), dtype=np.float64)
    friction = np.zeros((vertex_count, 3), dtype=np.float64)
    stop = first_vertex + vertex_count
    for prim in feature.contact_primitive_types():
        geometry = Geometry()
        feature.contact_gradient(prim, geometry)
        count = geometry.instances().size()
        if count == 0:
            continue
        index = np.asarray(view(geometry.instances().find("i")), dtype=np.int64).reshape(count)
        gradient = np.asarray(view(geometry.instances().find("grad")), dtype=np.float64).reshape(count, 3)
        keep = (index >= first_vertex) & (index < stop)
        if not keep.any():
            continue
        target = friction if prim.endswith(FRICTION_SUFFIX) else normal
        # A force is the negated gradient of the potential, and the exported gradient
        # belongs to the incremental potential, whose contact term carries a dt^2.
        np.add.at(target, index[keep] - first_vertex, -gradient[keep] * scale)
    return normal, friction


def force_summary(normal: np.ndarray, friction: np.ndarray) -> dict:
    """Scalar summary of a force field: the net vector's magnitude, the summed magnitudes and the peak.

    The net and the summed magnitudes differ by exactly what matters for a garment gripping
    a limb: a sleeve squeezing an arm carries large opposing local forces whose vector sum
    largely cancels, so the net understates what the skin feels.
    """
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
