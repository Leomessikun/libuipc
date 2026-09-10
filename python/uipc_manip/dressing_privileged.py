"""Low-dimensional simulator state of a dressing slot, read only by an asymmetric critic.

The actor sees the segmented point cloud and nothing else. The critic is
discarded after training, so it may read what the simulator knows exactly, as
in the asymmetric actor-critic of Pinto et al. (2018): where the opening ring
sits relative to the arm, how the tool holds the cuff, how far the arm has
entered. Positions are expressed in a frame fixed to the arm, so two bodies that
present the same relative geometry give the same state wherever they stand.
The episode step is left out on purpose: a time limit is bootstrapped as a
non-terminal transition, which assumes a value that does not depend on time.
"""

from __future__ import annotations

import numpy as np

from .dressing_reward import DressingProgress, opening_threaded

PRIVILEGED_GARMENTS = ("hospital_gown", "tshirt_4", "tshirt_26", "tshirt_68", "tshirt_392")
"""Every garment the live bake knows, in a fixed order for the one-hot block."""

PRIVILEGED_LAYOUT = (
    ("arm", 3),
    ("tool", 3),
    ("tool_clearance", 1),
    ("tool_rotation", 6),
    ("opening_center", 3),
    ("opening_normal", 3),
    ("opening_radius", 2),
    ("garment_center", 3),
    ("progress", 5),
    ("tracking_error", 1),
    ("garment", len(PRIVILEGED_GARMENTS)),
)
"""Blocks of the state, in order. Lengths are metres in the arm frame of
:func:`arm_frame`, except the tracking error, in centimetres so that a slipping
grip reads of order one.

* ``arm``: forearm length, then the shoulder along and across the forearm.
* ``tool``: the tool point.
* ``tool_clearance``: the tool's distance to the nearest arm point.
* ``tool_rotation``: first two columns of the grip's rotation since the reset.
* ``opening_center``, ``opening_normal``: the opening ring's centroid and unit
  normal, the normal oriented by the ring's vertex order.
* ``opening_radius``: the ring's mean radius, and its largest minus smallest.
* ``garment_center``: the centroid of every garment vertex.
* ``progress``: forearm and upper-arm ratios, on the forearm, on the upper arm,
  and the winding test of :func:`~uipc_manip.dressing_reward.opening_threaded`.
* ``tracking_error``: the held vertices' largest distance to their targets.
* ``garment``: one-hot over :data:`PRIVILEGED_GARMENTS`, zero for any other.
"""

PRIVILEGED_DIM = sum(n for _, n in PRIVILEGED_LAYOUT)


def arm_frame(finger: np.ndarray, elbow: np.ndarray, shoulder: np.ndarray) -> np.ndarray:
    """Rows of an orthonormal frame on the arm: along the forearm to the elbow, toward the shoulder, their normal."""
    finger = np.asarray(finger, dtype=np.float64)
    e1 = np.asarray(elbow, dtype=np.float64) - finger
    e1 = e1 / max(float(np.linalg.norm(e1)), 1e-9)
    s = np.asarray(shoulder, dtype=np.float64) - finger
    e2 = s - (s @ e1) * e1
    if np.linalg.norm(e2) < 1e-9:
        # A straight arm leaves the bend plane undefined; any perpendicular will do.
        e2 = np.cross(e1, [0.0, 0.0, 1.0]) if abs(e1[2]) < 0.9 else np.cross(e1, [1.0, 0.0, 0.0])
    e2 = e2 / np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)])


def grip_rotation(initial_offsets: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rotation taking the held patch's reset offsets onto its current ones (Kabsch)."""
    h = np.asarray(initial_offsets, dtype=np.float64).T @ np.asarray(offsets, dtype=np.float64)
    u, _, vt = np.linalg.svd(h)
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0.0 else -1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def privileged_state(
    cloth: np.ndarray,
    *,
    opening_idx: np.ndarray,
    finger: np.ndarray,
    elbow: np.ndarray,
    shoulder: np.ndarray,
    arm_points: np.ndarray,
    tool: np.ndarray,
    offsets: np.ndarray,
    initial_offsets: np.ndarray,
    progress: DressingProgress,
    tracking_error: float,
    garment: str,
) -> np.ndarray:
    """One slot's state, laid out as :data:`PRIVILEGED_LAYOUT`."""
    finger = np.asarray(finger, dtype=np.float64)
    tool = np.asarray(tool, dtype=np.float64)
    frame = arm_frame(finger, elbow, shoulder)

    def local(p: np.ndarray) -> np.ndarray:
        return frame @ (np.asarray(p, dtype=np.float64) - finger)

    shoulder_local = local(shoulder)
    ring = np.asarray(cloth[opening_idx], dtype=np.float64)
    center = ring.mean(axis=0)
    rel = ring - center
    # Newell's normal: the ring's area vector, oriented by its vertex order.
    normal = np.cross(rel, np.roll(rel, -1, axis=0)).sum(axis=0)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    radii = np.linalg.norm(rel, axis=1)
    rotation = frame @ grip_rotation(initial_offsets, offsets) @ frame.T
    threaded, _ = opening_threaded(cloth, opening_idx, finger, np.asarray(shoulder, dtype=np.float64))
    one_hot = np.zeros(len(PRIVILEGED_GARMENTS))
    if garment in PRIVILEGED_GARMENTS:
        one_hot[PRIVILEGED_GARMENTS.index(garment)] = 1.0
    parts = [
        [float(np.linalg.norm(np.asarray(elbow, dtype=np.float64) - finger)), shoulder_local[0], shoulder_local[1]],
        local(tool),
        [float(np.min(np.linalg.norm(np.asarray(arm_points, dtype=np.float64) - tool[None, :], axis=1)))],
        rotation[:, :2].T.reshape(-1),
        local(center),
        frame @ normal,
        [float(radii.mean()), float(radii.max() - radii.min())],
        local(np.asarray(cloth, dtype=np.float64).mean(axis=0)),
        [progress.forearm_ratio, progress.upperarm_ratio, float(progress.on_forearm), float(progress.on_upperarm), float(threaded)],
        [100.0 * float(tracking_error)],
        one_hot,
    ]
    return np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts]).astype(np.float32)
