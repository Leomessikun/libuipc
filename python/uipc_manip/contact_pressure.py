"""Contact pressure on a limb, per square centimetre and per anatomical band.

Force alone is the wrong quantity to bound when dressing a person. A hundred newtons spread over a
forearm and a hundred newtons under a cuff edge are the same force and a different injury, and the
study behind ISO/TS 15066's table measured an index fingertip at 58 N median force but 185 N/cm^2
median peak pressure, so dividing a resultant by a nominal area under-reports by roughly six.
Published robot-dressing work reports a single scalar at the robot's wrist and therefore cannot
distinguish the two cases; a per-vertex contact force can.

**Patches, not vertices.** Pressure is force over area, and the arm mesh's vertex areas span a factor
of 150 around a median of 1.06 cm^2 (measured, 1409 vertices over 2036.7 cm^2), so a per-vertex
quotient would read a small vertex as a hot spot and would move with mesh resolution. Instead every
vertex anchors a patch of its neighbours within :data:`PATCH_RADIUS_M`, whose area is a square
centimetre on a flat sheet, and the patch's summed force is divided by its own summed area. Refining
the mesh adds vertices to both sums, so the quotient converges instead of drifting.

**Bands, not equal fractions.** Vertices are not spread evenly along the limb: over the fingertip to
shoulder axis the eighths hold 671, 119, 68, 60, 51, 95, 104 and 241 vertices, the hand alone
carrying half. The bands below are cut from the fingertip, elbow and shoulder landmarks, and the
elbow gets a band of its own because that is where a sleeve catches.
"""

from __future__ import annotations

import numpy as np

PATCH_RADIUS_M = 0.005642
"""Patch radius whose disc area is one square centimetre."""

ELBOW_BAND = 0.25
"""Fraction of each segment nearest the elbow that is counted as the elbow band."""

BANDS = ("hand", "forearm", "elbow", "upperarm")


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric area of each vertex: a third of every triangle it belongs to [m^2]."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    corners = vertices[faces]
    twice = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = 0.5 * np.linalg.norm(twice, axis=1)
    out = np.zeros(len(vertices), dtype=np.float64)
    for corner in range(3):
        np.add.at(out, faces[:, corner], area / 3.0)
    return out


def band_of(vertices: np.ndarray, finger, elbow, shoulder) -> np.ndarray:
    """Index into :data:`BANDS` for each vertex, from the two arm segments.

    A vertex belongs to whichever segment it projects onto more closely; within a segment its
    position along it decides the band. Only the fingertip, elbow and shoulder are guaranteed to
    exist as landmarks, so the wrist is not used.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    finger = np.asarray(finger, dtype=np.float64)
    elbow = np.asarray(elbow, dtype=np.float64)
    shoulder = np.asarray(shoulder, dtype=np.float64)

    def fraction(start, end):
        axis = end - start
        length = float(axis @ axis)
        if length <= 0.0:
            return np.zeros(len(vertices)), np.full(len(vertices), np.inf)
        t = np.clip((vertices - start) @ axis / length, 0.0, 1.0)
        closest = start + t[:, None] * axis
        return t, np.linalg.norm(vertices - closest, axis=1)

    lower_t, lower_d = fraction(finger, elbow)     # fingertip to elbow
    upper_t, upper_d = fraction(elbow, shoulder)   # elbow to shoulder
    on_lower = lower_d <= upper_d
    band = np.empty(len(vertices), dtype=np.int64)
    # The quarter of each segment nearest the elbow is the elbow band; on the forearm the far
    # quarter from the elbow is the hand.
    band[on_lower] = np.where(lower_t[on_lower] >= 1.0 - ELBOW_BAND, BANDS.index("elbow"),
                              np.where(lower_t[on_lower] <= ELBOW_BAND, BANDS.index("hand"),
                                       BANDS.index("forearm")))
    band[~on_lower] = np.where(upper_t[~on_lower] <= ELBOW_BAND, BANDS.index("elbow"),
                               BANDS.index("upperarm"))
    return band


def patch_neighbours(vertices: np.ndarray, radius: float = PATCH_RADIUS_M) -> list[np.ndarray]:
    """For each vertex, the vertices within ``radius`` of it, itself included.

    Computed once for a limb that does not move, so a decision only pays the summation.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        return [np.asarray(idx, dtype=np.int64) for idx in cKDTree(vertices).query_ball_point(vertices, radius)]
    except ImportError:
        out = []
        for point in vertices:
            out.append(np.flatnonzero(np.linalg.norm(vertices - point, axis=1) <= radius))
        return out


class PressureMap:
    """Per-band force and per-square-centimetre peak pressure on one limb.

    Built once per limb; :meth:`summarise` is called with each decision's per-vertex force.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, finger, elbow, shoulder,
                 *, radius: float = PATCH_RADIUS_M) -> None:
        self.areas = vertex_areas(vertices, faces)
        self.bands = band_of(vertices, finger, elbow, shoulder)
        neighbours = patch_neighbours(vertices, radius)
        # Flattened neighbour lists, so a decision is one bincount rather than a Python loop.
        self._owner = np.repeat(np.arange(len(neighbours), dtype=np.int64), [len(n) for n in neighbours])
        self._member = np.concatenate(neighbours) if neighbours else np.zeros(0, dtype=np.int64)
        self.patch_area = np.bincount(self._owner, weights=self.areas[self._member], minlength=len(self.areas))

    def patch_pressure(self, force: np.ndarray) -> np.ndarray:
        """Pressure at each vertex's patch [N/m^2], from per-vertex force vectors."""
        magnitude = np.linalg.norm(np.atleast_2d(force), axis=1)
        patch_force = np.bincount(self._owner, weights=magnitude[self._member], minlength=len(self.areas))
        return np.divide(patch_force, self.patch_area, out=np.zeros_like(patch_force),
                         where=self.patch_area > 0.0)

    def summarise(self, force: np.ndarray, *, prefix: str = "") -> dict:
        """Per-band resultant, summed magnitude and peak patch pressure, plus the limb's peak."""
        force = np.atleast_2d(np.asarray(force, dtype=np.float64))
        magnitude = np.linalg.norm(force, axis=1)
        pressure = self.patch_pressure(force)
        out: dict[str, float] = {}
        for index, name in enumerate(BANDS):
            where = self.bands == index
            # The resultant and the summed magnitudes differ by what matters for a sleeve gripping a
            # limb: opposing local forces cancel in the vector sum but not on the skin.
            out[f"{prefix}{name}_resultant_n"] = float(np.linalg.norm(force[where].sum(axis=0))) if where.any() else 0.0
            out[f"{prefix}{name}_summed_n"] = float(magnitude[where].sum())
            out[f"{prefix}{name}_peak_kpa"] = float(pressure[where].max(initial=0.0)) / 1000.0
        out[f"{prefix}peak_kpa"] = float(pressure.max(initial=0.0)) / 1000.0
        out[f"{prefix}peak_band"] = BANDS[int(self.bands[int(np.argmax(pressure))])] if len(pressure) else ""
        return out
