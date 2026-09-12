"""Per-square-centimetre contact pressure on a limb (CPU)."""

import numpy as np
import pytest

from uipc_manip.contact_pressure import (
    BANDS,
    PATCH_RADIUS_M,
    PressureMap,
    band_of,
    patch_neighbours,
    vertex_areas,
)


def _grid(n: int, side: float = 0.1):
    """A flat n x n sheet of side metres, as vertices and triangles."""
    xs = np.linspace(0.0, side, n)
    vx, vy = np.meshgrid(xs, xs, indexing="ij")
    vertices = np.stack([vx.ravel(), vy.ravel(), np.zeros(n * n)], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, b, c], [b, d, c]]
    return vertices, np.asarray(faces, dtype=np.int64)


def test_vertex_areas_sum_to_the_surface_area():
    vertices, faces = _grid(9, side=0.1)
    assert vertex_areas(vertices, faces).sum() == pytest.approx(0.01)  # 0.1 m square


def test_the_patch_radius_covers_one_square_centimetre():
    assert np.pi * PATCH_RADIUS_M ** 2 == pytest.approx(1.0e-4, rel=1e-3)


def test_pressure_is_force_over_area_for_a_uniform_load():
    # One newton spread evenly over a 0.1 m square is 100 Pa everywhere.
    vertices, faces = _grid(21, side=0.1)
    area = vertex_areas(vertices, faces)
    force = np.zeros((len(vertices), 3))
    force[:, 2] = area / area.sum()  # one newton in total, in proportion to area
    pressure = PressureMap(vertices, faces, [0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]).patch_pressure(force)
    interior = (vertices[:, 0] > 0.02) & (vertices[:, 0] < 0.08) & (vertices[:, 1] > 0.02) & (vertices[:, 1] < 0.08)
    assert np.allclose(pressure[interior], 100.0, rtol=0.05)


def test_the_patch_makes_pressure_independent_of_mesh_resolution():
    # The reason for patches rather than vertices: the same physical load on a finer mesh must not
    # read as a higher pressure. A per-vertex quotient would, since each vertex takes a smaller area.
    readings = []
    for n in (15, 29):
        vertices, faces = _grid(n, side=0.1)
        area = vertex_areas(vertices, faces)
        force = np.zeros((len(vertices), 3))
        force[:, 2] = 100.0 * area  # a uniform 100 Pa, however the sheet is discretised
        readings.append(PressureMap(vertices, faces, [0, 0, 0], [0.05, 0, 0], [0.1, 0, 0]).patch_pressure(force).max())
    assert readings[0] == pytest.approx(readings[1], rel=0.1)


def test_a_concentrated_load_reads_far_above_a_spread_one():
    # The case the field's single wrist scalar cannot tell apart: the same newton under a cuff edge
    # against the same newton over a limb.
    vertices, faces = _grid(21, side=0.1)
    area = vertex_areas(vertices, faces)
    field = PressureMap(vertices, faces, [0, 0, 0], [0.05, 0, 0], [0.1, 0, 0])
    spread = np.zeros((len(vertices), 3))
    spread[:, 2] = area / area.sum()
    point = np.zeros((len(vertices), 3))
    point[len(vertices) // 2, 2] = 1.0
    assert field.patch_pressure(point).max() > 20.0 * field.patch_pressure(spread).max()


class _Arm:
    """A straight arm along +x: fingertip at 0, elbow at 0.38, shoulder at 0.66."""

    finger = np.array([0.0, 0.0, 0.0])
    elbow = np.array([0.38, 0.0, 0.0])
    shoulder = np.array([0.66, 0.0, 0.0])


def _arm_mesh(count: int = 60, width: float = 0.02):
    """A flat strip along the arm's axis; collinear vertices would give zero-area triangles."""
    xs = np.linspace(0.0, 0.66, count)
    lower = np.stack([xs, np.zeros(count), np.zeros(count)], axis=1)
    upper = np.stack([xs, np.full(count, width), np.zeros(count)], axis=1)
    vertices = np.concatenate([lower, upper])
    faces = []
    for i in range(count - 1):
        faces += [[i, i + 1, count + i], [i + 1, count + i + 1, count + i]]
    return vertices, np.asarray(faces, dtype=np.int64)


def test_the_bands_follow_the_landmarks_and_the_elbow_gets_its_own():
    vertices, _ = _arm_mesh()
    bands = band_of(vertices, _Arm.finger, _Arm.elbow, _Arm.shoulder)
    named = [BANDS[b] for b in bands[: len(vertices) // 2]]  # the lower row spans the whole arm
    vertices = vertices[: len(vertices) // 2]
    assert named[0] == "hand"                      # at the fingertip
    assert named[len(vertices) // 4] == "forearm"  # mid-forearm
    # Either side of the elbow is the elbow band, which is where a sleeve catches.
    near_elbow = np.argmin(np.abs(vertices[:, 0] - 0.38))
    assert named[near_elbow] == "elbow"
    assert named[-1] == "upperarm"                 # at the shoulder
    assert set(named) == set(BANDS)


def test_the_summary_separates_a_gripping_sleeve_from_a_pushing_one():
    vertices, faces = _arm_mesh()
    field = PressureMap(vertices, faces, _Arm.finger, _Arm.elbow, _Arm.shoulder)
    # A sleeve squeezing the limb: opposing forces whose vector sum cancels but which the skin feels.
    squeeze = np.zeros((len(vertices), 3))
    # The two strip rows at the elbow, pressing toward each other.
    mid = int(np.argmin(np.abs(vertices[: len(vertices) // 2, 0] - 0.38)))
    squeeze[mid] = [0.0, 3.0, 0.0]
    squeeze[mid + len(vertices) // 2] = [0.0, -3.0, 0.0]
    out = field.summarise(squeeze)
    assert out["elbow_resultant_n"] == pytest.approx(0.0, abs=1e-9)
    assert out["elbow_summed_n"] == pytest.approx(6.0)
    assert out["peak_band"] == "elbow" and out["peak_kpa"] > 0.0


def test_the_summary_reports_every_band_even_when_nothing_touches():
    vertices, faces = _arm_mesh()
    out = PressureMap(vertices, faces, _Arm.finger, _Arm.elbow, _Arm.shoulder).summarise(np.zeros((len(vertices), 3)))
    for name in BANDS:
        assert out[f"{name}_resultant_n"] == 0.0 and out[f"{name}_peak_kpa"] == 0.0
    assert out["peak_kpa"] == 0.0


def test_a_prefix_namespaces_the_keys():
    vertices, faces = _arm_mesh()
    out = PressureMap(vertices, faces, _Arm.finger, _Arm.elbow, _Arm.shoulder).summarise(
        np.zeros((len(vertices), 3)), prefix="arm_")
    assert "arm_peak_kpa" in out and "peak_kpa" not in out


def test_the_neighbour_lists_include_the_vertex_itself():
    vertices, _ = _grid(5, side=0.1)
    for index, neighbours in enumerate(patch_neighbours(vertices, PATCH_RADIUS_M)):
        assert index in neighbours
