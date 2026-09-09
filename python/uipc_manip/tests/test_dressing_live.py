"""CPU tests for the online drape bake's geometry and for live cell placement."""

import numpy as np
import pytest

from uipc_manip.dressing_bake import (
    PULL_SCHEDULES,
    canonical_transform,
    cuff_semantics,
    drop_reported_faces,
    load_index_tables,
    socket_frame,
)
from uipc_manip.dressing_live import (
    LiveCellConfig,
    apply_transform,
    build_target_socket_frame,
    canonical_to_world_transform,
)


def _ring(centre, axis, radius, n=6):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    e1 = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = e1 - (e1 @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([centre + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles])


def test_socket_frame_is_orthonormal_and_carries_the_insertion_axis():
    centre = np.array([0.1, -0.5, 0.9])
    axis = np.array([1.0, 0.05, 0.1])
    axis = axis / np.linalg.norm(axis)
    T = socket_frame(centre, axis, _ring(centre, axis, 0.1))
    assert np.allclose(T[:3, 3], centre)
    assert np.allclose(T[:3, 2], axis)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)


def test_cuff_semantics_measures_a_synthetic_opening():
    centre = np.array([0.0, 0.0, 1.0])
    axis = np.array([1.0, 0.0, 0.0])
    ring = _ring(centre, axis, 0.09)
    # Two alignment points: [shoulder_end, hand_end]; the insertion axis runs from cuff into sleeve.
    verts = np.concatenate([ring, np.stack([centre + axis * 0.4, centre - axis * 0.05])])
    sem = cuff_semantics(verts, np.arange(6), np.array([6, 7]))
    assert sem["opening_radius_mean_m"] == pytest.approx(0.09, abs=1e-9)
    assert np.allclose(sem["opening_center"], centre)
    assert np.allclose(sem["insertion_axis"], axis)


def test_placement_puts_the_opening_outside_the_fingertip_on_the_arm_axis():
    finger = np.array([0.0, 0.0, 1.0])
    shoulder = np.array([0.5, 0.1, 1.0])
    clearance = 0.1
    target = build_target_socket_frame(finger, shoulder, clearance=clearance)
    arm_axis = (shoulder - finger) / np.linalg.norm(shoulder - finger)
    assert np.allclose(target[:3, 2], arm_axis)
    assert np.allclose(target[:3, 3], finger - arm_axis * clearance)
    # A canonical opening maps onto that frame, so its centre lands there.
    centre = np.array([0.116, -0.492, 0.896])
    axis = np.array([0.997, 0.042, 0.061])
    axis = axis / np.linalg.norm(axis)
    source = socket_frame(centre, axis, _ring(centre, axis, 0.099))
    transform = canonical_to_world_transform(source, finger, shoulder, clearance=clearance)
    assert np.allclose(apply_transform(centre[None, :], transform)[0], target[:3, 3])
    # The rigid map preserves the opening radius.
    ring_world = apply_transform(_ring(centre, axis, 0.099), transform)
    assert np.linalg.norm(ring_world - ring_world.mean(axis=0), axis=1).mean() == pytest.approx(0.099, abs=1e-9)


def test_canonical_transform_grounds_and_flips_to_z_up():
    rng = np.random.default_rng(0)
    raw = rng.uniform(-0.1, 0.1, size=(50, 3))
    placed = canonical_transform(raw, "tshirt_26", 4.0)
    assert placed.shape == raw.shape
    # Scaling is uniform, so pairwise distances scale by exactly the factor.
    d_raw = np.linalg.norm(raw[1:] - raw[0], axis=1)
    d_placed = np.linalg.norm(placed[1:] - placed[0], axis=1)
    assert np.allclose(d_placed, 4.0 * d_raw, rtol=1e-9)
    # A gown uses a different Euler triple and offset than a tshirt.
    assert not np.allclose(canonical_transform(raw, "hospital_gown", 4.0), placed)


def test_dropping_reported_faces_removes_every_triangle_touching_them(tmp_path):
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.int32)
    report = tmp_path / "close_mesh.obj"
    # An edge-only report: two vertices, no triangle. Vertex 4 belongs to one face.
    report.write_text("v 2.0 0.0 0.0\nv 1.0 1.0 0.0\nl 1 2\n")
    kept = drop_reported_faces(verts, faces, report)
    assert kept.tolist() == [[0, 1, 2]]
    with pytest.raises(RuntimeError, match="back to the source mesh"):
        (tmp_path / "far.obj").write_text("v 9.0 9.0 9.0\n")
        drop_reported_faces(verts, faces, tmp_path / "far.obj")


def test_every_bakeable_garment_has_a_schedule_and_in_range_indices():
    tables = load_index_tables(LiveCellConfig().bake.index_module)
    for garment in PULL_SCHEDULES:
        grasp = np.asarray(tables.grasping_particle_indices[garment])
        opening = np.asarray(tables.shoulder_polygon_particle_indices[garment])
        align = np.asarray(tables.alignment_line_indices[garment])
        assert grasp.size > 0 and opening.size == 6 and align.size == 2
        assert grasp.min() >= 0 and opening.min() >= 0 and align.min() >= 0
        assert float(tables.cloth_scales[garment]) > 0.0
        schedule = PULL_SCHEDULES[garment]
        assert schedule.distance_m > 0 and schedule.pull_seconds > 0 and schedule.settle_seconds > 0
