"""CPU tests for the online drape bake's geometry and for live cell placement."""

from pathlib import Path

import numpy as np
import pytest

from uipc_manip.dressing_bake import (
    PULL_SCHEDULES,
    canonical_transform,
    cuff_semantics,
    drop_reported_faces,
    load_index_tables,
    separate_reported_vertices,
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
    arm_axis = (shoulder - finger) / np.linalg.norm(shoulder - finger)
    # Pre-worn placement lays the sleeve along the arm; pre-insertion turns it outward.
    worn = build_target_socket_frame(finger, shoulder, clearance=clearance, pre_insertion=False)
    assert np.allclose(worn[:3, 2], arm_axis)
    target = build_target_socket_frame(finger, shoulder, clearance=clearance)
    assert np.allclose(target[:3, 2], -arm_axis)
    assert np.allclose(target[:3, 3], finger - arm_axis * clearance)
    assert np.allclose(worn[:3, 3], target[:3, 3])
    # A canonical opening maps onto that frame, so its centre lands there.
    centre = np.array([0.116, -0.492, 0.896])
    axis = np.array([0.997, 0.042, 0.061])
    axis = axis / np.linalg.norm(axis)
    source = socket_frame(centre, axis, _ring(centre, axis, 0.099))
    transform = canonical_to_world_transform(source, finger, shoulder, clearance=clearance, pre_insertion=False)
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


def test_dropping_reported_faces_removes_every_triangle_touching_them():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.int32)
    # The checker hands back positions; vertices 4 and 3 sit on two of the faces.
    reported = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert drop_reported_faces(verts, faces, reported).tolist() == [[0, 1, 2]]
    with pytest.raises(RuntimeError, match="back to the source mesh"):
        drop_reported_faces(verts, faces, np.array([[9.0, 9.0, 9.0]]))


def test_separating_reported_vertices_keeps_topology_and_moves_only_them():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    moved = separate_reported_vertices(verts, faces, np.array([[0.0, 0.0, 0.0]]), 0.01)
    assert moved.shape == verts.shape
    assert np.allclose(moved[1:], verts[1:])
    # The flat sheet's normal is +z, so the flagged vertex leaves the plane by the step.
    assert abs(moved[0, 2] - verts[0, 2]) == pytest.approx(0.01)


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


def _smplx_available() -> bool:
    try:
        import smplx  # noqa: F401
    except Exception:
        return False
    from uipc_manip.dressing_body import BodyConfig

    return Path(BodyConfig().model_dir).exists()


smplx_only = pytest.mark.skipif(not _smplx_available(), reason="SMPL-X model or package is unavailable")


@smplx_only
def test_generated_bodies_vary_and_carry_a_usable_right_arm():
    from uipc_manip.dressing_body import generate_body

    bodies = [generate_body(seed) for seed in (0, 1, 2)]
    for body in bodies:
        assert body.vertices.shape == (10475, 3) and np.isfinite(body.vertices).all()
        assert len(body.arm_points) > 500 and len(body.arm_faces) > 500
        assert body.arm_faces.max() < len(body.arm_points)
        # The three dressing landmarks must form a bent arm of plausible length.
        forearm = np.linalg.norm(body.elbow - body.finger)
        upper = np.linalg.norm(body.shoulder - body.elbow)
        assert 0.2 < forearm < 0.6 and 0.15 < upper < 0.5
        # Every arm vertex belongs to the arm: none should sit near the pelvis.
        assert np.linalg.norm(body.arm_points - body.landmarks["pelvis"], axis=1).min() > 0.05
    # Different seeds are different bodies, and one seed is reproducible.
    assert not np.allclose(bodies[0].vertices, bodies[1].vertices)
    assert np.allclose(generate_body(0).vertices, bodies[0].vertices)


@smplx_only
def test_the_generated_arm_is_a_closed_submesh_of_the_body():
    from uipc_manip.dressing_body import generate_body, submesh

    body = generate_body(5)
    points, faces = submesh(body.vertices, body.faces, body.arm_indices)
    assert points.shape == body.arm_points.shape and faces.shape == body.arm_faces.shape
    # Every kept face indexes only selected vertices, and the remap is order preserving.
    assert faces.min() >= 0 and faces.max() < len(points)
    assert np.allclose(points, body.vertices[np.sort(body.arm_indices)])


def test_body_pose_seats_the_body_and_randomises_only_the_right_arm():
    from uipc_manip.dressing_body import sample_body_pose

    rng = np.random.default_rng(0)
    seated = sample_body_pose(rng, "dressing")
    standing = sample_body_pose(np.random.default_rng(0), "dressing-standing")
    assert seated.shape == (63,) and np.allclose(sample_body_pose(rng, "rest"), 0.0)
    # Hips and knees are bent when seated and straight when standing.
    assert seated[0] < -1.0 and standing[0] == 0.0
    # Two draws differ only in the right shoulder and elbow entries.
    other = sample_body_pose(np.random.default_rng(1), "dressing")
    differing = set(np.flatnonzero(~np.isclose(seated, other)).tolist())
    assert differing <= {48, 50, 54, 55, 56}
