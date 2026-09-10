"""CPU tests for what makes tshirt_68 and tshirt_392 usable as live cells.

The socket roll, the crossing untangler and the close-pair separation are pure
geometry; the last test reads Newton's offline tshirt_68 drape when it is on disk.
"""

from pathlib import Path

import numpy as np
import pytest

from uipc_manip.dressing_bake import crossing_pairs, separate_close_primitives, socket_frame, untangle_crossings
from uipc_manip.dressing_live import (
    OFFLINE_DRAPE_DIR,
    apply_transform,
    build_target_socket_frame,
    canonical_to_world_transform,
    gravity_aligned_socket,
)


def _ring(centre, axis, radius, n=6):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    e1 = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([centre + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles])


def test_the_re_rolled_socket_keeps_origin_and_axis_and_does_not_depend_on_the_first_ring_vertex():
    centre = np.array([0.12, -0.5, 0.9])
    axis = np.array([0.997, 0.04, 0.06])
    axis /= np.linalg.norm(axis)
    ring = _ring(centre, axis, 0.09)
    frames = [gravity_aligned_socket(socket_frame(centre, axis, np.roll(ring, k, axis=0))) for k in range(6)]
    for T in frames:
        assert np.allclose(T, frames[0], atol=1e-12)
        assert np.allclose(T[:3, 3], centre) and np.allclose(T[:3, 2], axis)
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0)
        # +Y is canonical up with the insertion component removed.
        up = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
        assert np.allclose(T[:3, 1], up / np.linalg.norm(up))


def test_placement_keeps_the_baked_hang_whatever_the_ring_order():
    """A point hanging under the opening in the bake must hang under it on every arm."""
    centre = np.array([0.12, -0.5, 0.9])
    axis = np.array([1.0, 0.0, 0.1])
    axis /= np.linalg.norm(axis)
    ring = _ring(centre, axis, 0.09)
    hanging = centre + np.array([0.05, 0.0, -0.3])
    finger, shoulder = np.array([0.0, 0.0, 1.0]), np.array([0.45, 0.2, 1.25])
    placed = []
    for k in range(6):
        source = socket_frame(centre, axis, np.roll(ring, k, axis=0))
        T = canonical_to_world_transform(source, finger, shoulder, clearance=0.2, hang_as_baked=True)
        c, h = apply_transform(np.stack([centre, hanging]), T)
        assert np.allclose(c, build_target_socket_frame(finger, shoulder, clearance=0.2)[:3, 3])
        assert h[2] < c[2] - 0.25  # still below the opening, not flipped over the arm
        placed.append(h)
    assert np.allclose(placed, placed[0], atol=1e-12)


def test_the_default_placement_uses_the_measured_socket_unchanged():
    """The re-roll is opt-in: by default the transform is the plain socket-to-target map."""
    centre = np.array([0.12, -0.5, 0.9])
    axis = np.array([1.0, 0.0, 0.1]) / np.linalg.norm([1.0, 0.0, 0.1])
    source = socket_frame(centre, axis, np.roll(_ring(centre, axis, 0.09), 2, axis=0))
    finger, shoulder = np.array([0.0, 0.0, 1.0]), np.array([0.45, 0.2, 1.25])
    target = build_target_socket_frame(finger, shoulder, clearance=0.2)
    assert np.allclose(canonical_to_world_transform(source, finger, shoulder, clearance=0.2), target @ np.linalg.inv(source))
    rerolled = canonical_to_world_transform(source, finger, shoulder, clearance=0.2, hang_as_baked=True)
    assert not np.allclose(rerolled, target @ np.linalg.inv(source))


def test_the_default_axis_is_the_forearm_and_the_opening_starts_on_it():
    """The opening centre lands on the fingertip-to-elbow line, a clearance step outside the fingertip."""
    from uipc_manip.dressing_live import LiveCellConfig

    finger, elbow, shoulder = np.array([0.0, 0.0, 1.0]), np.array([0.35, 0.0, 1.0]), np.array([0.5, 0.0, 1.3])
    centre = np.array([0.12, -0.5, 0.9])
    axis = np.array([1.0, 0.0, 0.1]) / np.linalg.norm([1.0, 0.0, 0.1])
    source = socket_frame(centre, axis, _ring(centre, axis, 0.09))
    cfg = LiveCellConfig()
    assert cfg.axis_landmark == "right_elbow" and cfg.clearance_m == pytest.approx(0.09)
    T = canonical_to_world_transform(source, finger, {"right_elbow": elbow, "right_shoulder": shoulder}[cfg.axis_landmark], clearance=cfg.clearance_m)
    opening = apply_transform(centre[None, :], T)[0]
    forearm = (elbow - finger) / np.linalg.norm(elbow - finger)
    rel = opening - finger
    assert rel @ forearm == pytest.approx(-0.09)
    assert np.linalg.norm(rel - (rel @ forearm) * forearm) == pytest.approx(0.0, abs=1e-12)
    # The opening plane faces along the forearm: the insertion axis is the flipped forearm.
    assert np.allclose(T[:3, :3] @ axis, -forearm)


def _poked_sheet():
    """A flat two-triangle sheet with a third triangle whose apex pokes 2 mm through it."""
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
         [0.3, 0.3, 0.5], [0.6, 0.3, 0.5], [0.45, 0.45, -0.002]]
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6]], dtype=np.int32)
    return verts, faces


def test_crossing_pairs_finds_the_poking_edges_and_untangling_clears_them():
    verts, faces = _poked_sheet()
    edges, tris = crossing_pairs(verts, faces)
    assert {tuple(e) for e in edges.tolist()} == {(4, 6), (5, 6)}
    assert set(tris.tolist()) <= {0, 1}
    fixed, moved = untangle_crossings(verts, faces, margin=1.0e-3)
    assert moved.tolist() == [6]
    assert len(crossing_pairs(fixed, faces)[1]) == 0
    # Only the apex moved, back to the side of its own triangle, a margin clear of the sheet.
    assert np.allclose(np.delete(fixed, 6, axis=0), np.delete(verts, 6, axis=0))
    assert fixed[6, 2] == pytest.approx(1.0e-3)


def test_separating_a_near_edge_pair_opens_it_to_the_gap():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, -0.5, 2.0e-5], [0.5, 0.5, 2.0e-5], [3.0, 3.0, 3.0]])
    report_points = verts[[0, 1, 2, 3]]
    gap = 1.3e-3
    out = separate_close_primitives(verts, report_points, np.array([[0, 1], [2, 3]]), np.zeros((0, 3)), gap)
    assert out is not None
    assert abs(out[2, 2] - out[0, 2]) == pytest.approx(gap)
    assert np.allclose(out[4], verts[4])
    assert separate_close_primitives(verts, report_points, np.zeros((0, 2)), np.zeros((0, 3)), gap) is None


@pytest.mark.skipif(not Path(OFFLINE_DRAPE_DIR, "tshirt_68__s3.7000.npz").exists(), reason="offline drape not on disk")
def test_the_offline_tshirt_68_drape_is_untangled_with_its_semantics_intact():
    from uipc_manip.dressing_live import load_offline_drape

    drape = load_offline_drape("tshirt_68")
    with np.load(Path(OFFLINE_DRAPE_DIR, "tshirt_68__s3.7000.npz")) as data:
        raw = np.asarray(data["particle_q"], dtype=np.float64)[: len(drape["cloth"])]
    assert len(crossing_pairs(raw, drape["faces"])[1]) > 0
    assert len(crossing_pairs(drape["cloth"], drape["faces"])[1]) == 0
    moved = np.flatnonzero(np.linalg.norm(drape["cloth"] - raw, axis=1) > 0.0)
    semantic = np.concatenate([drape["grasp_idx"], drape["opening_idx"], drape["alignment_idx"]])
    assert 0 < moved.size <= 12 and not np.isin(moved, semantic).any()
    assert np.linalg.norm(drape["cloth"] - raw, axis=1).max() < 0.01


def test_reach_past_opening_measures_the_outward_side_only():
    socket = np.eye(4)
    socket[:3, 3] = [0.1, 0.0, 1.0]
    cloth = np.array([[0.1, 0.0, 1.0], [0.1, 0.0, 1.4], [0.1, 0.2, 0.75], [0.1, 0.0, 0.9]])
    from uipc_manip.dressing_live import reach_past_opening

    assert reach_past_opening(cloth, socket) == pytest.approx(0.25)
    assert reach_past_opening(cloth[:2], socket) == pytest.approx(0.0)


def _index_tables_present() -> bool:
    from uipc_manip.dressing_live import LiveCellConfig

    return Path(LiveCellConfig().bake.index_module).exists() and Path(LiveCellConfig().bake.garment_dir).exists()


def _box(lo, hi):
    """A closed axis-aligned box as (points, faces)."""
    import trimesh

    box = trimesh.creation.box(bounds=np.array([lo, hi], dtype=np.float64))
    return np.asarray(box.vertices, dtype=np.float64), np.asarray(box.faces, dtype=np.int64)


def test_garment_arm_gap_reports_crossings_as_zero_and_otherwise_the_surface_distance():
    from uipc_manip.dressing_live import garment_arm_gap

    arm, arm_faces = _box([0.0, -0.01, -0.01], [0.6, 0.01, 0.01])
    fin = np.array([[0.3, 0.0, 0.03], [0.4, 0.0, 0.03], [0.35, 0.0, 0.08]])
    assert garment_arm_gap(fin, np.array([[0, 1, 2]]), arm, arm_faces) == pytest.approx(0.02)
    # Dropping the fin through the top face is a crossing, whatever the vertex distances say.
    through = fin - np.array([0.0, 0.0, 0.05])
    assert garment_arm_gap(through, np.array([[0, 1, 2]]), arm, arm_faces) == 0.0


@pytest.mark.skipif(not _index_tables_present(), reason="garment index tables or raw meshes are unavailable")
def test_a_cell_whose_sleeve_would_reach_the_arm_moves_out_until_it_clears():
    """A sleeve longer than the clearance is pushed out in 1 cm steps; a short one is left alone."""
    from uipc_manip.dressing_live import CanonicalDrape, LiveCellConfig, LiveCellFactory, SimpleBody, garment_arm_gap

    finger, shoulder = np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.0, 1.0])
    arm, arm_faces = _box([0.0, -0.01, 0.99], [0.6, 0.01, 1.01])
    landmarks = {"right_finger": finger, "right_elbow": 0.5 * (finger + shoulder), "right_shoulder": shoulder}
    body = SimpleBody(arm, arm_faces, arm, landmarks, "test")
    socket = np.eye(4)

    def drape(reach):
        # A sleeve fin in the canonical frame: two ring points and an apex ``reach`` outward past the opening.
        cloth = np.array([[0.0, 0.05, 0.0], [0.0, -0.05, 0.0], [0.0, 0.0, -reach], [0.0, 0.0, 0.3]])
        return CanonicalDrape("g", 1.0, cloth, np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32), np.zeros(3), np.array([3]),
                              np.array([3]), np.array([0, 1]), np.array([3, 2]), socket, 0.05)

    factory = LiveCellFactory(LiveCellConfig(clearance_m=0.20, min_arm_gap_m=0.003))
    factory._bodies[0] = body
    factory._drapes["short"], factory._drapes["long"] = drape(0.10), drape(0.25)
    assert factory.clearance_for("short", 0) == pytest.approx(0.20)
    long_clearance = factory.clearance_for("long", 0)
    assert long_clearance == pytest.approx(0.26)  # the apex has to clear the fingertip face by 3 mm
    cloth, _ = factory._drapes["long"].place(body.landmarks, clearance=long_clearance)
    assert garment_arm_gap(cloth, factory._drapes["long"].faces, arm, arm_faces) >= 0.003
    # A cell that clears nowhere in range is refused rather than forced into the world.
    from uipc_manip.dressing_live import NoClearPlacement

    tight = LiveCellFactory(LiveCellConfig(clearance_m=0.20, min_arm_gap_m=0.003, max_clearance_m=0.24))
    tight._bodies[0], tight._drapes["long"] = body, drape(0.25)
    with pytest.raises(NoClearPlacement, match="long on body 0"):
        tight.clearance_for("long", 0)
