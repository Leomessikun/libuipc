"""CPU tests for the dressing reward geometry, threading test, and asset helpers."""

import numpy as np
import pytest

from uipc_manip.dressing_assets import erode_arm_mesh, vertex_normals
from uipc_manip.dressing_reward import WangRewardConfig, early_turn, line_triangles, opening_threaded, wang_progress


def _ring(center, axis, radius, n=6):
    axis = axis / np.linalg.norm(axis)
    e1 = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = e1 - (e1 @ axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([center + radius * (np.cos(a) * e1 + np.sin(a) * e2) for a in angles])


@pytest.fixture
def arm():
    finger = np.array([0.0, 0.0, 1.0])
    elbow = np.array([0.3, 0.0, 1.0])
    shoulder = np.array([0.3, 0.3, 1.0])
    return finger, elbow, shoulder


def _cloth_with_ring(ring):
    # A tiny "garment": the ring plus one far-away vertex; polygon = the ring.
    cloth = np.concatenate([ring, np.array([[5.0, 5.0, 5.0]])])
    polygon = np.arange(6)
    tri = np.array([(0, i, i + 1) for i in range(1, 5)])
    return cloth, polygon, tri


def test_line_triangles_hits_a_facing_ring(arm):
    finger, elbow, _ = arm
    ring = _ring(np.array([0.1, 0.0, 1.0]), elbow - finger, 0.08)
    cloth, polygon, tri = _cloth_with_ring(ring)
    hit, pts = line_triangles(elbow, (finger - elbow) / 0.3, cloth[tri])
    assert hit.any()
    assert np.allclose(pts[hit][:, 0], 0.1, atol=1e-6)


def test_progress_before_on_forearm_and_on_upperarm(arm):
    finger, elbow, shoulder = arm
    cfg = WangRewardConfig()
    human = np.array([[10.0, 10.0, 10.0]])
    kwargs = dict(polygon_idx=np.arange(6), triangle_idx=np.array([(0, i, i + 1) for i in range(1, 5)]), cuff_idx=np.arange(6), finger=finger, elbow=elbow, shoulder=shoulder, human_points=human, cfg=cfg)
    # In front of the fingertip: pre-insertion, reward is minus the finger distance.
    cloth, _, _ = _cloth_with_ring(_ring(np.array([-0.1, 0.0, 1.0]), elbow - finger, 0.08))
    pre = wang_progress(cloth, **kwargs)
    assert not pre.on_forearm and not pre.on_upperarm
    assert pre.reward == pytest.approx(-0.1, abs=1e-6)
    # A third of the way up the forearm.
    cloth, _, _ = _cloth_with_ring(_ring(np.array([0.1, 0.0, 1.0]), elbow - finger, 0.08))
    fore = wang_progress(cloth, **kwargs)
    assert fore.on_forearm and not fore.on_upperarm
    assert fore.forearm_distance == pytest.approx(0.1, abs=1e-6)
    assert fore.forearm_ratio == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert fore.reward == pytest.approx(0.1, abs=1e-6)
    # Half way up the upper arm: forearm length plus five times the upper-arm distance.
    cloth, _, _ = _cloth_with_ring(_ring(np.array([0.3, 0.15, 1.0]), shoulder - elbow, 0.08))
    up = wang_progress(cloth, **kwargs)
    assert up.on_upperarm
    assert up.upperarm_ratio == pytest.approx(0.5, abs=1e-6)
    assert up.task_reward == pytest.approx(0.3 + 5.0 * 0.15, abs=1e-6)
    assert up.center_align == pytest.approx(cfg.center_align_reward_w)
    assert up.reward > fore.reward > pre.reward


def test_collision_penalty_when_cuff_touches_body(arm):
    finger, elbow, shoulder = arm
    cfg = WangRewardConfig()
    ring = _ring(np.array([-0.1, 0.0, 1.0]), elbow - finger, 0.08)
    cloth, polygon, tri = _cloth_with_ring(ring)
    near = wang_progress(cloth, polygon_idx=polygon, triangle_idx=tri, cuff_idx=polygon, finger=finger, elbow=elbow, shoulder=shoulder, human_points=ring[:1] + 1e-4, cfg=cfg)
    assert near.collision == -1.0
    assert near.reward == pytest.approx(-0.1 + cfg.collision_w * -1.0, abs=1e-6)


def test_opening_threaded_requires_ring_between_finger_and_shoulder(arm):
    finger, _, shoulder = arm
    axis = shoulder - finger
    on_arm = _ring(finger + 0.4 * axis, axis, 0.08)
    assert opening_threaded(on_arm, np.arange(6), finger, shoulder)[0]
    in_front = _ring(finger - 0.2 * axis, axis, 0.08)
    assert not opening_threaded(in_front, np.arange(6), finger, shoulder)[0]
    beside = _ring(finger + 0.4 * axis + np.array([0.0, 0.0, 0.3]), axis, 0.08)
    assert not opening_threaded(beside, np.arange(6), finger, shoulder)[0]


def test_erode_arm_mesh_shrinks_toward_axis():
    # A short tube around the x axis, with outward normals from the winding.
    n_ring, n_len, r = 12, 5, 0.05
    verts, faces = [], []
    for i in range(n_len):
        for j in range(n_ring):
            a = 2.0 * np.pi * j / n_ring
            verts.append([0.1 * i, r * np.cos(a), r * np.sin(a)])
    for i in range(n_len - 1):
        for j in range(n_ring):
            a, b = i * n_ring + j, i * n_ring + (j + 1) % n_ring
            c, d = a + n_ring, b + n_ring
            faces += [[a, b, c], [b, d, c]]
    verts, faces = np.asarray(verts), np.asarray(faces)
    normals = vertex_normals(verts, faces)
    assert normals.shape == verts.shape
    eroded = erode_arm_mesh(verts, faces, 0.01, np.array([0.0, 0.0, 0.0]), np.array([0.4, 0.0, 0.0]))
    radii = np.linalg.norm(eroded[:, 1:], axis=1)
    # A 12-gon's averaged vertex normals are not exactly radial, hence the loose tolerance.
    assert np.allclose(radii, r - 0.01, atol=1e-4)
    # Flipped winding must still erode inward.
    eroded_flipped = erode_arm_mesh(verts, faces[:, ::-1], 0.01, np.array([0.0, 0.0, 0.0]), np.array([0.4, 0.0, 0.0]))
    assert np.allclose(np.linalg.norm(eroded_flipped[:, 1:], axis=1), r - 0.01, atol=1e-4)


def test_early_turn_flags_the_concave_side_of_the_elbow_only(arm):
    finger, elbow, shoulder = arm
    # The arm bends toward +y at the elbow, so the concave side of the bend is the +y / -x corner.
    inside = elbow + np.array([-0.03, 0.04, 0.0])
    outside = elbow + np.array([0.04, -0.04, 0.0])
    assert early_turn(inside, finger, elbow, shoulder)
    assert not early_turn(outside, finger, elbow, shoulder)
    # Same side, but outside both projected quarter-segments: not an early turn. The paper's
    # region is a union of projections, so a point beside the hand at small +y still projects
    # into the first quarter of the upper arm; 0.1 m of +y puts it past that quarter.
    assert not early_turn(finger + np.array([0.0, 0.1, 0.0]), finger, elbow, shoulder)
    # Height off the bend plane does not change the verdict.
    assert early_turn(inside + np.array([0.0, 0.0, 0.1]), finger, elbow, shoulder)
    # Mirrored arm: the verdict follows the geometry, not the world frame.
    mirror = np.array([1.0, -1.0, 1.0])
    assert early_turn(inside * mirror, finger * mirror, elbow * mirror, shoulder * mirror)
    assert not early_turn(outside * mirror, finger * mirror, elbow * mirror, shoulder * mirror)
