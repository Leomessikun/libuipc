"""CPU tests for the privileged dressing state an asymmetric critic reads."""

import numpy as np
import pytest

from uipc_manip.dressing_privileged import (
    PRIVILEGED_DIM,
    PRIVILEGED_GARMENTS,
    PRIVILEGED_LAYOUT,
    arm_frame,
    grip_rotation,
    privileged_state,
)
from uipc_manip.dressing_reward import DressingProgress

PROGRESS = DressingProgress(
    reward=0.0, task_reward=0.1, on_forearm=True, on_upperarm=False, forearm_distance=0.1, upperarm_distance=0.0,
    forearm_ratio=1.0 / 3.0, upperarm_ratio=0.0, collision=0.0, center_align=0.0,
)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def _block(state, name):
    start = 0
    for key, n in PRIVILEGED_LAYOUT:
        if key == name:
            return state[start : start + n]
        start += n
    raise KeyError(name)


def _scene(grip=None):
    """A forearm along +x from the fingertip, the shoulder bent toward +y, a ring around the forearm."""
    rng = np.random.default_rng(0)
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ring = np.stack([[0.1, 0.08 * np.cos(a), 0.08 * np.sin(a)] for a in angles])
    initial = rng.normal(scale=0.01, size=(12, 3))
    return {
        "cloth": np.concatenate([ring, rng.normal(scale=0.05, size=(40, 3)) + [0.05, 0.0, 0.1]]),
        "opening_idx": np.arange(6),
        "finger": np.zeros(3),
        "elbow": np.array([0.3, 0.0, 0.0]),
        "shoulder": np.array([0.35, 0.25, 0.0]),
        "arm_points": np.stack([np.linspace(0.0, 0.3, 31), np.zeros(31), np.zeros(31)], axis=1),
        "tool": np.array([-0.05, 0.02, 0.1]),
        "offsets": initial if grip is None else initial @ grip.T,
        "initial_offsets": initial,
    }


def test_blocks_read_the_arm_frame_geometry():
    grip = _rotation([0.2, 1.0, 0.3], 0.4)
    state = privileged_state(**_scene(grip), progress=PROGRESS, tracking_error=0.004, garment="tshirt_68")
    assert state.shape == (PRIVILEGED_DIM,) and state.dtype == np.float32 and np.isfinite(state).all()
    # The scene's arm frame is the world frame, so every block reads directly.
    np.testing.assert_allclose(_block(state, "arm"), [0.3, 0.35, 0.25], atol=1e-6)
    np.testing.assert_allclose(_block(state, "tool"), [-0.05, 0.02, 0.1], atol=1e-6)
    assert _block(state, "tool_clearance")[0] == pytest.approx(np.linalg.norm([-0.05, 0.02, 0.1]), abs=1e-6)
    np.testing.assert_allclose(_block(state, "tool_rotation"), grip[:, :2].T.reshape(-1), atol=1e-5)
    np.testing.assert_allclose(_block(state, "opening_center"), [0.1, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(np.abs(_block(state, "opening_normal")), [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(_block(state, "opening_radius"), [0.08, 0.0], atol=1e-6)
    np.testing.assert_allclose(_block(state, "progress")[:4], [1.0 / 3.0, 0.0, 1.0, 0.0], atol=1e-6)
    assert _block(state, "tracking_error")[0] == pytest.approx(0.4)
    assert _block(state, "garment").tolist() == [float(g == "tshirt_68") for g in PRIVILEGED_GARMENTS]
    other = privileged_state(**_scene(grip), progress=PROGRESS, tracking_error=0.004, garment="jacket")
    assert not _block(other, "garment").any()


def test_state_does_not_depend_on_where_the_body_stands():
    scene = _scene(_rotation([1.0, 0.0, 0.5], -0.3))
    q, t = _rotation([0.3, -0.4, 1.0], 1.1), np.array([0.4, -1.2, 0.9])
    moved = {k: (v @ q.T + t if k not in ("opening_idx", "offsets", "initial_offsets") else v) for k, v in scene.items()}
    moved["offsets"], moved["initial_offsets"] = scene["offsets"] @ q.T, scene["initial_offsets"] @ q.T
    kwargs = {"progress": PROGRESS, "tracking_error": 0.01, "garment": "tshirt_26"}
    np.testing.assert_allclose(privileged_state(**moved, **kwargs), privileged_state(**scene, **kwargs), atol=1e-5)


def test_grip_rotation_and_arm_frame():
    rng = np.random.default_rng(1)
    patch = rng.normal(size=(20, 3))
    r = _rotation([1.0, 2.0, 3.0], 0.7)
    np.testing.assert_allclose(grip_rotation(patch, patch @ r.T), r, atol=1e-9)
    np.testing.assert_allclose(grip_rotation(patch, patch), np.eye(3), atol=1e-9)
    # A straight arm still yields a right-handed orthonormal frame along the forearm.
    frame = arm_frame(np.zeros(3), np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.0, 0.6]))
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0) and np.allclose(frame[0], [0.0, 0.0, 1.0])
