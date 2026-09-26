import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from uipc_manip.grab_motion import ARM_POSE_INDICES, BodyMotion, retarget_poses


def test_retarget_preserves_recipient_and_non_arm_pose():
    rng = np.random.default_rng(2)
    source = rng.normal(0, .1, (4, 63))
    base = rng.normal(0, .2, 63)
    out = retarget_poses(source, 10, base, [0, .1, .2], amplitude=.5, ramp_s=0)
    np.testing.assert_allclose(out[0], base, atol=1e-7)
    other = sorted(set(range(21)) - set(ARM_POSE_INDICES))
    np.testing.assert_allclose(out.reshape(-1, 21, 3)[:, other],
                               np.broadcast_to(base.reshape(21, 3)[other], (3, len(other), 3)), atol=1e-7)


def test_rotation_interpolation_crosses_pi_without_full_spin():
    source = np.zeros((2, 63))
    source[:, 16 * 3 + 2] = np.deg2rad([179., -179.])
    out = retarget_poses(source, 1., np.zeros(63), [0., .5, 1.], ramp_s=0)
    angle = Rotation.from_rotvec(out.reshape(-1, 21, 3)[:, 16]).magnitude()
    np.testing.assert_allclose(angle, np.deg2rad([0, 1, 2]), atol=1e-7)


def test_rejects_future_outside_recording():
    with pytest.raises(ValueError, match="beyond"):
        retarget_poses(np.zeros((2, 63)), 30, np.zeros(63), [0., 1.])


def test_motion_sampling_and_pickle_free_roundtrip(tmp_path):
    vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                         [[0, 0, 1], [1, 0, 1], [0, 1, 1]]], float)
    motion = BodyMotion(np.array([0., 1.]), vertices, vertices.copy(),
                        np.array([[0, 1, 2]]), np.array([0, 1]), np.zeros((2, 4, 3)), {})
    motion.save(tmp_path / "motion.npz")
    loaded = BodyMotion.load(tmp_path / "motion.npz")
    np.testing.assert_allclose(loaded.sample(.5)[0], vertices[0] + [0, 0, .5])
    np.testing.assert_array_equal(loaded.sample(-1)[0], vertices[0])
    np.testing.assert_array_equal(loaded.sample(5)[0], vertices[1])
    with pytest.raises(FileExistsError):
        motion.save(tmp_path / "motion.npz")
