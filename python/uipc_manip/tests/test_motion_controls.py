import numpy as np
from scipy.spatial.transform import Rotation

from uipc_manip.motion_controls import bounded_correction, gicp, pause_masks


def test_gicp_recovers_small_rigid_motion_without_correspondences():
    rng = np.random.default_rng(91)
    p = rng.normal(size=(150, 3)) * [.12, .035, .07] + [.3, -.4, .9]
    r = Rotation.from_rotvec([.015, -.025, .01]).as_matrix()
    t = np.array([.003, -.002, .004])
    q = (p @ r.T + t)[rng.permutation(len(p))]
    fit, info = gicp(p, q)
    assert info["valid"]
    np.testing.assert_allclose(fit[:3, :3], r, atol=1e-5)
    np.testing.assert_allclose(fit[:3, 3], t, atol=1e-5)


def test_missing_observations_do_not_create_motion():
    transform, info = gicp(np.empty((0, 3)), np.zeros((20, 3)))
    assert not info["valid"]
    np.testing.assert_array_equal(transform, np.eye(4))
    transform[0, 3] = 1.
    np.testing.assert_allclose(bounded_correction(transform, [0, 0, 0], .01), [.01, 0, 0])


def test_causal_pause_does_not_read_future_and_yoke_has_same_budget():
    class Motion:
        def __init__(self, future_gain):
            self.future_gain = future_gain
        def sample(self, t):
            x = max(0., t - 1.) * self.future_gain
            return None, None, np.array([[x, 0, 0]] * 4)
    times = np.arange(20) * .1
    a = pause_masks(Motion(1), times, onset_s=0, horizon_s=.4, threshold_m=.02, yoked_shift=7)
    b = pause_masks(Motion(5), times, onset_s=0, horizon_s=.4, threshold_m=.02, yoked_shift=7)
    np.testing.assert_array_equal(a["causal_pause"][:11], b["causal_pause"][:11])
    assert not a["causal_pause"][:11].any()
    assert a["oracle_pause"][:11].any()
    assert a["oracle_pause"].sum() == a["yoked_pause"].sum()
