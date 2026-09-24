import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.claude/worktrees/residual-rl/python'))
from uipc_manip.obs import ObsSpec
from rollout_controls import ControlProfile, crop_observation, force_input, rigid_rotation, tracking_scale


class ControlsTest(unittest.TestCase):
    def test_force_reaction_clip_and_causal_smoothing(self):
        profile = ControlProfile(force_source='body_contact', force_scale=.01, force_clip=.04, force_ema=.5)
        result = force_input(profile, [0., 10., 0.], [90., 0., 0.], np.zeros(3))
        np.testing.assert_allclose(result, [0., -.02, 0.], atol=1e-8)
        result = force_input(profile, [0., 0., 0.], [90., 0., 0.], result)
        np.testing.assert_allclose(result, [0., -.01, 0.], atol=1e-8)

    def test_crop_preserves_arm_and_nearby_cloth(self):
        spec = ObsSpec(10)
        points = np.array([[0., 0., 0.], [.01, 0., -.02], [0., 0., -.4]])
        features = np.array([[0., 1., 0., 0.], [1., 0., 0., 0.], [1., 0., 0., 0.]])
        original = spec.pack_labeled(points, features, np.array([0., 0., .3]), np.zeros(3), attached=True)
        cropped = crop_observation(original, spec, ControlProfile(cloth_crop_below_arm_m=.3))
        _, feat, valid, _ = spec.unpack_numpy(cropped[None])
        self.assertEqual(int(feat[0, :, 1].sum()), 1)
        self.assertEqual(int(feat[0, :, 0].sum()), 1)
        self.assertEqual(int(feat[0, :, 3].sum()), 1)
        self.assertNotEqual(int(valid.sum()), int(spec.unpack_numpy(original[None])[2].sum()))

    def test_guard_includes_rotation_of_grasp_patch(self):
        offsets = np.array([[.1, 0., 0.], [-.1, 0., 0.]])
        kwargs = dict(anchor=np.zeros(3), offsets=offsets, held=offsets,
                      max_translation=.01, max_rotation=.2, budget=.018)
        scale = tracking_scale(np.array([0, 0, 0, 0, 0, 1]), **kwargs)
        self.assertEqual(scale, .5)
        self.assertEqual(tracking_scale(np.zeros(6), **kwargs), 1.)

    def test_recorded_tool_rotation_maps_initial_grasp_offsets(self):
        before = np.array([[.03, -.02, .01], [-.04, .01, .02], [.01, .05, -.03], [.01, -.04, -.01]])
        theta = .7
        expected = np.array([[np.cos(theta), -np.sin(theta), 0.],
                             [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        after = before @ expected.T
        actual = rigid_rotation(before, after)
        np.testing.assert_allclose(before @ actual.T, after, atol=1e-12)
        np.testing.assert_allclose(actual @ actual.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(actual), 1.)


if __name__ == '__main__':
    unittest.main()
