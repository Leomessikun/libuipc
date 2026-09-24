"""Geometry regressions: use the sleeve side, and reject an off-arm tube."""
import unittest

import numpy as np

from physical_sleeve import SleeveSections, first_stationary_window, measure


class PhysicalSleeveTests(unittest.TestCase):
    def setUp(self):
        # A sleeve ends at x=0; the mesh continues into a torso at x=3.
        n = 24
        xs = np.linspace(0., 3., 31)
        angles = np.arange(n) * 2 * np.pi / n
        self.vertices = np.array([[x, .07 * np.cos(a), .07 * np.sin(a)] for x in xs for a in angles])
        faces = []
        for j in range(len(xs) - 1):
            for k in range(n):
                a, b = j * n + k, j * n + (k + 1) % n
                faces.extend([[a, b, b + n], [a, b + n, a + n]])
        # Armhole at x=1.1. The closest free boundary must be x=0.
        self.sections = SleeveSections(self.vertices, np.asarray(faces), np.arange(11 * n, 12 * n))
        self.landmarks = np.array([[-.1, 0., 0.], [.55, 0., 0.], [1.2, 0., 0.]])

    def test_sections_follow_sleeve_not_torso(self):
        centers = np.array([p.mean(0) for p in self.sections.points(self.vertices)])
        np.testing.assert_allclose(centers[:, 0], [0., .275, .55, .825], atol=1e-10)
        self.assertTrue(measure(self.sections, self.vertices, self.landmarks)['sleeve_wrapped'])

    def test_off_arm_is_not_threaded(self):
        shifted = self.vertices + [0., .3, 0.]
        self.assertFalse(measure(self.sections, shifted, self.landmarks)['sleeve_wrapped'])

    def test_only_one_section_threaded_is_not_a_complete_sleeve(self):
        displaced = self.vertices.copy()
        displaced[:, 1] += np.clip(displaced[:, 0] * .6, 0., .5)
        result = measure(self.sections, displaced, self.landmarks)
        self.assertTrue(result['rings'][0]['wrapped'])
        self.assertFalse(result['sleeve_wrapped'])

    def test_moving_past_target_is_not_a_stationary_hold(self):
        mask = np.ones(6, dtype=bool)
        moving = np.ones((5, 6)) * .1
        self.assertIsNone(first_stationary_window(mask, moving, 2))
        moving[3:] = 0.
        self.assertEqual(first_stationary_window(mask, moving, 2), 3)
        mask[-1] = False
        self.assertIsNone(first_stationary_window(mask, moving, 2))


if __name__ == '__main__':
    unittest.main()
