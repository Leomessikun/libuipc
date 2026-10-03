"""Information-boundary and physical-label checks for the effect-code pilot."""
import unittest

import numpy as np

from effect_codes import context_inputs, effect_vector, kmeans, nearest


class EffectCodeTests(unittest.TestCase):
    def descriptors(self):
        d = np.zeros((9, 4, 9), np.float32)
        d[:, :, 3] = 1.
        d[:, :, 6] = .08
        return d

    def test_isolated_wrapping_flicker_is_not_an_event(self):
        d = self.descriptors(); d[-1, 0, 8] = 1.
        x = effect_vector(d)
        np.testing.assert_array_equal(x[20:32], 0.)

    def test_persistent_entry_and_exit_remain_distinguishable(self):
        entry = self.descriptors(); entry[3:, 0, 8] = 1.
        exit_path = self.descriptors(); exit_path[:3, 0, 8] = 1.
        self.assertEqual(effect_vector(entry)[20], 4.)
        self.assertEqual(effect_vector(entry)[24], 4.)
        self.assertEqual(effect_vector(exit_path)[20], -4.)
        self.assertEqual(effect_vector(exit_path)[28], 4.)

    def test_invalid_geometry_is_not_a_zero_effect(self):
        d = self.descriptors(); d[-1, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            effect_vector(d)

    def test_current_and_future_actions_cannot_enter_context(self):
        rng = np.random.default_rng(0)
        features, tools = rng.normal(size=(10, 50)), rng.normal(size=(10, 3))
        actions, base = rng.normal(size=(9, 6)), rng.normal(size=(10, 6))
        a = context_inputs(features, tools, actions, base, np.eye(3))
        changed = actions.copy(); changed[5:] += 1000.
        moved = features.copy(); moved[6:] += 1000.
        b = context_inputs(moved, tools, changed, base, np.eye(3))
        np.testing.assert_array_equal(a[:6], b[:6])
        self.assertGreater(np.linalg.norm(a[6] - b[6]), 1.)

    def test_online_truncated_history_matches_episode_history(self):
        rng = np.random.default_rng(1)
        features, tools = rng.normal(size=(10, 50)), rng.normal(size=(10, 3))
        actions, base = rng.normal(size=(9, 6)), rng.normal(size=(10, 6))
        whole = context_inputs(features, tools, actions, base, np.eye(3))
        local = context_inputs(features[4:8], tools[4:8], actions[4:7], base[4:8], np.eye(3))
        np.testing.assert_array_equal(whole[7], local[-1])

    def test_partition_only_depends_on_supplied_training_examples(self):
        x = np.r_[np.zeros((10, 2)), np.ones((10, 2))]
        centers = kmeans(x, 2)
        assignment = nearest(x, centers)
        self.assertEqual(len(np.unique(assignment[:10])), 1)
        self.assertNotEqual(assignment[0], assignment[-1])
        np.testing.assert_array_equal(centers, kmeans(x, 2))


if __name__ == '__main__':
    unittest.main()
