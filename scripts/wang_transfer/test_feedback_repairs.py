"""Analytic examples checking fallback and information constraints, not dressing."""
import unittest

import numpy as np

from compile_feedback_repairs import compile_repairs


class FeedbackRepairSemantics(unittest.TestCase):
    def test_mixture_can_preserve_improvement_when_each_fixed_repair_regresses(self):
        r = compile_repairs([[1.], [1.]], [[[.9, .1]], [[.1, .9]]], [.4, .4])
        self.assertAlmostEqual(r["worst_model_gain"], .1)
        self.assertAlmostEqual(r["best_single_repair_worst_gain"], 0.)
        np.testing.assert_allclose(r["mixture_weights"], [0., .5, .5])

    def test_only_observable_response_allows_branch_specific_repairs(self):
        values = [[[.9, .1], [.9, .1]], [[.1, .9], [.1, .9]]]
        alias = compile_repairs([[.5, .5], [.5, .5]], values, [.4, .4])
        observable = compile_repairs([[.9, .1], [.1, .9]], values, [.4, .4])
        self.assertAlmostEqual(alias["worst_model_gain"], .1)
        self.assertAlmostEqual(observable["worst_model_gain"], .42)
        # A latent-state oracle's .5 gain is unavailable from either observation.
        self.assertLess(observable["worst_model_gain"], .5)

    def test_unsupported_repairs_keep_frozen_policy(self):
        r = compile_repairs([[1.], [1.]], [[[.1, .1]], [[.2, .2]]], [.4, .4])
        self.assertAlmostEqual(r["worst_model_gain"], 0.)
        self.assertEqual(r["mixture_weights"][0], 1.)


if __name__ == "__main__":
    unittest.main()
