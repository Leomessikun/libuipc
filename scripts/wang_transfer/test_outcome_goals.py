"""Physical-coordinate and data-boundary checks for the outcome pilot."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from outcome_goals import InteractionCoordinates, future_indices, transport, valid_prefix
from probe_outcome_repairs import Budget, BudgetExhausted
from train_outcome_goals import dataset


def sleeve_and_arm():
    angle = np.linspace(0., 2 * np.pi, 20, endpoint=False)
    rings = [np.stack((np.full(20, x), .05 * np.cos(angle), .05 * np.sin(angle)), axis=1)
             for x in (.1, .2, .3, .4)]
    vertices = np.concatenate(rings)
    sections = SimpleNamespace(vertices=vertices, armhole=np.arange(60, 80), axis=np.array([1., 0., 0.]),
                               fractions=[0., .25, .5, .75],
                               points=lambda cloth: [cloth[i:i + 20] for i in (0, 20, 40)])
    return sections, np.array([[0., 0., 0.], [.5, 0., 0.], [.7, .3, 0.]])


class TestOutcomeGoals(unittest.TestCase):
    def test_geometry_is_invariant_to_world_pose(self):
        sections, landmarks = sleeve_and_arm()
        original = InteractionCoordinates(sections, landmarks).describe(sections.vertices)
        # A proper, non-axis-aligned rigid transform of garment AND recipient.
        q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
        q[:, 0] *= np.linalg.det(q)
        shift = np.array([2., -3., .8])
        moved = SimpleNamespace(**vars(sections))
        moved.vertices = sections.vertices @ q.T + shift
        moved.axis = sections.axis @ q.T
        actual = InteractionCoordinates(moved, landmarks @ q.T + shift).describe(moved.vertices)
        np.testing.assert_allclose(actual, original, atol=1e-6)

    def test_stationary_source_transports_to_stationary_target(self):
        sections, landmarks = sleeve_and_arm()
        start = InteractionCoordinates(sections, landmarks).describe(sections.vertices)
        target = start.copy()
        target[:, :3] += [.1, -.2, .05]
        target[:, 3:6] = [0., 1., 0.]
        target[:, 6] *= 1.7
        target[:, 7] = .01
        target[:, 8] = 0.
        result = transport(np.repeat(start[None], 25, axis=0), target)
        np.testing.assert_allclose(result, np.repeat(target[None], 25, axis=0), atol=1e-6)

    def test_goals_cannot_cross_invalid_transition_or_completion_hold(self):
        self.assertEqual(valid_prefix([True, True, False, True], {}), 2)
        self.assertEqual(valid_prefix([True] * 20, {'success_state': 8}), 8)
        self.assertEqual(valid_prefix([True] * 20, {'sim_error': 'solver diverged'}), 0)
        end = valid_prefix([True] * 7 + [False], {'sim_error': 'grasp tracking exceeded validity limit'})
        for t in range(end):
            future = future_indices(t, end - t)
            self.assertTrue(np.all(future > t))
            self.assertTrue(np.all(future <= end))

    def test_uncertain_material_correspondence_is_rejected(self):
        sections, landmarks = sleeve_and_arm()
        sections.fractions[2] = .35
        with self.assertRaisesRegex(ValueError, 'correspondence'):
            InteractionCoordinates(sections, landmarks)

    def test_budget_includes_setup_and_partial_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = Budget(Path(directory), 1, 8)
            for _ in range(6):
                budget.before_substep()
            with self.assertRaises(BudgetExhausted):
                budget.before_substep()
            self.assertEqual(budget.state()['charged_decisions'], 1)

    def test_branch_context_is_never_a_goal_or_action_training_target(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'branch.npz'
            descriptors = np.zeros((12, 4, 9)); descriptors[:3] = np.nan
            np.savez(file, features=np.ones((12, 50)), tools=np.zeros((12, 3)), rotation=np.eye(3),
                     descriptor=descriptors, base=np.zeros((12, 6)), actions=np.zeros((11, 6)),
                     landmarks=np.array([[0., 0., 0.], [.5, 0., 0.], [.7, .3, 0.]]))
            manifest = dict(horizon=1, validation_garment='old_holdout', episodes=[dict(
                path=str(file), valid_decisions=11, start_decision=3, garment='training_target',
                validation=True, counterfactual=True)])
            data = dataset(manifest, frames=4, seed=0, chunk=1)
            self.assertEqual(len(data['action']), 4)
            self.assertTrue(np.isfinite(data['geometry']).all())
            self.assertTrue(data['val'].all())
            self.assertTrue(data['counterfactual'].all())


if __name__ == '__main__':
    unittest.main()
