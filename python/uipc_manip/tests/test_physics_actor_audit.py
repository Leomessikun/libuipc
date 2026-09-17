"""Action audit gates must use executed intervals and task progress, not critic value."""
import numpy as np
import pytest

from uipc_manip.physics_actor_audit import finite_action_differences, task_improvement


def test_finite_action_differences_use_the_clipped_interval_and_ignore_x_rotation():
    action = np.array([0.99, -0.98, 0.2, 0.7, 0.1, -0.3])
    weights = np.array([1, 2, 3, 4, 5, 6])
    calls = []
    def evaluate(a):
        calls.append(a.copy())
        return dict(value=float(a @ weights), task=float(2 * a @ weights))
    gradients, rows = finite_action_differences(action, 0.1, evaluate)
    expected = weights.copy()
    expected[3] = 0
    assert np.allclose(gradients["value"], expected)
    assert np.allclose(gradients["task"], 2 * expected)
    assert len(calls) == 10 and np.abs(calls).max() <= 1
    assert rows[0]["span"] == pytest.approx(0.11)
    assert all(a[3] == action[3] for a in calls)


def test_task_gate_rejects_grasp_loss_unconfirmed_gain_and_baseline_variation():
    def row(c, grasp=True):
        return dict(coverage=c, all_grasps_valid=grasp)
    base = [row(0.2), row(0.201)]
    assert task_improvement(base, [row(0.23), row(0.24)])["accepted"]
    assert not task_improvement(base, [row(0.23), row(0.24, False)])["accepted"]
    assert not task_improvement(base, [row(0.23), row(0.205)])["accepted"]
    assert not task_improvement([row(0.2), row(0.3)], [row(0.25), row(0.35)])["accepted"]
    assert not task_improvement(base, [row(float("nan")), row(0.24)])["accepted"]
