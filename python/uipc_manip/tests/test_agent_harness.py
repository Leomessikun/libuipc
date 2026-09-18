"""The agent's tool surface: clipping to the controller, decision accounting and the call log."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from uipc_manip import agent_harness as ah
from uipc_manip.tests.test_decision_features import BUDGET, pack


class FakeEnv:
    """Minimal stand-in: records the actions it is stepped with and returns a fixed observation."""

    def __init__(self, max_translation=0.008, max_rotation=0.08, commanded=0.008):
        self.cfg = SimpleNamespace(max_translation=max_translation, max_rotation=max_rotation)
        self.spec = SimpleNamespace(point_budget=BUDGET)
        self.num_envs, self.action_dim = 1, 6
        self.commanded = commanded
        self.stepped: list[np.ndarray] = []
        self.cells = [SimpleNamespace(garment="tshirt_26", human=14046)]
        self._obs = np.stack([pack(cloth=[(0.0, 0.0, 0.02)], arm=[(0.1, 0.0, 0.0)])])

    def reset(self, seeds):
        self.stepped.clear()
        return self._obs

    def step(self, actions, reset_on_done=True):
        self.stepped.append(np.asarray(actions)[0].copy())
        info = dict(upperarm_ratio=0.4, forearm_ratio=1.0, tracking_error=0.001, grasp_valid=True,
                    commanded_translation_m=self.commanded, accepted_anchor_translation_m=self.commanded,
                    collision_rejected_substeps=0, tether_rejected_substeps=0)
        return self._obs, np.zeros(1), np.zeros(1, dtype=bool), [info]


def session(max_decisions=10, **kwargs):
    s = ah.DressingSession(FakeEnv(**kwargs), max_decisions=max_decisions)
    s.reset(7)
    return s


def test_a_command_is_scaled_into_normalized_action_units():
    s = session()
    s.move(dx=0.004, repeat=1)
    assert np.allclose(s.env.stepped[0][:3], [0.5, 0.0, 0.0])
    assert np.allclose(s.env.stepped[0][3:], 0.0)


def test_a_command_beyond_the_controller_limit_is_clipped_not_refused():
    s = session()
    out = s.move(dx=0.05, repeat=1)
    assert np.allclose(s.env.stepped[0][:3], [1.0, 0.0, 0.0])
    assert out["decisions_left"] == 9
    assert s.calls[-1]["arguments"]["clipped"] is True


def test_repeat_spends_one_decision_each_and_the_budget_is_enforced():
    s = session(max_decisions=3)
    s.move(dz=0.001, repeat=3)
    assert len(s.env.stepped) == 3 and s.decisions_left == 0
    with pytest.raises(ah.DecisionBudgetExhausted):
        s.move(dz=0.001)
    with pytest.raises(ValueError):
        s.move(dz=0.001, repeat=0)


def test_the_response_reports_how_far_the_garment_moved_per_commanded_metre():
    s = session(max_decisions=40, commanded=0.008)
    assert s.response() is None
    # The window needs one more centroid than commands, so it stays silent one decision longer.
    s.move(dx=0.008, repeat=8)
    assert s.response() is None
    s.move(dx=0.008, repeat=1)
    # The fake environment returns a fixed observation, so the garment never moves.
    assert s.response() == pytest.approx(0.0)


def test_observe_reports_only_deployment_side_quantities():
    s = session()
    out = s.observe()
    assert {"decisions_left", "tool_position", "goal_from_tool", "garment_points", "arm_points",
            "garment_response", "cloth_arm_min_gap"} <= set(out)
    assert not any(k in out for k in ("upperarm_ratio", "forearm_ratio", "tracking_error"))


def test_report_returns_the_task_metrics_and_is_logged_as_privileged():
    s = session()
    s.move(dx=0.001)
    row = s.report()
    assert row["upperarm_ratio"] == pytest.approx(0.4)
    assert s.calls[-1]["call"] == "report"


def test_run_policy_needs_a_policy():
    s = session()
    with pytest.raises(RuntimeError):
        s.run_policy(1)


def test_run_policy_spends_the_budget_and_marks_its_decisions():
    env = FakeEnv()
    policy = SimpleNamespace(act=lambda obs, deterministic=True: np.zeros((1, 6), dtype=np.float32))
    s = ah.DressingSession(env, max_decisions=5, agent=policy)
    s.reset(1)
    s.move(dx=0.002)
    s.run_policy(2)
    assert [row["source"] for row in s.trace] == ["agent", "policy", "policy"]
    assert s.decisions_left == 2


def test_every_call_is_logged_in_order():
    s = session()
    s.observe()
    s.move(dx=0.001)
    assert [c["call"] for c in s.calls] == ["reset", "observe", "observe", "observe", "move"]
    assert s.calls[0]["arguments"]["seed"] == 7


def test_render_needs_an_image_directory():
    s = session()
    with pytest.raises(RuntimeError):
        s.render()
