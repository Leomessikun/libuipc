"""The scripted expert's last stage, which ends on the progress reading (CPU)."""

import types

import numpy as np

from uipc_manip.dressing_heuristic import HeuristicDressingPolicy


class _Cell:
    """A straight arm along +x, with the four garment vertices the expert reads."""

    finger = np.array([0.0, 0.0, 0.0])
    elbow = np.array([0.3, 0.0, 0.0])
    shoulder = np.array([0.6, 0.0, 0.0])
    opening_idx = np.array([0, 1])
    alignment_idx = (2, 3)
    arm_points = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.6, 0.0, 0.0]])


class _Env:
    action_dim = 6
    num_envs = 1

    def __init__(self, ratio: float = 0.0) -> None:
        self.cells = [_Cell()]
        self.cfg = types.SimpleNamespace(max_translation=0.00866, max_rotation=0.0873)
        self._anchor = np.array([[0.55, 0.0, 0.10]])
        self.ratio = float(ratio)

    def positions(self):
        return [np.array([[0.55, 0.01, 0.0], [0.55, -0.01, 0.0], [0.55, 0.02, 0.0], [0.55, -0.02, 0.0]])]


class _ProgressEnv(_Env):
    """An environment that reports the reward's reading, as the live one does."""

    def progress(self):
        return [types.SimpleNamespace(upperarm_ratio=self.ratio)]


def _drive(ratios, stage: int = 6):
    """Stage name and commanded magnitude after each decision, from ``stage``."""
    env = _ProgressEnv()
    policy = HeuristicDressingPolicy(env)
    policy.stage[0] = stage
    out = []
    for ratio in ratios:
        env.ratio = float(ratio)
        action = policy.actions()
        out.append((policy.stage_names()[0], float(np.abs(action).sum())))
    return out


def test_a_dressed_reading_ends_the_last_stage():
    steps = _drive([0.5, 0.96, 0.96])
    assert steps[0][0] == "last" and steps[0][1] > 0.0
    assert steps[1] == ("done", 0.0)
    assert steps[2] == ("done", 0.0)


def test_a_reading_that_falls_back_ends_it_too():
    # The opening pushed past the shoulder leaves the metric's view: Wang's upper-arm ray keeps only
    # hits in front of the shoulder, so the reading and the reward collapse together.
    assert [s[0] for s in _drive([0.60, 0.62, 0.0])] == ["last", "last", "done"]


def test_a_small_dip_does_not_end_it():
    assert [s[0] for s in _drive([0.60, 0.58, 0.59])] == ["last", "last", "last"]


def test_earlier_stages_ignore_the_reading():
    assert _drive([0.99, 0.99], stage=5)[0][0] in ("elbow_hook", "last")


def test_an_environment_without_progress_keeps_the_old_behaviour():
    env = _Env()
    assert not hasattr(env, "progress")
    policy = HeuristicDressingPolicy(env)
    policy.stage[0] = 6
    action = policy.actions()
    assert policy.stage_names()[0] == "last" and float(np.abs(action).sum()) > 0.0
