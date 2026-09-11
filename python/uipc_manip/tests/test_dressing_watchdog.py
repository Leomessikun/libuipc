"""The dressing environment's decision watchdog, Newton cap and anchor tether (CPU)."""

import numpy as np

from uipc_manip.dressing_env import DressingConfig, decision_time_limit, tether_allows


def test_budget_is_floor_times_factor_until_eight_decisions_are_timed():
    assert decision_time_limit([1.0] * 7, floor_s=30.0, factor=8.0) == 240.0


def test_budget_follows_the_median_and_never_drops_below_the_floor():
    assert decision_time_limit([5.0] * 8 + [100.0], floor_s=30.0, factor=8.0) == 40.0
    assert decision_time_limit([1.0] * 20, floor_s=30.0, factor=8.0) == 30.0


def test_defaults_cap_newton_far_above_a_normal_step():
    cfg = DressingConfig()
    assert cfg.newton_max_iterations == 128
    assert (cfg.decision_time_floor_s, cfg.decision_time_factor) == (30.0, 8.0)


def test_tether_drops_a_move_that_opens_the_gap_past_it():
    held, anchor = np.zeros(3), np.array([0.015, 0.0, 0.0])
    assert tether_allows(np.array([0.018, 0.0, 0.0]), held, anchor, 0.02)
    assert not tether_allows(np.array([0.025, 0.0, 0.0]), held, anchor, 0.02)


def test_tether_lets_a_move_close_a_gap_already_past_it():
    held, anchor = np.zeros(3), np.array([0.05, 0.0, 0.0])
    assert tether_allows(np.array([0.04, 0.0, 0.0]), held, anchor, 0.02)
    assert not tether_allows(np.array([0.06, 0.0, 0.0]), held, anchor, 0.02)


def test_no_tether_allows_every_move():
    assert tether_allows(np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), None)
