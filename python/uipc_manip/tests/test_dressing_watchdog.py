"""The dressing environment's decision watchdog, Newton cap and anchor tether (CPU)."""

from dataclasses import replace

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
    assert cfg.anchor_tether_m == 0.06


def test_the_watchdog_is_on_by_default_and_can_be_switched_off_per_world():
    # A trip raises for the whole world, so an evaluation world runs without it: one slow
    # configuration would otherwise end all of its episodes and void the round.
    assert DressingConfig().decision_watchdog is True
    assert replace(DressingConfig(), decision_watchdog=False).decision_watchdog is False


def test_tether_drops_a_move_that_opens_the_gap_past_it():
    held = np.zeros((2, 3))
    current = held + [0.015, 0.0, 0.0]
    assert tether_allows(held + [0.018, 0.0, 0.0], held, current, 0.02)
    assert not tether_allows(held + [0.025, 0.0, 0.0], held, current, 0.02)


def test_tether_lets_a_move_close_a_gap_already_past_it():
    held = np.zeros((2, 3))
    current = held + [0.05, 0.0, 0.0]
    assert tether_allows(held + [0.04, 0.0, 0.0], held, current, 0.02)
    assert not tether_allows(held + [0.06, 0.0, 0.0], held, current, 0.02)


def test_tether_bounds_the_farthest_vertex_not_the_patch_centre():
    held = np.array([[0.03, 0.0, 0.0], [-0.03, 0.0, 0.0]])
    quarter_turn = np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]])
    assert np.allclose(quarter_turn.mean(axis=0), held.mean(axis=0))
    assert not tether_allows(quarter_turn, held, held, 0.02)


def test_no_tether_allows_every_move():
    held = np.zeros((2, 3))
    assert tether_allows(held + [1.0, 0.0, 0.0], held, held, None)
