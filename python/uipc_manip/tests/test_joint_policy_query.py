"""Checks on the joint policy-improvement and query-allocation operator."""
from __future__ import annotations

import numpy as np
import pytest

from uipc_manip import joint_policy_query as jpq


def exponential_update(p, mu, eta):
    q = np.asarray(p, dtype=float) * np.exp(np.asarray(mu, dtype=float) / eta)
    return q / q.sum()


def test_worked_example_matches_the_derivation():
    """The three-action example the design was written around."""
    p = np.array([0.40, 0.30, 0.30])
    mu = np.array([0.50, 0.62, 0.70])
    sigma = np.array([0.02, 0.03, 0.25])
    cost = np.array([1.0, 1.0, 4.0])
    lam = jpq.thresholds(sigma, cost, budget=100.0, beta=2.0)
    assert lam == pytest.approx([0.004, 0.006, 0.10], rel=1e-12)
    q = jpq.solve_policy(p, mu, lam, eta=0.1)
    # nu solves to 0.57746; the design's table reports 0.192, 0.432, 0.376.
    assert q == pytest.approx([0.192, 0.432, 0.376], abs=1e-3)
    # The third action still rises, but the highest mean alone does not hand it the mass
    # that an uncorrected exponential update would.
    plain = exponential_update(p, mu, 0.1)
    assert plain == pytest.approx([0.1107, 0.2757, 0.6136], abs=5e-4)
    assert q[2] < plain[2] and q[2] > p[2]


def test_no_error_penalty_is_the_ordinary_exponential_update():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(5))
        mu = rng.normal(size=5)
        lam = jpq.thresholds(rng.uniform(0.1, 1.0, 5), rng.uniform(1, 4, 5), 50.0, beta=0.0)
        assert lam == pytest.approx(np.zeros(5))
        assert jpq.solve_policy(p, mu, lam, eta=0.3) == pytest.approx(exponential_update(p, mu, 0.3), abs=1e-9)


def test_a_growing_budget_recovers_the_ordinary_update():
    p = np.array([0.4, 0.3, 0.3]); mu = np.array([0.5, 0.62, 0.70])
    sigma = np.array([0.02, 0.03, 0.25]); cost = np.array([1.0, 1.0, 4.0])
    # The thresholds fall as 1/sqrt(B), so the residual gap is of their own order.
    for budget, tol in ((1e6, 2e-2), (1e9, 1e-3), (1e12, 1e-4)):
        far = jpq.solve_policy(p, mu, jpq.thresholds(sigma, cost, budget, 2.0), eta=0.1)
        assert far == pytest.approx(exponential_update(p, mu, 0.1), abs=tol)


def test_an_action_inside_its_dead_zone_does_not_move():
    p = np.array([0.5, 0.25, 0.25])
    mu = np.array([0.0, 0.0, 0.0])
    lam = np.array([1.0, 1.0, 1.0])
    # Every difference is inside every threshold, so nothing may move at all.
    assert jpq.solve_policy(p, mu, lam, eta=0.1) == pytest.approx(p)


def test_allocation_attains_the_cauchy_schwarz_bound():
    rng = np.random.default_rng(3)
    for _ in range(30):
        w = rng.normal(size=6)
        w -= w.mean()  # a policy change sums to zero
        sigma = rng.uniform(0.05, 1.5, 6)
        cost = rng.uniform(0.5, 5.0, 6)
        budget = 500.0
        n = jpq.allocate(w, sigma, cost, budget, n_min=0.0)
        assert float((cost * n).sum()) == pytest.approx(budget)
        s = float((np.abs(w) * sigma * np.sqrt(cost)).sum())
        assert jpq.improvement_variance(w, sigma, n) == pytest.approx(s ** 2 / budget, rel=1e-9)
        # No other feasible allocation does better.
        for _ in range(20):
            other = n * rng.uniform(0.3, 3.0, 6)
            other *= budget / float((cost * other).sum())
            assert jpq.improvement_variance(w, sigma, other) >= jpq.improvement_variance(w, sigma, n) - 1e-12


def test_allocation_is_evaluation_demand_proportional_to_change_times_volatility():
    w = np.array([0.5, 0.5, 0.0])
    sigma = np.array([1.0, 2.0, 9.0])
    cost = np.array([1.0, 1.0, 1.0])
    n = jpq.allocate(w, sigma, cost, budget=300.0, n_min=0.0)
    assert n[1] == pytest.approx(2.0 * n[0])
    # An action the plan does not move earns no budget from the Neyman term, whatever
    # its volatility; that is exactly why a floor is needed.
    assert n[2] == pytest.approx(0.0)


def test_the_floor_keeps_every_candidate_measured_and_stays_inside_the_budget():
    w = np.array([0.5, 0.5, 0.0])
    sigma = np.array([1.0, 2.0, 9.0]); cost = np.array([1.0, 1.0, 2.0])
    n = jpq.allocate(w, sigma, cost, budget=300.0, n_min=4.0)
    assert n.min() >= 4.0 - 1e-12
    assert n[2] == pytest.approx(4.0)
    assert float((cost * n).sum()) <= 300.0 + 1e-9


def test_the_floor_cannot_exceed_the_budget():
    with pytest.raises(ValueError):
        jpq.allocate(np.ones(3), np.ones(3), np.ones(3), budget=2.0, n_min=1.0)


def test_plan_respects_the_budget_and_reports_its_own_variance():
    p = np.array([0.25] * 4)
    mu = np.array([0.1, 0.4, 0.35, -0.2])
    sigma = np.array([0.1, 0.3, 0.2, 0.15]); cost = np.array([2.0, 2.0, 2.0, 2.0])
    out = jpq.plan(p, mu, sigma, cost, budget=200.0, eta=0.2, beta=1.0, n_min=2.0)
    assert out.spent <= 200.0 + 1e-9
    assert out.n.min() >= 2.0 - 1e-12
    assert out.q.sum() == pytest.approx(1.0)
    assert out.variance == pytest.approx(jpq.improvement_variance(out.q - p, sigma, out.n))


def test_the_planned_policy_beats_staying_put_on_the_planned_objective():
    rng = np.random.default_rng(11)
    for _ in range(20):
        k = int(rng.integers(3, 7))
        p = rng.dirichlet(np.ones(k))
        mu = rng.normal(scale=0.5, size=k)
        sigma = rng.uniform(0.05, 0.6, k); cost = rng.uniform(1.0, 4.0, k)
        out = jpq.plan(p, mu, sigma, cost, budget=400.0, eta=0.25, beta=1.5, n_min=1.0)
        here = jpq.objective(p, out.q, mu, sigma, out.n, eta=0.25, beta=1.5)
        stay = jpq.objective(p, p, mu, sigma, out.n, eta=0.25, beta=1.5)
        assert here >= stay - 1e-9


def test_the_collected_data_update_agrees_with_the_plan_it_was_allocated_for():
    """The two stages must not be mixed, but they must meet where the plan is optimal.

    At the planned policy the square-root penalty's gradient equals the L1 thresholds
    exactly, so optimising the post-collection objective with the planned allocation
    must return the planned policy.
    """
    p = np.array([0.4, 0.3, 0.3]); mu = np.array([0.5, 0.62, 0.70])
    sigma = np.array([0.02, 0.03, 0.25]); cost = np.array([1.0, 1.0, 4.0])
    out = jpq.plan(p, mu, sigma, cost, budget=100.0, eta=0.1, beta=2.0, n_min=0.5)
    again = jpq.update(p, mu, sigma, out.n, eta=0.1, beta=2.0)
    assert again == pytest.approx(out.q, abs=2e-3)


def test_the_update_refuses_evaluations_that_were_never_collected():
    with pytest.raises(ValueError):
        jpq.update(np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.array([1.0, 1.0]),
                   np.array([3.0, 0.0]), eta=0.1, beta=1.0)


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        jpq.solve_policy(np.array([0.5, 0.4]), np.zeros(2), np.zeros(2), eta=0.1)
    with pytest.raises(ValueError):
        jpq.solve_policy(np.array([0.5, 0.5]), np.zeros(2), np.zeros(2), eta=0.0)
    with pytest.raises(ValueError):
        jpq.thresholds(np.ones(2), np.ones(2), budget=0.0, beta=1.0)
