"""Decide how far to move a policy and how much to measure, in one optimisation.

The usual improvement step collects data, estimates action values, then moves the
policy, and hopes the estimate was good enough for the move it chose. This solves for
both at once: the probability change and the number of simulator evaluations that
change needs, under one budget.

Write ``p`` for the old probabilities over a finite action set at one state, ``q`` for
the new ones, ``mu_i`` for the expected return of taking action ``i`` and then following
the frozen old policy, ``sigma_i`` for the standard deviation of a *single* such
evaluation, ``c_i`` for what one costs and ``n_i`` for how many are bought. The local
improvement and its error are

    Delta(q)      = sum_i (q_i - p_i) mu_i
    Var[Delta(q)] = sum_i (q_i - p_i)^2 sigma_i^2 / n_i

the coefficient being ``q_i - p_i`` and not ``q_i``, because what has to be estimated
accurately is the *change*, not the policy's absolute value. Maximising

    Delta(q) - eta * KL(q || p) - beta * sqrt(Var[Delta(q)])   s.t. sum_i c_i n_i <= B

over ``q`` and ``n`` together has two consequences worth having.

First, the allocation that minimises the variance at fixed ``q`` is Neyman allocation
with the policy change as the stratum weight,

    n_i  proportional to  |q_i - p_i| * sigma_i / sqrt(c_i),

so evaluation demand is *the probability mass about to move* times the return's
volatility over the root of what a measurement costs.

Second, substituting it back turns the square root of the variance into a weighted L1
penalty on the change, with

    lambda_i = beta * sigma_i * sqrt(c_i / B),

which is a per-action threshold on the scale of a standard error rather than a hand-set
uncertainty bonus. The policy update is then a soft-thresholded exponential reweighting
that leaves an action alone when the evidence does not support moving it, and reduces to
the ordinary exponential update as the budget grows.

Two departures from the derivation as first written, each forced by a measurement on
this project's dressing task:

* **A floor on the allocation.** Neyman allocation gives zero budget to an action the
  plan does not move, so its estimates never improve and it can never be discovered to
  be worth moving. ``n_min`` keeps every candidate measured.
* **Independence is assumed and, here, earned.** The variance above drops the
  covariances between candidates. Where evaluations can share a random seed they do not
  vanish. On this simulator they may be kept: restoring is exact and carries the
  environment's own generator state, so the only remaining randomness is the solver's,
  which no caller controls, and the measured correlation between candidates across
  repeats is -0.04 to +0.04 on three different scores over 288 pairs.

Nothing here is a safety bound. ``beta`` is a regulariser: the variance is estimated
from few samples, the solver's own bias does not average away, and a small sample can
miss a rare failure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EXP_CLIP = 60.0
"""Largest exponent evaluated, so a wide bisection bracket cannot overflow."""


def shrink(x: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Soft threshold: the part of ``x`` that exceeds ``lam`` in magnitude."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def thresholds(sigma, cost, budget: float, beta: float) -> np.ndarray:
    """``lambda_i = beta sigma_i sqrt(c_i / B)``, the per-action dead zone."""
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if budget <= 0:
        raise ValueError("The evaluation budget must be positive")
    return float(beta) * sigma * np.sqrt(cost / float(budget))


def policy_from_nu(p: np.ndarray, mu: np.ndarray, lam: np.ndarray, eta: float,
                   nu: float) -> np.ndarray:
    return p * np.exp(np.clip(shrink(mu - nu, lam) / eta, -EXP_CLIP, EXP_CLIP))


def solve_policy(p, mu, lam, eta: float, tol: float = 1e-12, iterations: int = 200) -> np.ndarray:
    """The maximiser of ``Delta - eta KL - sum lambda_i |q_i - p_i|`` on the simplex.

    ``sum_i q_i(nu)`` is non-increasing in ``nu``, so one bisection finds the multiplier.
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    if p.shape != mu.shape or p.shape != lam.shape:
        raise ValueError("p, mu and lambda must have the same shape")
    if not np.isclose(p.sum(), 1.0) or (p < 0).any():
        raise ValueError("p must be a probability vector")
    if eta <= 0:
        raise ValueError("eta must be positive")
    span = float(np.abs(mu).max() + lam.max() + eta * EXP_CLIP + 1.0)
    lo, hi = float(mu.min()) - span, float(mu.max()) + span
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if policy_from_nu(p, mu, lam, eta, mid).sum() > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    q = policy_from_nu(p, mu, lam, eta, 0.5 * (lo + hi))
    total = q.sum()
    # Every action may sit inside its dead zone, in which case q is p already.
    return q / total if total > 0 else p.copy()


def allocate(weights, sigma, cost, budget: float, n_min: float = 0.0) -> np.ndarray:
    """Neyman allocation with the policy change as the stratum weight, plus a floor.

    ``weights`` is ``|q_i - p_i|``. The floor is paid first and the remainder allocated,
    so the returned plan satisfies ``sum_i c_i n_i <= budget`` whenever the floor does.
    """
    w = np.abs(np.asarray(weights, dtype=np.float64))
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    floor = float(n_min) * cost
    if floor.sum() > budget + 1e-12:
        raise ValueError("The floor alone exceeds the budget")
    remainder = float(budget) - floor.sum()
    scale = float((w * sigma * np.sqrt(cost)).sum())
    if scale <= 0 or remainder <= 0:
        return np.full(w.shape, float(n_min))
    return float(n_min) + remainder * w * sigma / (np.sqrt(cost) * scale)


def improvement_variance(weights, sigma, n) -> float:
    """``sum_i (q_i - p_i)^2 sigma_i^2 / n_i``; infinite where a weight is unmeasured."""
    w = np.asarray(weights, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    live = w != 0.0
    if np.any(n[live] <= 0):
        return float("inf")
    return float(((w[live] ** 2) * (sigma[live] ** 2) / n[live]).sum())


@dataclass
class Plan:
    q: np.ndarray
    """The probabilities the budget supports moving to."""
    n: np.ndarray
    """Evaluations per action, before rounding."""
    thresholds: np.ndarray
    nu: float
    variance: float
    """The improvement's estimated variance under this plan."""
    spent: float

    def rounded(self, n_min: int = 1) -> np.ndarray:
        return np.maximum(np.rint(self.n).astype(int), int(n_min))


def plan(p, mu, sigma, cost, budget: float, *, eta: float, beta: float,
         n_min: float = 1.0) -> Plan:
    """Solve for the policy change and the evaluation allocation together.

    The thresholds do not depend on ``q``, so the pair is solved in one pass rather than
    by alternation: the policy first, then the allocation its change implies. The floor
    is charged before the thresholds, so the dead zones reflect the budget actually
    available to the Neyman part.
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    floor_cost = float(n_min) * float(cost.sum())
    free = float(budget) - floor_cost
    if free <= 0:
        raise ValueError("The floor alone exhausts the budget; raise B or lower n_min")
    lam = thresholds(sigma, cost, free, beta)
    q = solve_policy(p, mu, lam, eta)
    n = allocate(q - p, sigma, cost, budget, n_min=n_min)
    return Plan(q=q, n=n, thresholds=lam, nu=float("nan"),
                variance=improvement_variance(q - p, sigma, n),
                spent=float((cost * n).sum()))


def update(p, mu, sigma, n_actual, *, eta: float, beta: float,
           iterations: int = 500, step: float = 0.5) -> np.ndarray:
    """The policy update against the data actually collected.

    The plan's closed form folds in the allocation it *asked* for. Once the evaluations
    are in hand the objective is the original one with ``n`` fixed to what was bought,

        max_q  sum_i (q_i - p_i) mu_i - eta KL(q || p) - beta sqrt(sum_i w_i^2 s_i^2/n_i)

    which is concave on the simplex, so mirror descent with an exponentiated step
    converges. Planned-but-uncollected evaluations are never treated as evidence.
    """
    p = np.asarray(p, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    n = np.asarray(n_actual, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("Every action needs at least one actual evaluation")
    v = (sigma ** 2) / n
    q = p.copy()
    for _ in range(iterations):
        w = q - p
        norm = np.sqrt(float((w ** 2 * v).sum()))
        # Gradient of the penalty, with the subgradient at the origin taken as zero.
        penalty = beta * (w * v) / norm if norm > 1e-15 else np.zeros_like(w)
        with np.errstate(divide="ignore"):
            grad = mu - eta * (np.log(np.maximum(q, 1e-300) / np.maximum(p, 1e-300)) + 1.0) - penalty
        q = q * np.exp(np.clip(step * (grad - float((q * grad).sum())) / max(eta, 1e-12),
                               -EXP_CLIP, EXP_CLIP))
        q = np.maximum(q, 0.0)
        q /= q.sum()
    return q


def objective(p, q, mu, sigma, n, *, eta: float, beta: float) -> float:
    """The quantity ``plan`` and ``update`` maximise, for checking either against it."""
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    w = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = float(np.sum(np.where(q > 0, q * np.log(np.maximum(q, 1e-300) / np.maximum(p, 1e-300)), 0.0)))
    return float((w * mu).sum() - eta * kl - beta * np.sqrt(improvement_variance(w, sigma, n)))
