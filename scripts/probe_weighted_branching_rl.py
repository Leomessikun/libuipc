#!/usr/bin/env python3
"""CPU mechanism check for probability-corrected branching policy gradients.

This is a finite stochastic test, not a dressing environment or a new-algorithm
claim. Importance resampling and likelihood-ratio gradients are established.
The priority functions below are exact oracles; no priority model is trained.
No SAC, physics simulator, teacher, or action-value derivative is used.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np


def parent_probabilities(weights, priority, uniform_fraction=0.05):
    """Positive proposal probabilities, normalized within each independent tree."""
    unnormalized = weights * priority
    normalizer = unnormalized.sum(axis=-1, keepdims=True)
    base = np.divide(
        unnormalized, normalizer,
        out=np.full_like(unnormalized, 1 / weights.shape[-1]),
        where=normalizer > 0,
    )
    return (1 - uniform_fraction) * base + uniform_fraction / weights.shape[-1]


def corrected_child_weights(weights, probabilities, parents):
    """For N draws from q, each child of i carries w_i / (N q_i).

    The total mass is preserved in expectation, not necessarily in each draw.
    Self-normalizing these weights would change the estimator.
    """
    return np.take_along_axis(weights / probabilities, parents, axis=-1) / parents.shape[-1]


def check_resampling_identity():
    """Enumerate all two-child outcomes, including duplicate ancestry."""
    weights = np.array([[0.2, 0.3, 0.5]])
    probabilities = np.array([[0.6, 0.3, 0.1]])
    terminal_gradient = np.array([[2.0, -1.0], [-3.0, 4.0], [0.7, -2.0]])
    expectation = np.zeros(2)
    normalized_expectation = np.zeros(2)
    mass = 0.0
    for ancestry in product(range(3), repeat=2):
        parents = np.array([ancestry])
        event_probability = probabilities[0, list(ancestry)].prod()
        child_weights = corrected_child_weights(weights, probabilities, parents)[0]
        estimate = (child_weights[:, None] * terminal_gradient[list(ancestry)]).sum(axis=0)
        expectation += event_probability * estimate
        normalized_expectation += event_probability * estimate / child_weights.sum()
        mass += event_probability * child_weights.sum()
    expected = weights[0] @ terminal_gradient
    np.testing.assert_allclose(expectation, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(mass, 1.0, atol=1e-14, rtol=0)
    assert np.linalg.norm(normalized_expectation - expected) > 0.1
    return {
        "exact_expected_gradient": expected.tolist(),
        "corrected_expectation": expectation.tolist(),
        "incorrect_self_normalized_expectation": normalized_expectation.tolist(),
    }


def run_estimator(mode, batches, particles, stages, seed):
    """A sequence of stochastic gates; every intermediate reward is zero.

    At gate t, action 1 passes with probability .75, action 0 with .25.
    Failure is absorbing. Policy parameters are separate Bernoulli logits.
    Success means passing every gate. Fixed event boundaries every two gates
    are supplied, not learned. Dead paths still consume counted decision slots.
    """
    rng = np.random.default_rng(seed)
    theta = np.linspace(-1.0, 1.0, stages)
    policy_probability = 1 / (1 + np.exp(-theta))
    pass_probability = 0.25 + 0.5 * policy_probability
    exact_success = pass_probability.prod()
    exact_gradient = (
        exact_success * 0.5 * policy_probability * (1 - policy_probability) / pass_probability
    )
    # E[(a-p)^2 | gate passed], for the oracle second-moment priority.
    conditional_score_second_moment = (
        policy_probability * 0.75 * (1 - policy_probability) ** 2
        + (1 - policy_probability) * 0.25 * policy_probability ** 2
    ) / pass_probability
    alive = np.ones((batches, particles), dtype=bool)
    weights = np.full((batches, particles), 1 / particles)
    scores = np.zeros((batches, particles, stages))
    rows = np.arange(batches)[:, None]
    resamples = 0
    start = time.monotonic()
    for t in range(stages):
        action = rng.random((batches, particles)) < policy_probability[t]
        scores[:, :, t] = alive * (action - policy_probability[t])
        passed = rng.random((batches, particles)) < (0.25 + 0.5 * action)
        alive &= passed
        if mode == "independent" or t == stages - 1 or (t + 1) % 2:
            continue
        remaining_success = pass_probability[t + 1:].prod()
        committor = alive * remaining_success
        if mode == "credit_oracle":
            future_score_norm = conditional_score_second_moment[t + 1:].sum()
            # Disjoint parameters across gates make the cross term exactly zero.
            second_moment = committor * ((scores ** 2).sum(axis=-1) + future_score_norm)
            priority = np.sqrt(second_moment)
        else:
            priority = committor
        probabilities = parent_probabilities(weights, priority)
        cdf = probabilities.cumsum(axis=-1)
        cdf[:, -1] = 1.0
        uniforms = rng.random((batches, particles))
        parents = (uniforms[:, :, None] > cdf[:, None, :]).sum(axis=-1)
        if mode == "uncorrected":
            weights = np.full_like(weights, 1 / particles)
        else:
            weights = corrected_child_weights(weights, probabilities, parents)
        alive = alive[rows, parents]
        scores = scores[rows, parents]
        resamples += 1
    terminal_weights = weights * alive
    estimates = (terminal_weights[:, :, None] * scores).sum(axis=1)
    success_estimates = terminal_weights.sum(axis=1)
    mean = estimates.mean(axis=0)
    variance = estimates.var(axis=0, ddof=1)
    standard_error = np.sqrt(variance / batches)
    return {
        "mode": mode,
        "exact_success": float(exact_success),
        "mean_success_estimate": float(success_estimates.mean()),
        "success_standard_error": float(success_estimates.std(ddof=1) / np.sqrt(batches)),
        "exact_gradient": exact_gradient.tolist(),
        "mean_gradient": mean.tolist(),
        "gradient_standard_error": standard_error.tolist(),
        "max_absolute_gradient_z_score": float(np.max(np.abs(mean - exact_gradient) / standard_error)),
        "gradient_variance_trace": float(variance.sum()),
        "gradient_mse": float(np.mean(np.sum((estimates - exact_gradient) ** 2, axis=-1))),
        "trees_with_success_fraction": float(alive.any(axis=1).mean()),
        "mean_final_mass": float(weights.sum(axis=1).mean()),
        "final_mass_standard_error": float(weights.sum(axis=1).std(ddof=1) / np.sqrt(batches)),
        "physical_decision_slots_per_tree": particles * stages,
        "resampling_rounds_per_tree": resamples,
        "elapsed_seconds": time.monotonic() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10000)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--stages", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    if args.batches < 2 or args.particles < 2 or args.stages < 2:
        parser.error("batches, particles and stages must all be at least two")
    if args.out.exists():
        parser.error("output already exists; choose a fresh file")
    identity = check_resampling_identity()
    results = [
        run_estimator(mode, args.batches, args.particles, args.stages, args.seed + i)
        for i, mode in enumerate(("independent", "committor_oracle", "credit_oracle", "uncorrected"))
    ]
    report = {
        "scope": "Finite stochastic mechanism check. Oracle priorities; no learned policy or dressing claim.",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "enumerated_resampling_check": identity,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for result in results:
        print(json.dumps({key: result[key] for key in (
            "mode", "mean_success_estimate", "gradient_variance_trace",
            "max_absolute_gradient_z_score", "trees_with_success_fraction", "elapsed_seconds",
        )}))


if __name__ == "__main__":
    main()
