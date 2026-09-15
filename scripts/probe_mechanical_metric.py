"""Analytic actuator/contact-spring falsification probe, not an IPC benchmark.

Compare Euclidean, response, and mechanical-response trust metrics at matched
predicted reward gain. A stiff environment can have a *small* response pullback
metric: stiffness also suppresses D. No force-safety claim follows from PSD alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def trust_step(g, metric, budget):
    """Maximize g @ delta subject to delta @ metric @ delta / 2 <= budget."""
    if budget <= 0:
        raise ValueError("Trust budget must be positive")
    np.linalg.cholesky(metric)
    p = np.linalg.solve(metric, g)
    norm2 = float(g @ p)
    return np.zeros_like(g) if norm2 == 0 else p * np.sqrt(2 * budget / norm2)


def probe(stiffness=100., actuator_stiffness=1., damping=1e-3, budget=1e-4):
    # Coordinates: normal, tangent. E = x^T K x/2 + c ||x-a||^2/2.
    k, c = np.diag([stiffness, 1.]), actuator_stiffness
    d = np.linalg.solve(k + c*np.eye(2), c*np.eye(2))
    reduced = c*np.eye(2) - c*c*np.linalg.inv(k + c*np.eye(2))
    metrics = dict(euclidean=np.eye(2), response=d.T@d+damping*np.eye(2),
                   mechanical_response=d.T@k@d+damping*np.eye(2))
    g = d.T @ np.ones(2)  # reward J(x) = x_normal + x_tangent
    reference_gain = float(g @ trust_step(g, metrics["euclidean"], budget))
    rows = []
    for name, metric in metrics.items():
        step = trust_step(g, metric, budget)
        equal_gain_step = step * reference_gain / (g @ step)
        dx = d @ equal_gain_step
        force = k @ dx
        assert np.isclose(step @ metric @ step / 2, budget)
        assert np.isclose(g @ equal_gain_step, reference_gain)
        rows.append(dict(metric=name, eigenvalues=np.linalg.eigvalsh(metric).tolist(),
                         normal_to_tangent_metric=float(metric[0, 0]/metric[1, 1]),
                         trust_step=step.tolist(), equal_gain_step=equal_gain_step.tolist(),
                         equal_gain_displacement=dx.tolist(), normal_force_change=float(force[0]),
                         reduced_energy_change=float(equal_gain_step @ reduced @ equal_gain_step / 2)))
    # Check the actual reduced energy against the original energy at the optimized state.
    a = np.array([.2, .1]); x = d@a
    assert np.isclose(a@reduced@a/2, x@k@x/2 + c*np.sum((x-a)**2)/2)
    return dict(model="analytic two-axis spring, not IPC", stiffness=stiffness,
                actuator_stiffness=c, damping=damping, trust_budget=budget,
                action_gradient=g.tolist(), sensitivity=d.tolist(), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {"cases": [probe(stiffness=k) for k in (1., 10., 100., 1000.)],
              "interpretation": "Mechanical-response curvature is not a force constraint. Validate on IPC before using it for actor trust."}
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
