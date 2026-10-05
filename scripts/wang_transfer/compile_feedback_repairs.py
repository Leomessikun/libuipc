"""Compile a small bank of observable feedback repairs into a policy mixture.

CPU prototype of a proposed supervision operator, not a trained dressing actor.
Values must include complete task outcomes after prefix, observed response,
suffix and the fixed fallback continuation. Routers must use observations only.
Mixing happens over complete repairs, never coordinate-wise over their actions.
The robust-mixture linear program itself is established optimization, not new.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def compile_repairs(response_probability, branch_value, baseline_value):
    """Finite-model optimization; inputs are point estimates, not certificates.

    response_probability: [models, observable branches].
    branch_value: [models, branches, candidate closed-loop suffixes].
    baseline_value: [models], full fallback policy success from the same root.
    """
    q = np.asarray(response_probability, dtype=float)
    v = np.asarray(branch_value, dtype=float)
    base = np.asarray(baseline_value, dtype=float)
    if q.ndim != 2 or v.ndim != 3 or v.shape[:2] != q.shape or base.shape != (len(q),):
        raise ValueError("Expected q[M,B], V[M,B,A], and baseline[M]")
    if not all(np.isfinite(x).all() for x in (q, v, base)):
        raise ValueError("Nonfinite evidence")
    if np.any(q < 0) or not np.allclose(q.sum(axis=1), 1):
        raise ValueError("Observable-response probabilities must sum to one")
    if np.any((v < 0) | (v > 1)) or np.any((base < 0) | (base > 1)):
        raise ValueError("Values must be complete success probabilities")
    assignments = list(itertools.product(range(v.shape[2]), repeat=q.shape[1]))
    gains = np.stack([np.sum(q * v[:, np.arange(q.shape[1]), assignment], axis=1) - base
                      for assignment in assignments], axis=1)
    # Explicit frozen-policy fallback has zero gain in every source model.
    gains = np.column_stack([np.zeros(len(q)), gains])
    objective = np.r_[np.zeros(gains.shape[1]), -1.]
    result = linprog(objective, A_ub=np.column_stack([-gains, np.ones(len(q))]),
                     b_ub=np.zeros(len(q)),
                     A_eq=np.array([np.r_[np.ones(gains.shape[1]), 0.]]), b_eq=[1.],
                     bounds=[(0., 1.)] * gains.shape[1] + [(None, None)], method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    weights = result.x[:-1]
    weights[np.abs(weights) < 1e-10] = 0.
    return dict(
        repair_assignments=["frozen_policy_fallback", *map(list, assignments)],
        mixture_weights=weights.tolist(), model_gain=(gains @ weights).tolist(),
        worst_model_gain=float(np.min(gains @ weights)),
        best_single_repair_worst_gain=float(gains.min(axis=0).max()),
        objective="max_lambda min_model sum_repair lambda[repair] * full_outcome_gain[model,repair]",
        status="Proposal compiler only. Estimated-source robustness is not a real-transfer or novel-algorithm claim.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.input.read_text())
    result = compile_repairs(data["response_probability"], data["branch_value"], data["baseline_value"])
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
