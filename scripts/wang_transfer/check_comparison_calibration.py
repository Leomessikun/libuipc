"""CPU-only falsification screen for comparison calibration, using completed Q4.

This checks ranking among existing actors; it is not actor post-training, action
chunk supervision, target physics calibration, or independent real-world data.
All choices below are fixed before running this screen.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.optimize import minimize

from run_sim2real_audit import DEFAULT_OUT, write_json


def ternary_offset(prior, labels):
    # Three logits with a unit quadratic penalty; no validation-set tuning.
    def loss(beta):
        logits = np.log(prior) + beta
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        nll = -np.log(probs[np.arange(len(labels)), labels]).sum()
        gradient = probs.copy()
        gradient[np.arange(len(labels)), labels] -= 1
        return nll + .5 * np.dot(beta, beta), gradient.sum(axis=0) + beta
    return minimize(loss, np.zeros(3), jac=True, method="BFGS").x


def main():
    start = time.monotonic()
    data = json.loads((DEFAULT_OUT / "matched_reset_summary.json").read_text())
    if data["validation_errors"]:
        raise RuntimeError("Unmatched reset configuration; do not calibrate")
    outcomes = {(r["condition"], r["policy"], r["body"]): int(r["accepted"]) for r in data["outcomes"]}
    conditions = [c for c in data["conditions"] if c != "nominal_repeat"]
    targets = [c for c in conditions if c != "nominal"]
    bodies = sorted({b for c, p, b in outcomes})
    train, test = bodies[:3], bodies[3:]
    policies = ["r1", "flow"]
    methods = {m: [] for m in ("nominal_selection", "ensemble_mean", "worst_model_gain",
                               "scalar_bias", "scalar_ridge", "ternary_offset")}
    folds = []
    for target in targets:
        sources = [c for c in conditions if c != target]
        prior, signed = {}, {}
        for b in bodies:
            for p in policies:
                values = [outcomes[c, p, b] - outcomes[c, "fmvp_sim", b] for c in sources]
                counts = np.bincount(np.asarray(values) + 1, minlength=3)
                prior[b, p] = (counts + .5) / (len(values) + 1.5)
                signed[b, p] = values
        x = np.asarray([prior[b, p] for b in train for p in policies])
        y = np.asarray([outcomes[target, p, b] - outcomes[target, "fmvp_sim", b] for b in train for p in policies])
        beta = ternary_offset(x, y + 1)
        gain = x[:, 2] - x[:, 0]
        # Additive correction shrunk toward zero; same six signed labels.
        bias = (y - gain).sum() / (len(y) + 1)
        features = np.stack([np.ones(len(gain)), gain], axis=1)
        # Ridge correction of source mean, not an absolute target-score learner.
        ridge = np.linalg.solve(features.T @ features + np.eye(2), features.T @ (y - gain))
        fold = dict(target=target, calibration_pairs=len(y), held_out_bodies=test,
                    ternary_logit_offset=beta.tolist(), scalar_bias=float(bias), methods={})
        fold_predictions = {m: [] for m in methods}
        for b in test:
            scores = {m: [] for m in methods}
            for p in policies:
                probs = prior[b, p]
                mean = float(probs[2] - probs[0])
                shifted = np.exp(np.log(probs) + beta - (np.log(probs) + beta).max())
                shifted /= shifted.sum()
                scores["nominal_selection"].append(outcomes["nominal", p, b] - outcomes["nominal", "fmvp_sim", b])
                scores["ensemble_mean"].append(mean)
                scores["worst_model_gain"].append(min(signed[b, p]))
                scores["scalar_bias"].append(mean + bias)
                scores["scalar_ridge"].append(mean + float(np.array([1., mean]) @ ridge))
                scores["ternary_offset"].append(float(shifted[2] - shifted[0]))
            truth = np.array([outcomes[target, p, b] - outcomes[target, "fmvp_sim", b] for p in policies])
            base = outcomes[target, "fmvp_sim", b]
            for m, values in scores.items():
                # Strictly positive gain; ties fall back to the base.
                i = int(np.argmax([0., *values]))
                selected = "fmvp_sim" if i == 0 else policies[i - 1]
                actual = outcomes[target, selected, b]
                row = dict(target=target, body=b, policy=selected, fallback=(i == 0),
                           actual_success=actual, base_success=base, gain=actual - base,
                           harmful=actual < base, regret=max(base, *(base + truth)) - actual,
                           gain_prediction_mse=float(np.mean((np.asarray(values) - truth) ** 2)))
                methods[m].append(row)
                fold_predictions[m].append(row)
        for m, rows in fold_predictions.items():
            fold["methods"][m] = dict(gain=sum(r["gain"] for r in rows), harmful=sum(r["harmful"] for r in rows),
                                       selected_updates=sum(not r["fallback"] for r in rows),
                                       gain_mse=float(np.mean([r["gain_prediction_mse"] for r in rows])))
        folds.append(fold)
    summaries = {}
    for m, rows in methods.items():
        updates = sum(not r["fallback"] for r in rows)
        summaries[m] = dict(decisions=len(rows), selected_updates=updates,
            harmful_updates=sum(r["harmful"] for r in rows),
            harmful_fraction_of_updates=sum(r["harmful"] for r in rows) / updates if updates else None,
            success_count=sum(r["actual_success"] for r in rows),
            baseline_success_count=sum(r["base_success"] for r in rows),
            net_gain=sum(r["gain"] for r in rows), regret=sum(r["regret"] for r in rows),
            gain_mse=float(np.mean([r["gain_prediction_mse"] for r in rows])))
    result = dict(
        method="Leave-one-stress-condition-out; six paired target labels on first three bodies; choose among base/r1/flow on last four.",
        summaries=summaries, folds=folds, selected_outcomes=methods, cpu_wall_seconds=time.monotonic() - start,
        limits=["Secondary matched-reset data; primary exact-state endpoint failed.",
                "Eight conditions reuse only four held-out bodies; 32 decisions are not 32 independent targets.",
                "No history features, candidate chunks, distillation, real trials, or confidence guarantee.",
                "Nominal r1 improvement absent; this only rejects/promotes a small calibration component.",
                "Penalty constants fixed to one; no target-test tuning. No statistical discovery claim."])
    write_json(DEFAULT_OUT / "comparison_calibration_screen.json", result)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
