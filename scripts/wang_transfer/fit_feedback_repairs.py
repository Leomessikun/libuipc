"""Source-only, replica-cross-fitted routing and complete-repair compilation.

This tiny pilot uses a decision stump as a diagnostic router. It is not an actor
update and cannot identify a real-target model. Empty conditional cells receive
zero success value, with their support saved explicitly.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from compile_feedback_repairs import compile_gain_bank
from feedback_repair_control import route


FEATURES = {"observable": "response_feature", "initial": "initial_feature", "privileged": "privileged_feature"}


def fit_router(rows, feature_key):
    rows = [r for r in rows if r["feedback_repair"].get(feature_key) is not None]
    if not rows:
        return dict(feature=None, threshold=0., left=0, right=0, missing_action=0, samples=0)
    x = np.asarray([r["feedback_repair"][feature_key] for r in rows])
    a = np.asarray([r["feedback_repair"]["controller"]["suffix"] for r in rows])
    y = np.asarray([r["accepted"] for r in rows], float)
    # Each suffix is allocated equally in every matched reset block. The factor
    # two is the known propensity correction, not a claim of exact-state pairing.
    scores = [float(np.sum(2 * y * (a == i))) for i in (0, 1)]
    best_action = int(np.argmax(scores))
    result = dict(feature=None, threshold=0., left=best_action, right=best_action,
                  missing_action=best_action, samples=len(rows))
    best = scores[best_action]
    for f in range(x.shape[1]):
        for threshold in np.unique(np.quantile(x[:, f], [.25, .5, .75])):
            left_mask = x[:, f] <= threshold
            if min(left_mask.sum(), (~left_mask).sum()) < 2:
                continue
            for left, right in ((0, 1), (1, 0)):
                action = np.where(left_mask, left, right)
                score = float(np.sum(2 * y * (a == action)))
                if score > best + 1e-8:
                    best = score
                    result.update(feature=f, threshold=float(threshold), left=left, right=right)
    result["training_ips_success"] = best / len(rows)
    return result


def compile_table(rows, conditions, baseline, feature_key):
    repairs, columns, routers, evidence = [None], [np.zeros(len(conditions))], [], []
    for prefix in range(3):
        pool = [r for r in rows if r["feedback_repair"]["controller"].get("prefix") == prefix]
        full_router = fit_router(pool, feature_key)
        routers.append(full_router)
        folds = {rep: fit_router([r for r in pool if r["repeat"] != rep], feature_key) for rep in (0, 1)}
        q, value, count = np.zeros((3, 3)), np.zeros((3, 3, 2)), np.zeros((3, 3, 2), int)
        sums = np.zeros_like(value)
        for row in pool:
            m = conditions.index(row["condition"])
            trace = row["feedback_repair"]
            # Third branch is termination before a suffix can execute; its
            # realized terminal outcome is shared by both suffix choices.
            b = 2 if not trace["branch_reached"] else route(folds[row["repeat"]], trace[feature_key])
            a = trace["controller"]["suffix"]
            q[m, b] += 1
            count[m, b, a] += 1
            sums[m, b, a] += float(row["accepted"])
        q /= q.sum(axis=1, keepdims=True)
        np.divide(sums, count, out=value, where=count > 0)
        for m in range(3):
            if count[m, 2].sum():
                value[m, 2, :] = sums[m, 2].sum() / count[m, 2].sum()
        for assignment in itertools.product((0, 1), repeat=2):
            if assignment[0] == assignment[1]:
                # A flat controller has directly measured complete outcomes;
                # use them rather than a noisier conditional reweighting.
                returns = np.array([np.mean([r["accepted"] for r in pool
                    if r["condition"] == condition and r["feedback_repair"]["controller"]["suffix"] == assignment[0]])
                    for condition in conditions])
            else:
                returns = np.sum(q[:, :2] * value[:, np.arange(2), assignment], axis=1) + q[:, 2] * value[:, 2, 0]
            repairs.append(dict(prefix=prefix, assignment=list(assignment)))
            columns.append(returns - baseline)
        evidence.append(dict(prefix=prefix, q=q.tolist(), conditional_value=value.tolist(),
                             count=count.tolist(), cross_fit_routers=folds,
                             unsupported_cells=int(np.count_nonzero(count[:, :2] == 0))))
    gains = np.stack(columns, axis=1)
    compiled = compile_gain_bank(gains)
    return dict(repairs=repairs, routers=routers, weights=compiled["mixture_weights"],
                compilation=compiled, evidence=evidence, gain_bank=gains.tolist())


def fit(out):
    files = sorted((Path(out) / "completed").glob("*.json"))
    rows = [r for p in files for r in json.loads(p.read_text())["rows"] if r["phase"] == "source"]
    if len(rows) != 84:
        raise ValueError(f"Need all 84 frozen source attempts, got {len(rows)}")
    conditions = ["nominal", "bending_x2", "density_x1p5"]
    baseline_rows = [r for r in rows if r["feedback_repair"]["controller"]["kind"] == "base"]
    baseline = np.array([np.mean([r["accepted"] for r in baseline_rows if r["condition"] == c]) for c in conditions])
    tables = {name: compile_table(rows, conditions, baseline, feature) for name, feature in FEATURES.items()}
    observed = tables["observable"]
    ids = [0] + [i for i, r in enumerate(observed["repairs"]) if r is not None and len(set(r["assignment"])) == 1]
    flat_gains = np.asarray(observed["gain_bank"])[:, ids]
    flat_fit = compile_gain_bank(flat_gains)
    tables["flat"] = dict(repairs=[observed["repairs"][i] for i in ids], weights=flat_fit["mixture_weights"],
                          compilation=flat_fit, routers=observed["routers"])
    # Stable, predeclared permutation of branch assignments between repair modes.
    candidates = observed["repairs"][1:]
    permutation = np.random.default_rng(2026100507).permutation(len(candidates))
    shuffled = {str(r["prefix"]) + ":" + str(r["assignment"]): candidates[j]["assignment"]
                for r, j in zip(candidates, permutation)}
    base_by_unit = {(r["body"], r["condition"], r["repeat"]): r["accepted"] for r in baseline_rows}
    repaired = [r for r in rows if r["feedback_repair"]["controller"]["kind"] == "fixed" and r["accepted"]
                and not base_by_unit[(r["body"], r["condition"], r["repeat"])]]
    new_replicas = sorted({r["repeat"] for r in repaired})
    advantage = observed["compilation"]["mean_model_gain"] - flat_fit["mean_model_gain"]
    # An unchanged/unsupported bank cannot justify 28 more target attempts.
    proceed = (new_replicas == [0, 1] and observed["compilation"]["worst_model_gain"] >= -1e-7
               and observed["compilation"]["mean_model_gain"] > 0 and advantage >= .1 - 1e-8)
    return dict(version=1, source_rows=len(rows), conditions=conditions, baseline=baseline.tolist(),
                tables=tables, shuffled_assignments=shuffled,
                source_gate=dict(proceed_to_target=bool(proceed), new_repair_successes=len(repaired),
                                 new_repair_replicas=new_replicas, observable_minus_flat_mean_gain=advantage,
                                 rule="new repairs in both replicas; nonnegative source worst-model gain; positive mean gain; >=.10 mean gain over robust flat mixture"),
                limitations=["Two bodies/two replicas are a rejection screen, not statistical validation.",
                             "Source LP uses replica-cross-fitted routes; the final router is refit on source only.",
                             "Matched reset blocks have nondeterministic settled states; no exact counterfactual claim.",
                             "Privileged-state router is an information ablation, not a guaranteed oracle upper bound.",
                             "No actor weights are updated by this diagnostic compilation."])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = fit(a.out)
    (a.out / "source_fit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["source_gate"], indent=2))
