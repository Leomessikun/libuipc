"""Is the better recovery identifiable from what the robot can see?

Reads collected counterfactual branches and asks, with one model class and one
cross-validation, how well the best macro at a state can be predicted from

* ``observation``: the deployed actor's own input at that decision,
* ``history``: that plus the previous decisions' observations and commands,
* ``privileged``: the simulator's low-dimensional state,
* ``constant``: no state at all, that is always the globally best macro.

Leave-one-state-out; every fold refits. The metrics are top-1 agreement with the
branch labels, pairwise ranking accuracy, and the return given up against a
per-state oracle. With tens of states this is a screen, not an estimate of a
deployable predictor's accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import decision_branches as db  # noqa: E402
from uipc_manip import decision_features as df  # noqa: E402


def load(run_dirs: list[Path], key: str):
    states, features, returns = [], {"observation": [], "history": [], "privileged": []}, []
    for run in run_dirs:
        result = json.load(open(run / "result.json"))
        arrays = np.load(run / "states.npz")
        budget = int(result["env"]["point_budget"])
        max_translation = float(result["env"]["max_translation"])
        by_state: dict[str, list[dict]] = {}
        for record in result["records"]:
            by_state.setdefault(record["state"], []).append(record)
        for state, records in sorted(by_state.items()):
            grouped = db.returns_by_macro(records, key)
            info = next(s for s in result["states"] if s["state"] == state)
            obs = arrays[f"{state}__observation"]
            past_obs = arrays[f"{state}__history_observations"]
            past_actions = arrays[f"{state}__history_actions"]
            features["observation"].append(df.observation_features(obs, budget))
            features["history"].append(df.history_features(np.concatenate([past_obs, obs[None]]), past_actions,
                                                           budget, max_translation))
            features["privileged"].append(df.privileged_features(arrays[f"{state}__privileged"]))
            states.append(dict(run=run.name, state=state, step=info["step"],
                               upperarm_ratio=info["upperarm_ratio"], macros=sorted(grouped)))
            returns.append(np.asarray([grouped[m].mean() for m in sorted(grouped)]))
    macros = states[0]["macros"]
    if any(s["macros"] != macros for s in states):
        raise ValueError("Runs disagree on the macro set")
    return states, {k: np.stack(v) for k, v in features.items()}, np.stack(returns), macros


def evaluate(x: np.ndarray | None, y: np.ndarray, alpha: float) -> dict:
    """Leave-one-state-out prediction of every macro's return; ``x`` None means constant."""
    n, m = y.shape
    chosen, predicted = np.zeros(n, dtype=int), np.zeros_like(y)
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        if x is None:
            predicted[i] = y[train].mean(axis=0)
        else:
            xtr, xte = df.standardize(x[train], x[i:i + 1])
            for j in range(m):
                predicted[i, j] = df.ridge_predict(df.ridge_fit(xtr, y[train, j], alpha), xte)[0]
        chosen[i] = int(np.argmax(predicted[i]))
    truth = np.argmax(y, axis=1)
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    ordered = np.mean([[np.sign(y[i, a] - y[i, b]) == np.sign(predicted[i, a] - predicted[i, b]) for a, b in pairs]
                       for i in range(n)])
    regret = y[np.arange(n), truth] - y[np.arange(n), chosen]
    return dict(top1=float(np.mean(chosen == truth)), pairwise=float(ordered), mean_regret=float(regret.mean()),
                median_regret=float(np.median(regret)), chosen=chosen.tolist(), truth=truth.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--key", default="sustained_coverage")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--min-consequence", type=float, default=0.0,
                   help="Keep only states where the best macro beats the policy by at least this much")
    args = p.parse_args()
    states, features, returns, macros = load(args.run_dirs, args.key)
    reference = macros.index("policy")
    gain = returns.max(axis=1) - returns[:, reference]
    keep = gain >= args.min_consequence
    print(f"states {len(states)} ({int(keep.sum())} kept at consequence >= {args.min_consequence}), macros {macros}")
    print(f"mean return by macro: " + ", ".join(f"{m} {v:.3f}" for m, v in
                                                sorted(zip(macros, returns[keep].mean(axis=0)), key=lambda kv: -kv[1])))
    rows = {}
    for name, x in (("constant", None), ("observation", features["observation"]),
                    ("history", features["history"]), ("privileged", features["privileged"])):
        rows[name] = evaluate(None if x is None else x[keep], returns[keep], args.alpha)
        rows[name]["features"] = 0 if x is None else int(x.shape[1])
    print(f"\n{'information set':14s} {'features':>8s} {'top-1':>6s} {'pairwise':>8s} {'mean regret':>11s} {'median regret':>13s}")
    for name, r in rows.items():
        print(f"{name:14s} {r['features']:8d} {r['top1']:6.2f} {r['pairwise']:8.2f} {r['mean_regret']:11.3f} {r['median_regret']:13.3f}")
    out = args.run_dirs[0].parent / "identifiability.json"
    out.write_text(json.dumps(dict(key=args.key, alpha=args.alpha, min_consequence=args.min_consequence,
                                   macros=macros, states=states, returns=returns.tolist(),
                                   kept=keep.tolist(), results=rows), indent=1) + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
