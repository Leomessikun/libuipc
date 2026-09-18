"""Consequence, reliability and state-dependence of the collected recovery branches.

Three questions, in the order that decides whether a learned recovery decision is
worth building at all:

1. Consequence: at a state the policy visits, does any macro beat the policy's own
   continuation by more than the spread of a macro's own repeats?
2. Reliability: do independent repeats rank the macros the same way?
3. State dependence: is the winning macro the same everywhere? If one macro wins at
   every state, a fixed recovery suffices and there is nothing to learn from the
   observation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import decision_branches as db  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--key", default="sustained_coverage")
    p.add_argument("--reference", default="policy")
    args = p.parse_args()
    summary = dict(key=args.key, runs=[], states=[])
    for run in args.run_dirs:
        result = json.load(open(run / "result.json"))
        if not result.get("completed"):
            print(f"# {run}: incomplete, using what is recorded")
        by_state: dict[str, list[dict]] = {}
        for record in result["records"]:
            by_state.setdefault(record["state"], []).append(record)
        info = {s["state"]: s for s in result["states"]}
        print(f"\n=== {run.name}  states {len(by_state)}  branches {len(result['records'])}  "
              f"{result['physical_decisions']} decisions / {result['seconds'] / 60:.1f} min  "
              f"window {result['window']} follow {result['follow']} repeats {result['repeats']}")
        print(f"{'state':34s} {'up':>5s} {'policy':>7s} {'best':>16s} {'best ret':>8s} {'C':>7s} {'spread':>7s} "
              f"{'dec':>3s} {'top1':>5s} {'pair':>5s}")
        rows = []
        for state, records in sorted(by_state.items()):
            returns = db.returns_by_macro(records, args.key)
            if args.reference not in returns:
                continue
            c = db.consequence(returns, args.reference)
            r = db.ranking_agreement(returns)
            rows.append(dict(run=run.name, state=state, step=info[state]["step"],
                             initial_upperarm=info[state]["upperarm_ratio"], **c, **{f"rank_{k}": v for k, v in r.items()}))
            print(f"{state:34s} {info[state]['upperarm_ratio']:5.2f} {c['reference_return']:7.3f} "
                  f"{c['best_macro']:>16s} {c['best_return']:8.3f} {c['consequence']:7.3f} {c['spread']:7.3f} "
                  f"{int(c['decisive']):3d} {r['top1_agreement']:5.2f} {r['pairwise_agreement']:5.2f}")
        if not rows:
            continue
        cons = np.asarray([r["consequence"] for r in rows])
        spread = np.asarray([r["spread"] for r in rows])
        decisive = np.asarray([r["decisive"] for r in rows])
        top1 = np.asarray([r["rank_top1_agreement"] for r in rows])
        pair = np.asarray([r["rank_pairwise_agreement"] for r in rows])
        winners = Counter(r["best_macro"] for r in rows)
        decisive_winners = Counter(r["best_macro"] for r in rows if r["decisive"])
        macro_means = {m: float(np.mean([r["means"][m] for r in rows])) for m in rows[0]["means"]}
        print(f"\n  consequence: median {np.median(cons):.3f}, mean {cons.mean():.3f}, "
              f"decisive at {int(decisive.sum())}/{len(rows)} states (gain above the repeat spread)")
        print(f"  repeat spread: median {np.median(spread):.3f}; ranking agreement top-1 {top1.mean():.2f}, "
              f"pairwise {pair.mean():.2f}")
        print(f"  best macro per state: {dict(winners)}")
        print(f"  best macro at decisive states: {dict(decisive_winners)}")
        print(f"  mean return by macro: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(macro_means.items(), key=lambda kv: -kv[1])))
        best_fixed = max(macro_means, key=lambda k: macro_means[k])
        oracle = float(np.mean([max(r["means"].values()) for r in rows]))
        print(f"  best single fixed macro '{best_fixed}' {macro_means[best_fixed]:.3f} against a per-state oracle "
              f"{oracle:.3f}: the value of choosing per state is {oracle - macro_means[best_fixed]:.3f}")
        summary["runs"].append(dict(run=run.name, states=len(rows), decisive=int(decisive.sum()),
                                    median_consequence=float(np.median(cons)), median_spread=float(np.median(spread)),
                                    top1_agreement=float(top1.mean()), pairwise_agreement=float(pair.mean()),
                                    winners=dict(winners), decisive_winners=dict(decisive_winners),
                                    macro_means=macro_means, best_fixed_macro=best_fixed,
                                    oracle_return=oracle, value_of_choosing=oracle - macro_means[best_fixed]))
        summary["states"].extend(rows)
    out = args.run_dirs[0].parent / "decision_summary.json"
    out.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
