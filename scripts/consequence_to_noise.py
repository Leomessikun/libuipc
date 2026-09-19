"""How much of an outcome does the choice of action explain, against run-to-run noise?

A learner can only act on a difference it can see. The counterfactual branch runs
repeated every intervention against itself with bitwise identical commands from a
bitwise identical restored state, so at each state the outcomes decompose the way a
one-way analysis of variance does: variation *between* the interventions, which is the
learnable signal, against variation *within* one intervention's own repeats, which is
the simulator's irreproducibility and nothing else.

Three statistics are reported, because they answer different questions and only the
last one governs whether a learner can improve a policy that is already competent:

* ``eta^2`` — the share of outcome variance the choice explains. It says whether a
  *bad* intervention is detectable, and it is dominated by the obviously bad ones.
* ``F`` — the same decomposition scaled by the noise, so it says how many repeats a
  detection costs.
* ``margin`` — the gap between the best intervention's mean and the second best's,
  divided by the standard deviation of one intervention's own repeats. Improving a
  competent policy means preferring the best action over the next best, so this is
  the ratio that decides whether the improvement is visible in one trial at all.

All three are dimensionless or reported beside the raw scale, so tasks whose outcomes
are measured in different units belong in the same table.

This reads finished branch runs only; it simulates nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def rows_for(path: Path, key: str, reference: str) -> list[dict]:
    result = json.load(open(path / "result.json"))
    grouped: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for record in result["records"]:
        grouped[(record["state"], record["macro"])].append(float(record[key]))
    by_state: dict[str, dict[str, np.ndarray]] = collections.defaultdict(dict)
    for (state, macro), values in grouped.items():
        by_state[state][macro] = np.asarray(values)
    out = []
    for state, macros in sorted(by_state.items()):
        if reference not in macros or min(len(v) for v in macros.values()) < 2:
            continue
        # The noise floor is what one macro's own repeats do; the signal is how far the
        # best macro's mean is from the reference macro's mean.
        spread = float(np.median([v.max() - v.min() for v in macros.values()]))
        means = {m: float(v.mean()) for m, v in macros.items()}
        best = max(means, key=lambda m: means[m])
        values = list(macros.values())
        counts = np.asarray([len(v) for v in values])
        group_means = np.asarray([v.mean() for v in values])
        grand = float(np.concatenate(values).mean())
        between = float((counts * (group_means - grand) ** 2).sum())
        within = float(sum(((v - v.mean()) ** 2).sum() for v in values))
        df_between, df_within = len(values) - 1, int(counts.sum()) - len(values)
        total = between + within
        out.append(dict(run=path.name, state=state, best=best,
                        consequence=means[best] - means[reference], spread=spread,
                        # eta squared: the share of outcome variance the choice explains.
                        eta_squared=between / total if total > 0 else float("nan"),
                        f_ratio=((between / df_between) / (within / df_within)
                                 if within > 0 and df_within > 0 else float("inf")),
                        top_two_margin=float(np.diff(np.sort(group_means)[-2:])[0]),
                        repeat_sd=float(np.sqrt(within / df_within)) if df_within > 0 else float("nan"),
                        degenerate=bool(total == 0.0)))
    return out


def report(name: str, rows: list[dict]) -> dict:
    live = [r for r in rows if not r["degenerate"]]
    if not live:
        print(f"{name:32s} {len(rows):4d}   every outcome identical: no variance to decompose")
        return dict(name=name, states=len(rows), degenerate_states=len(rows))
    consequence = np.asarray([r["consequence"] for r in live])
    spread = np.asarray([r["spread"] for r in live])
    eta = np.asarray([r["eta_squared"] for r in live])
    f = np.asarray([r["f_ratio"] for r in live])
    sd = np.asarray([r["repeat_sd"] for r in live])
    margin = np.asarray([r["top_two_margin"] for r in live])
    visible = sd > 0
    margin_ratio = margin[visible] / sd[visible]
    summary = dict(name=name, states=len(rows), degenerate_states=len(rows) - len(live),
                   median_consequence=float(np.median(consequence)),
                   median_spread=float(np.median(spread)),
                   median_eta_squared=float(np.median(eta)), mean_eta_squared=float(eta.mean()),
                   median_f_ratio=float(np.median(f)),
                   fraction_choice_explains_half=float(np.mean(eta > 0.5)),
                   median_top_two_margin=float(np.median(margin)),
                   median_margin_over_sd=float(np.median(margin_ratio)) if margin_ratio.size else float("nan"),
                   fraction_margin_below_sd=float(np.mean(margin_ratio < 1.0)) if margin_ratio.size else float("nan"))
    print(f"{name:32s} {len(live):4d} {summary['median_consequence']:+11.4f} "
          f"{summary['median_spread']:11.4f} {summary['median_eta_squared']:8.3f} "
          f"{summary['median_f_ratio']:11.1f} {summary['median_margin_over_sd']:8.2f} "
          f"{100 * summary['fraction_margin_below_sd']:10.0f} %")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", action="append", default=[], metavar="NAME=DIR[,DIR...]:KEY",
                   help="A named group of branch runs and the outcome key to read")
    p.add_argument("--reference", default="policy")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    print(f"{'task':32s} {'states':>4s} {'consequence':>11s} {'noise floor':>11s} "
          f"{'eta^2':>8s} {'F':>11s} {'margin/sd':>8s} {'margin<sd':>12s}")
    summaries, states = [], []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        dirs, key = rest.rsplit(":", 1)
        rows = [r for d in dirs.split(",") for r in rows_for(Path(d), key, args.reference)]
        if not rows:
            print(f"{name:32s} no comparable states")
            continue
        summaries.append(report(name, rows))
        states.extend(dict(task=name, **r) for r in rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(reference=args.reference, tasks=summaries, states=states),
                                       indent=1) + "\n")
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
