"""Can the behavior prior produce the recovery that actually helps?

The counterfactual branch study found macros that beat the policy's own continuation at
the states it visits. Those actions come from outside the prior's data: they are fixed
directions in the arm's frame, not anything a teacher demonstrated. Reversing them
through the behavior flow at the same observation says whether the prior could have
produced them, and how far into its noise distribution one would have to reach.

The policy's own action at the same state is reversed alongside as the reference: it is
the action the prior does produce, so its latent is the scale the macros are read
against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import decision_branches as db  # noqa: E402
from uipc_manip import flow_reversal as fr  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--branches", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--step-m", type=float, default=0.008)
    args = p.parse_args()
    from uipc_manip.fql import FQLAgent

    agent, _ = FQLAgent.load(args.checkpoint, device=args.device)
    observations, actions, labels = [], [], []
    for run in args.branches:
        result = json.load(open(run / "result.json"))
        arrays = np.load(run / "states.npz")
        max_translation = float(result["env"]["max_translation"])
        best = {}
        by_state: dict[str, list] = {}
        for record in result["records"]:
            by_state.setdefault(record["state"], []).append(record)
        for state, records in by_state.items():
            returns = db.returns_by_macro(records)
            best[state] = db.consequence(returns)
        for info in result["states"]:
            state = info["state"]
            directions = {k: np.asarray(v, dtype=float) for k, v in info["directions"].items()}
            observation = arrays[f"{state}__observation"]
            for macro in db.MACROS:
                if macro["kind"] != "direction":
                    continue
                # The first decision of the macro's window is the one taken at this state.
                action = db.macro_action(macro, 0, result["window"], directions, np.zeros(6),
                                         args.step_m, max_translation)
                observations.append(observation)
                actions.append(action)
                labels.append(dict(run=run.name, state=state, macro=macro["name"], step=info["step"],
                                   is_best=bool(best[state]["best_macro"] == macro["name"]),
                                   consequence=float(best[state]["means"][macro["name"]] -
                                                     best[state]["means"]["policy"])))
    report = fr.reversal_report(agent, np.stack(observations), np.stack(actions))
    rows = []
    for label, norm, percentile, error in zip(labels, report["latent_norm"], report["latent_percentile"],
                                              report["reconstruction_error"], strict=True):
        rows.append(dict(**label, latent_norm=float(norm), latent_percentile=float(percentile),
                         reconstruction_error=float(error)))
    print(f"{len(rows)} (state, macro) pairs over {len({r['state'] for r in rows})} states\n")
    print(f"{'macro':18s} {'n':>4s} {'consequence':>11s} {'|z| median':>10s} {'percentile':>10s} {'>99th':>6s} {'recon median':>12s}")
    for macro in sorted({r["macro"] for r in rows}):
        group = [r for r in rows if r["macro"] == macro]
        print(f"{macro:18s} {len(group):4d} {np.mean([r['consequence'] for r in group]):+11.4f} "
              f"{np.median([r['latent_norm'] for r in group]):10.3f} "
              f"{np.median([r['latent_percentile'] for r in group]):10.3f} "
              f"{np.mean([r['latent_percentile'] > 0.99 for r in group]):6.2f} "
              f"{np.median([r['reconstruction_error'] for r in group]):12.4f}")
    winners = [r for r in rows if r["is_best"]]
    print(f"\nthe macro that wins at its own state ({len(winners)} states): "
          f"|z| median {np.median([r['latent_norm'] for r in winners]):.3f}, "
          f"percentile {np.median([r['latent_percentile'] for r in winners]):.3f}, "
          f"reconstruction {np.median([r['reconstruction_error'] for r in winners]):.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), rows=rows), indent=1) + "\n")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
