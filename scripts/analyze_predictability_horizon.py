"""Growth rates and predictability horizons from ``measure_predictability_horizon.py`` outputs.

For every snapshot and perturbation group: the mean pairwise RMS vertex distance between
continuations at each decision, the least-squares exponential growth rate over the window
before saturation, its doubling time, and the first decision at which the separation
exceeds 1 mm and 1 cm. Printed as a table and written next to the traces as
``horizon.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def pairwise_rms(runs: list[np.ndarray]) -> np.ndarray:
    """Mean over pairs of the RMS vertex distance, per decision."""
    d = [np.sqrt(((a - b) ** 2).sum(-1).mean(-1)) for a, b in itertools.combinations(runs, 2)]
    return np.mean(d, axis=0)


def growth_rate(d: np.ndarray, saturation: float) -> tuple[float, int]:
    """Slope of log d over the decisions before d reaches ``saturation`` (at least three points)."""
    keep = np.where(d < saturation)[0]
    n = int(keep[-1]) + 1 if len(keep) else 0
    n = max(n, 3)
    t = np.arange(1, n + 1)
    y = np.log(np.maximum(d[:n], 1e-12))
    slope = float(np.polyfit(t, y, 1)[0])
    return slope, n


def first_exceed(d: np.ndarray, level: float) -> int | None:
    k = np.where(d > level)[0]
    return int(k[0]) + 1 if len(k) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--saturation-m", type=float, default=0.02, help="Fit the growth rate below this separation")
    args = p.parse_args()
    result = json.load(open(args.run_dir / "result.json"))
    traces = np.load(args.run_dir / "traces.npz")
    dt = float(result["decision_seconds"])
    out = dict(cell=result["arguments"]["cell"], decision_seconds=dt, snapshots=[])
    print(f"{result['arguments']['cell']}  decision {dt:.2f} s  max translation {1000 * result['max_translation_m']:.2f} mm")
    print(f"{'step':>4s} {'stage':12s} {'up':>5s} {'group':10s} {'eps':>6s} {'n':>2s} {'d1 mm':>7s} {'d5 mm':>7s} {'d10 mm':>7s} {'d20 mm':>7s} {'d40 mm':>7s} {'rate/dec':>8s} {'double':>7s} {'>1mm':>5s} {'>1cm':>5s} {'final up spread':>15s}")
    for snap in result["snapshots"]:
        step = snap["step"]
        groups: dict[float, list[str]] = {}
        for k in traces.files:
            if k.startswith(f"s{step}_eps"):
                eps = float(k.split("_eps")[1].split("_r")[0])
                groups.setdefault(eps, []).append(k)
        entry = dict(step=step, stage=snap["stage"], initial_upperarm=snap["initial"]["upperarm_ratio"], groups=[])
        for eps in sorted(groups):
            runs = [traces[k] for k in sorted(groups[eps])]
            d = pairwise_rms(runs)
            rate, n = growth_rate(d, args.saturation_m)
            finals = [r["trace"][-1]["upperarm_ratio"] for r in snap["runs"] if r["eps"] == eps and r["group"] != "expert_closed_loop"]
            rec = dict(eps=eps, runs=len(runs), rms_by_decision_m=d.tolist(), growth_rate_per_decision=rate,
                       growth_rate_per_second=rate / dt, fit_decisions=n,
                       doubling_decisions=(np.log(2) / rate if rate > 0 else None),
                       first_over_1mm=first_exceed(d, 1e-3), first_over_1cm=first_exceed(d, 1e-2),
                       final_upperarm_spread=float(np.ptp(finals)) if finals else None)
            entry["groups"].append(rec)
            g = "identical" if eps == 0 else "perturbed"
            pick = lambda i: f"{1000 * d[i - 1]:7.3f}" if len(d) >= i else "      -"
            dbl = f"{rec['doubling_decisions']:7.1f}" if rec["doubling_decisions"] else "      -"
            print(f"{step:4d} {snap['stage']:12s} {snap['initial']['upperarm_ratio']:5.2f} {g:10s} {eps:6.0e} {len(runs):2d} {pick(1)} {pick(5)} {pick(10)} {pick(20)} {pick(40)} {rate:8.3f} {dbl} {str(rec['first_over_1mm']):>5s} {str(rec['first_over_1cm']):>5s} {rec['final_upperarm_spread'] if rec['final_upperarm_spread'] is None else round(rec['final_upperarm_spread'], 3):>15}")
        out["snapshots"].append(entry)
    (args.run_dir / "horizon.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
