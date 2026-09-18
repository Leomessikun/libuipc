"""Do the signals a supervisor would trigger on separate failure from success?

A supervisor is only useful if the condition it fires on happens in the episodes that
fail and not in the ones that succeed. This measures the base rate of each candidate
signal in both, from the same teacher's own privileged traces, before any threshold is
chosen. Nothing is trained and no simulator runs.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import dressing_supervisor as ds  # noqa: E402


def signals(privileged: np.ndarray, stall_window: int) -> dict:
    """Per-decision containment, tracking and arc progress, with the on-arm mask."""
    states = [ds.sleeve_position(row) for row in privileged]
    arc = np.asarray([s["arc"] for s in states])
    progress = np.full(len(arc), np.nan)
    progress[stall_window:] = arc[stall_window:] - arc[:-stall_window]
    return dict(containment=np.asarray([s["containment"] for s in states]),
                tracking_cm=np.asarray([s["tracking_cm"] for s in states]),
                on_arm=np.asarray([s["on_arm"] for s in states]),
                upperarm=np.asarray([s["upperarm_ratio"] for s in states]),
                arc_progress=progress)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--stall-window", type=int, default=20)
    p.add_argument("--success-coverage", type=float, default=0.7)
    p.add_argument("--grasp-limit-cm", type=float, default=2.0)
    args = p.parse_args()
    groups: dict[str, list[dict]] = {"succeeded": [], "failed": []}
    for path in sorted(glob.glob(str(args.dataset / "episodes" / "*.npz"))):
        data = np.load(path, allow_pickle=True)
        record = data["record"].item()
        record = json.loads(record) if isinstance(record, str) else record
        privileged = data["privileged"].astype(np.float64)
        dressed = float(privileged[-1, 25]) >= args.success_coverage
        held = float(privileged[:, 29].max()) <= args.grasp_limit_cm
        groups["succeeded" if (dressed and held) else "failed"].append(
            dict(cell=f"{record['garment']}/{record['human']}", **signals(privileged, args.stall_window)))
    print(f"{len(groups['succeeded'])} successful and {len(groups['failed'])} failed episodes, "
          f"stall window {args.stall_window} decisions\n")
    print(f"{'signal':34s} {'succeeded':>22s} {'failed':>22s}")
    for name, extract in (
            ("containment while on the arm", lambda g: g["containment"][g["on_arm"]]),
            ("tracking error, cm", lambda g: g["tracking_cm"]),
            ("arc progress per window, on arm", lambda g: g["arc_progress"][g["on_arm"] & np.isfinite(g["arc_progress"])])):
        cells = []
        for outcome in ("succeeded", "failed"):
            values = np.concatenate([extract(g) for g in groups[outcome]]) if groups[outcome] else np.array([np.nan])
            cells.append(f"{np.median(values):7.3f} [{np.percentile(values, 10):6.3f},{np.percentile(values, 90):6.3f}]")
        print(f"{name:34s} {cells[0]:>22s} {cells[1]:>22s}")
    print("\nfraction of decisions a threshold would fire on (successful | failed):")
    for label, extract, thresholds, above in (
            ("containment >", lambda g: g["containment"][g["on_arm"]], (0.4, 0.6, 0.8, 1.0, 1.2), True),
            ("tracking cm >", lambda g: g["tracking_cm"], (1.0, 1.2, 1.5, 1.8, 2.0), True),
            ("arc progress <", lambda g: g["arc_progress"][g["on_arm"] & np.isfinite(g["arc_progress"])],
             (0.005, 0.01, 0.02, 0.04), False)):
        for threshold in thresholds:
            rates = []
            for outcome in ("succeeded", "failed"):
                values = np.concatenate([extract(g) for g in groups[outcome]]) if groups[outcome] else np.array([np.nan])
                rates.append(float((values > threshold).mean() if above else (values < threshold).mean()))
            print(f"  {label} {threshold:<6g} {100 * rates[0]:6.1f} % | {100 * rates[1]:6.1f} %")


if __name__ == "__main__":
    main()
