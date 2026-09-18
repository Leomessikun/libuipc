"""Per-cell comparison of the scripted teacher with and without the supervisor."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    args = p.parse_args()
    result = json.load(open(args.run / "result.json"))
    arms = list(dict.fromkeys(e["arm"] for e in result["episodes"]))
    by_cell: dict[str, dict] = {}
    for e in result["episodes"]:
        by_cell.setdefault(e["cell"], {})[e["arm"]] = e
    print(f"{result['decisions']} decisions, {len(by_cell)} cells, arms {arms}, "
          f"{result['seconds'] / 60:.1f} min\n")
    header = f"{'cell':26s}" + "".join(f"{a + ' cov':>14s}{a + ' ok':>8s}" for a in arms) + "  overrides"
    print(header)
    flips = Counter()
    for cell, entry in sorted(by_cell.items()):
        row = f"{cell:26s}"
        for arm in arms:
            e = entry.get(arm)
            row += f"{e['sustained_coverage']:14.3f}{int(e['success']):8d}" if e else f"{'-':>14s}{'-':>8s}"
        overrides = entry.get(arms[-1], {}).get("overrides", {})
        row += "  " + ", ".join(f"{k} {v}" for k, v in sorted(overrides.items()) if k != "teacher")
        print(row)
        if len(arms) == 2 and all(a in entry for a in arms):
            before, after = entry[arms[0]]["success"], entry[arms[1]]["success"]
            flips[("fixed" if after and not before else "broken" if before and not after
                   else "both" if before else "neither")] += 1
    print()
    for arm in arms:
        rows = [e for e in result["episodes"] if e["arm"] == arm]
        reasons = Counter()
        for e in rows:
            reasons.update({k: v for k, v in e.get("overrides", {}).items() if k != "teacher"})
        print(f"{arm:12s} successes {sum(e['success'] for e in rows):2d}/{len(rows)} | "
              f"dressed {sum(e['sustained_coverage'] >= 0.7 for e in rows):2d} | "
              f"grasp valid {sum(e['whole_episode_grasp_valid'] for e in rows):2d} | "
              f"mean sustained {np.mean([e['sustained_coverage'] for e in rows]):.4f} | "
              f"mean tracking {100 * np.mean([e['max_tracking_error'] for e in rows]):.2f} cm | "
              f"overrides {dict(reasons)}")
    if flips:
        print(f"\nper cell: {dict(flips)}")


if __name__ == "__main__":
    main()
