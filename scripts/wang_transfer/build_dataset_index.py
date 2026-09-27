"""One index of accepted checkpoint rollouts across collection runs, split by body.

Reads each run's ``attempts.jsonl``, keeps accepted episodes whose file exists, deduplicates by content
hash (or path), and writes ``index.json`` with per-episode garment, body, pose region, transitions, gripper
load and the run it came from. Train/validation/test are split by body id (a body never appears in two
splits), stratified by pose region, so held-out numbers measure generalisation to new bodies.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    episodes, seen = [], set()
    for run in a.runs:
        for line in (run / "attempts.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("accepted"):
                continue
            path = run / r["path"] if (run / r["path"]).exists() else Path(r["log"]).with_suffix("") / r["path"]
            if not path.exists():
                continue
            key = r.get("sha256") or str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            body = int(r["body"])
            episodes.append(dict(path=str(path.resolve()), run=run.name, garment=r.get("garment", "tshirt_26"),
                                 body=body, region=body // 1000 - 1, transitions=int(r.get("transitions") or 0),
                                 gripper_p90_N=r.get("gripper_p90_N"), gripper_peak_N=r.get("gripper_peak_N"),
                                 stage=r.get("stage"), seed=r.get("seed")))
    rng = np.random.default_rng(a.seed)
    by_region = collections.defaultdict(set)
    for e in episodes:
        by_region[e["region"]].add(e["body"])
    split_of = {}
    for region, bodies in sorted(by_region.items()):
        bodies = sorted(bodies)
        rng.shuffle(bodies)
        n_val = max(1, round(a.val * len(bodies))) if len(bodies) >= 5 else 0
        n_test = max(1, round(a.test * len(bodies))) if len(bodies) >= 5 else 0
        for i, b in enumerate(bodies):
            split_of[b] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for e in episodes:
        e["split"] = split_of[e["body"]]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    summary = {s: dict(episodes=sum(e["split"] == s for e in episodes),
                       transitions=sum(e["transitions"] for e in episodes if e["split"] == s),
                       bodies=len({e["body"] for e in episodes if e["split"] == s}))
               for s in ("train", "val", "test")}
    garments = collections.Counter(e["garment"] for e in episodes)
    a.out.write_text(json.dumps(dict(runs=[str(r) for r in a.runs], summary=summary, garments=garments,
                                     episodes=episodes), indent=1))
    print(json.dumps(dict(summary=summary, garments=garments), indent=1))


if __name__ == "__main__":
    main()
