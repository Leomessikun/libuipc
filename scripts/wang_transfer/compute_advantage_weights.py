"""Per-state advantage weights for post-training from successes and failures alike.

For every encoded state (episode, step) of a finetune_fmvp_bc feature directory, the weight is

    A = V(s_{t+k}) - V(s_t),   w = clip(exp(A / beta), w_min, w_max),

with V the privileged outcome value (outcome_value.py) trained on the logging policy's own rollouts: actions
followed by a rise in predicted success are imitated more, actions followed by a fall less. Failed episodes
therefore contribute their good actions and discount their bad ones instead of being discarded. States within
k of an episode's end use the final outcome as V(s_{t+k}). Writes one weights file per shard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome_value import OutcomeValue, archive_features  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--outcomes", type=Path, required=True, help="outcomes.json keyed by episode path")
    p.add_argument("--value", type=Path, required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--w-min", type=float, default=.05)
    p.add_argument("--w-max", type=float, default=20.)
    p.add_argument("--mode", choices=("advantage", "success_only", "uniform"), default="advantage")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    value = OutcomeValue(args.value)
    episodes = json.loads((args.features / "episodes.json").read_text())
    outcomes = json.loads(args.outcomes.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    cache = {}
    stats = []
    for shard in sorted(args.features.glob("shard_*.npz")):
        data = np.load(shard)
        steps, eps = data["steps"], data["episodes"]
        weights = np.ones(len(steps), np.float32)
        for e in np.unique(eps):
            ep = episodes[int(e)]
            info = outcomes[ep["path"]]
            mask = eps == e
            if args.mode == "uniform":
                continue
            if args.mode == "success_only":
                weights[mask] = float(info["outcome"])
                continue
            if ep["path"] not in cache:
                with np.load(ep["path"], allow_pickle=False) as z:
                    n = len(z["tcp"])
                    v = value(archive_features(z, info["garment"], range(n)))
                cache[ep["path"]] = v
            v = cache[ep["path"]]
            n = len(v)
            t = steps[mask]
            later = np.where(t + args.k < n, v[np.minimum(t + args.k, n - 1)], float(info["outcome"]))
            adv = later - v[np.minimum(t, n - 1)]
            weights[mask] = np.clip(np.exp(adv / args.beta), args.w_min, args.w_max)
            stats.append((float(info["outcome"]), float(np.mean(adv)), float(np.mean(weights[mask]))))
        np.savez(args.out / shard.name, weights=weights)
    if stats:
        s = np.array(stats)
        for label, m in (("success", s[:, 0] == 1), ("failure", s[:, 0] == 0)):
            print(f"[weights] {label}: episodes {int(m.sum())}, mean advantage {s[m, 1].mean():+.4f}, "
                  f"mean weight {s[m, 2].mean():.3f}")
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=1))


if __name__ == "__main__":
    main()
