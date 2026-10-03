"""Offline audit: how predictable is the final dressing outcome from the privileged simulator state?

Uses the archived r1 rollouts of the v4/v5 multi-garment collections (successes and failures, static arm).
Two questions decide whether a learned continuation value can replace the short-horizon progress score
inside an IPC lookahead teacher:

1. Outcome noise: replicas with an identical garment, body, placement offset and seed are re-runs of the
   same episode. Their outcome agreement bounds what any evaluator can predict.
2. Predictability: from privileged per-state features (no point cloud), how well does a classifier separate
   eventual success from eventual grasp loss, as a function of the decisions remaining before the failure?

CPU only. Writes a JSON report; no simulation.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

RUNS = ("fmvp_scaled_multigarment_v4_20260925", "fmvp_scaled_multigarment_v5_20260926")
GARMENTS = ("tshirt_4", "tshirt_26", "tshirt_68", "tshirt_392", "hospital_gown")


def records(root):
    out = []
    for run in RUNS:
        for line in (root / run / "attempts.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["file"] = os.path.join(row["log"][:-4], row["path"])
            err = row.get("sim_error") or ""
            row["outcome"] = "success" if row.get("accepted") else ("grasp" if "grasp" in err else "other")
            out.append(row)
    return out


def features(path, garment, stride):
    """Per-state privileged features, the label horizon and time index."""
    z = np.load(path, allow_pickle=False)
    finger, elbow, shoulder = z["finger"], z["elbow"], z["shoulder"]
    arm = shoulder - finger
    length = np.linalg.norm(arm)
    axis = arm / length
    tcp = z["tcp"]
    rel = tcp - finger
    along = rel @ axis / length
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    force = np.linalg.norm(z["gripper_force"], axis=1)
    track = np.concatenate([[0.], z["tracking_error"]])
    act = np.concatenate([np.zeros((1, 6)), z["actions"]])
    upper, fore = z["upperarm_ratio"], z["forearm_ratio"]
    cuff, prox = z["sleeve_cuff_s"], z["sleeve_proximal_upper_fraction"]
    t = np.arange(len(tcp))
    rows = []
    for k in range(0, len(tcp), stride):
        lo = max(0, k - 10)
        rows.append([
            k / 450., along[k], radial[k], force[k], force[lo:k + 1].max(), track[k], track[lo:k + 1].max(),
            upper[k], fore[k], cuff[k], prox[k], prox[k] - prox[lo], upper[k] - upper[lo],
            *act[k], np.linalg.norm(act[lo:k + 1, :3], axis=1).mean(),
            *[float(garment == g) for g in GARMENTS]])
    return np.asarray(rows, np.float32), t[::stride], len(tcp)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/ge47gax/kun/libuipc/output/uipc_manip"))
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=3000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [r for r in records(args.root) if r["outcome"] in ("success", "grasp") and os.path.exists(r["file"])]
    report = dict(episodes_total=len(rows), outcomes=dict(collections.Counter(r["outcome"] for r in rows)))

    # 1. replica agreement: identical garment/body/offset/seed
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["garment"], r["body"], tuple(r["offset_mm"]), r["seed"])].append(r["outcome"] == "success")
    pairs = [g for g in groups.values() if len(g) >= 2]
    agree = [float(len(set(g)) == 1) for g in pairs]
    rates = np.array([np.mean(g) for g in pairs])
    report["replicas"] = dict(groups_with_replicas=len(pairs), outcome_agreement=float(np.mean(agree)) if pairs else None,
                              mixed_groups=int(sum(1 - a for a in agree)),
                              mean_success=float(rates.mean()) if pairs else None)

    # 2. predictability from privileged state
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    rows = rows[:args.max_episodes]
    bodies = sorted({r["body"] for r in rows})
    test_bodies = set(rng.choice(bodies, size=max(1, len(bodies) // 5), replace=False).tolist())
    X, y, remain, split, ep = [], [], [], [], []
    for i, r in enumerate(rows):
        f, t, n = features(r["file"], r["garment"], args.stride)
        X.append(f)
        y.append(np.full(len(f), r["outcome"] == "success"))
        # decisions until the failure (failed) or until the end (success)
        remain.append(n - 1 - t)
        split.append(np.full(len(f), r["body"] in test_bodies))
        ep.append(np.full(len(f), i))
    X, y, remain, split, ep = map(np.concatenate, (X, y, remain, split, ep))
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, random_state=0)
    clf.fit(X[~split], y[~split])
    prob = clf.predict_proba(X[split])[:, 1]
    yt, rt, tt = y[split], remain[split], X[split][:, 0] * 450
    out = dict(train_states=int((~split).sum()), test_states=int(split.sum()), test_bodies=len(test_bodies),
               auc_all=float(roc_auc_score(yt, prob)))
    # AUC by time index (early prediction) and by decisions remaining before a failure
    by_time = {}
    for lo, hi in ((0, 40), (40, 80), (80, 120), (120, 200), (200, 450)):
        m = (tt >= lo) & (tt < hi)
        if m.sum() > 50 and len(set(yt[m])) == 2:
            by_time[f"t{lo}-{hi}"] = dict(auc=float(roc_auc_score(yt[m], prob[m])), n=int(m.sum()))
    out["auc_by_time"] = by_time
    by_remaining = {}
    for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 120), (120, 450)):
        # failures that far from their failure vs successes at the same time indices
        fail = (~yt) & (rt >= lo) & (rt < hi)
        times = tt[fail]
        if fail.sum() < 30:
            continue
        succ = yt & (tt >= times.min()) & (tt <= times.max())
        m = fail | succ
        by_remaining[f"{lo}-{hi}_before_failure"] = dict(auc=float(roc_auc_score(yt[m], prob[m])),
                                                         failures=int(fail.sum()), successes=int(succ.sum()))
    out["auc_by_decisions_before_failure"] = by_remaining
    report["privileged_predictability"] = out
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
