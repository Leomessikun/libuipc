"""Train the outcome value (outcome_value.py) on archived r1 rollouts; split by body; report held-out AUC.

Labels: 1 for an accepted episode, 0 for a grasp-loss episode (other failures are excluded). Every state of
an episode carries its final label, i.e. the Monte-Carlo success of continuing with the logging policy.
Run with a Python that has torch (e.g. the curl env); the saved model needs only numpy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome_value import NAMES, OutcomeValue, archive_features, save  # noqa: E402

RUNS = ("fmvp_scaled_multigarment_v4_20260925", "fmvp_scaled_multigarment_v5_20260926")


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = y.astype(bool)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/ge47gax/kun/libuipc/output/uipc_manip"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--min-step", type=int, default=20)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude-bodies", type=Path, default=None, help="Text file of body ids never used (test bodies)")
    p.add_argument("--drop", nargs="*", default=[], help="Feature-name prefixes to leave out (e.g. action_ grip)")
    args = p.parse_args()
    import torch

    excluded = set(int(b) for b in args.exclude_bodies.read_text().split()) if args.exclude_bodies else set()
    rows = []
    for run in RUNS:
        for line in (args.root / run / "attempts.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            err = r.get("sim_error") or ""
            if not (r.get("accepted") or "grasp" in err) or int(r["body"]) in excluded:
                continue
            r["file"] = os.path.join(r["log"][:-4], r["path"])
            if os.path.exists(r["file"]):
                rows.append(r)
    rng = np.random.default_rng(args.seed)
    bodies = sorted({int(r["body"]) for r in rows})
    val_bodies = set(rng.choice(bodies, size=max(1, len(bodies) // 5), replace=False).tolist())
    X, y, val, tt = [], [], [], []
    for r in rows:
        with np.load(r["file"], allow_pickle=False) as z:
            n = len(z["tcp"])
            # Decision states only: an accepted episode's verified hold after its success state is not a
            # decision the policy has to make, and teaches "standing still near the top means success".
            end = int(r["success_state"]) if r.get("accepted") and r.get("success_state") is not None else n
            steps = list(range(args.min_step, end, args.stride))
            if not steps:
                continue
            X.append(archive_features(z, r["garment"], steps))
        y.append(np.full(len(steps), float(bool(r.get("accepted")))))
        val.append(np.full(len(steps), int(r["body"]) in val_bodies))
        tt.append(np.asarray(steps))
    X, y, val, tt = map(np.concatenate, (X, y, val, tt))
    keep = [i for i, n in enumerate(NAMES) if not any(n.startswith(d) for d in args.drop)]
    X_full = X
    X = X[:, keep]
    mean, std = X[~val].mean(0), X[~val].std(0) + 1e-6
    torch.manual_seed(args.seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 128), torch.nn.ReLU(), torch.nn.Linear(128, 128),
                              torch.nn.ReLU(), torch.nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.tensor((X[~val] - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y[~val], dtype=torch.float32)
    xv = torch.tensor((X[val] - mean) / std, dtype=torch.float32)
    log = []
    for epoch in range(args.epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 1024):
            b = perm[i:i + 1024]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b])[:, 0], yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            with torch.no_grad():
                pv = torch.sigmoid(net(xv)[:, 0]).numpy()
            log.append(dict(epoch=epoch + 1, loss=float(loss), val_auc=auc(y[val], pv)))
            print(log[-1], flush=True)
    layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()) for m in net if isinstance(m, torch.nn.Linear)]
    meta = dict(features=list(NAMES), keep=keep, dropped=args.drop, runs=list(RUNS), episodes=len(rows), states=int(len(X)),
                val_bodies=sorted(val_bodies), excluded_bodies=sorted(excluded), stride=args.stride,
                min_step=args.min_step, epochs=args.epochs, seed=args.seed, log=log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(args.out, mean.astype(np.float32), std.astype(np.float32), layers, meta)
    model = OutcomeValue(args.out)
    pv = model(X_full[val])
    # sensitivity: realistic measurement noise on the force/tracking features of the full vector
    rng2 = np.random.default_rng(1)
    noisy = X_full[val].copy()
    for name, scale in (("grip_N", 10.), ("grip_max_N", 10.), ("track_m", .001), ("track_max_m", .001)):
        j = NAMES.index(name)
        noisy[:, j] = np.maximum(noisy[:, j] + rng2.normal(0, scale, len(noisy)), 0.)
    sensitivity = float(np.mean(np.abs(model(noisy) - pv)))
    by_time = {f"t{lo}-{hi}": auc(y[val][(tt[val] >= lo) & (tt[val] < hi)], pv[(tt[val] >= lo) & (tt[val] < hi)])
               for lo, hi in ((20, 60), (60, 120), (120, 200), (200, 450))}
    report = dict(val_auc=auc(y[val], pv), val_auc_by_time=by_time, noise_sensitivity=sensitivity, dropped=args.drop, train_states=int((~val).sum()),
                  val_states=int(val.sum()), episodes=len(rows))
    (args.out.parent / (args.out.stem + "_report.json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
