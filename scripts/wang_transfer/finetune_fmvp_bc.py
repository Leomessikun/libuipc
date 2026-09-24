"""Filtered behaviour cloning of fmvp_sim.pt on accepted Genesis+IPC rollouts (whole actor trunk).

Run under the `curl` Python 3.9 environment (the FMVP encoder needs torch_geometric). Two stages:

* ``features``: every state of every accepted episode is passed once through the frozen point
  encoder (force input zero, as in collection); the gripper row's 50-d feature, the original
  actor's 12 outputs and the executed action are saved. Episodes are deduplicated by content hash
  and spread over worker processes.
* ``train``: the three-layer trunk is fine-tuned from the released weights on those features.
  Targets are the executed translations in the model frame (they carry the collector's slowdown
  near the shoulder and any lookahead corrections) and zero during the verified hold. The model's
  vertical rotation output is anchored to the original. A trust term keeps every output near the
  released actor's, and each body contributes equal total weight.

The checkpoint keeps the released state-dict layout, so ``uipc_manip.wang_bridge`` serves it
unchanged. This is supervised self-imitation, not offline RL: it can only reproduce behaviour that
already succeeded, faster to reach and more often.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / ".claude/worktrees/residual-rl/python"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def accepted_episodes(datasets: list[Path]) -> list[dict]:
    seen, out = set(), []
    for dataset in datasets:
        for line in (dataset / "attempts.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("accepted"):
                continue
            key = row.get("sha256") or digest(Path(row["path"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(path=row["path"], body=int(row["body"]), sha256=key))
    return out


def dagger_episodes(dirs: list[Path]) -> list[dict]:
    """Every recorded attempt under ``dirs`` that ran the IPC lookahead, accepted or not. Only the
    states the lookahead evaluated are labelled: their executed action is its choice."""
    out = []
    for root in dirs:
        for log in sorted(Path(root).rglob("lookahead.jsonl")):
            states = sorted({json.loads(line)["state"] for line in log.read_text().splitlines() if line.strip()})
            for path in sorted(log.parent.glob("*.npz")):
                out.append(dict(path=str(path), body=int(log.parent.name.split("_")[1]), sha256=digest(path),
                                labels=states))
    return out


def encode_worker(args) -> str:
    episodes, out_path, checkpoint, yaw = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(PACKAGE))
    import torch
    from torch_geometric.data import Batch, Data
    from uipc_manip.obs import ObsSpec
    from uipc_manip.wang_bridge import GRIPPER, ReferencePolicy, to_reference_cloud

    torch.set_num_threads(1)
    policy = ReferencePolicy(checkpoint, device="cpu", yaw_deg=yaw)
    rotation = policy.rotation.astype(np.float32)          # ours -> model: v @ R.T
    spec = ObsSpec(768)
    feats, logits, targets, weights, bodies, kinds = [], [], [], [], [], []
    for episode in episodes:
        with np.load(episode["path"], allow_pickle=False) as data:
            obs = np.asarray(data["obs"][:-1], np.float32)
            actions = np.asarray(data["actions"], np.float32)
            controller = np.asarray(data["controller_id"], np.int8)
        labels = episode.get("labels")
        steps = range(len(actions)) if labels is None else [t for t in labels if t < len(actions)]
        for t in steps:
            pos, flags, valid, _ = spec.unpack_numpy(obs[t])
            ref_pos, ref_flags = to_reference_cloud(pos[valid], flags[valid], yaw_deg=yaw, voxel=policy.voxel)
            batch = Batch.from_data_list([Data(x=torch.from_numpy(ref_flags), pos=torch.from_numpy(ref_pos))])
            with torch.no_grad():
                encoded, _ = policy.encoder(batch, torch.zeros(1, 3)) if policy.film else policy.encoder(batch)
                row = encoded[batch.x[:, GRIPPER] == 1]
                out = policy.trunk(row)[0]
            feats.append(row[0].numpy())
            logits.append(out.numpy())
            if labels is not None:                          # IPC lookahead's choice at a visited state
                targets.append(np.clip(actions[t, :3] @ rotation.T, -1, 1)); kinds.append(6)
            elif controller[t] == 2:                        # verified hold: stand still
                targets.append(np.zeros(3, np.float32)); kinds.append(2)
            else:                                           # executed translation, model frame
                targets.append(np.clip(actions[t, :3] @ rotation.T, -1, 1)); kinds.append(int(controller[t]))
            bodies.append(episode["body"])
        print(f"[features] body={episode['body']} states={len(steps)}", flush=True)
    np.savez(out_path, feats=np.stack(feats), logits=np.stack(logits), targets=np.stack(targets),
             bodies=np.asarray(bodies), kinds=np.asarray(kinds, np.int8))
    return str(out_path)


def features(args) -> None:
    from multiprocessing import get_context

    episodes = accepted_episodes(args.datasets) + dagger_episodes(args.dagger)
    if args.exclude_bodies:
        episodes = [e for e in episodes if e["body"] not in set(args.exclude_bodies)]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes.json").write_text(json.dumps(episodes, indent=1))
    shards = [episodes[i::args.workers] for i in range(args.workers)]
    jobs = [(s, args.out / f"shard_{i:02d}.npz", str(args.checkpoint), args.yaw) for i, s in enumerate(shards) if s]
    with get_context("spawn").Pool(len(jobs)) as pool:
        for path in pool.imap_unordered(encode_worker, jobs):
            print(f"[shard] {path}", flush=True)
    print(f"[features] {len(episodes)} episodes from {len({e['body'] for e in episodes})} bodies", flush=True)


def train(args) -> None:
    import torch

    shards = sorted(args.features.glob("shard_*.npz"))
    parts = [np.load(p) for p in shards]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("feats", "logits", "targets", "bodies", "kinds")}
    n = len(cat["feats"])
    bodies, counts = np.unique(cat["bodies"], return_counts=True)
    per_body = dict(zip(bodies.tolist(), counts.tolist()))
    weight = np.array([1.0 / per_body[b] for b in cat["bodies"]], np.float32)
    weight *= np.where(cat["kinds"] == 2, args.hold_weight, 1.0)
    weight *= np.where(cat["kinds"] == 6, args.dagger_weight, 1.0)
    weight /= weight.mean()
    print(f"[train] {n} states, {len(bodies)} bodies, holds {int((cat['kinds'] == 2).sum())}, "
          f"lookahead labels {int((cat['kinds'] == 6).sum())}", flush=True)

    payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    trunk = torch.nn.Sequential(torch.nn.Linear(50, 1024), torch.nn.ReLU(), torch.nn.Linear(1024, 1024),
                                torch.nn.ReLU(), torch.nn.Linear(1024, 12))
    start = state if args.init is None else torch.load(str(args.init), map_location="cpu", weights_only=False)["model_state_dict"]
    trunk.load_state_dict({k[len("trunk."):]: v for k, v in start.items() if k.startswith("trunk.")}, strict=True)
    x = torch.from_numpy(cat["feats"]).float()
    base = torch.from_numpy(cat["logits"]).float()[:, :6].tanh()
    target = torch.from_numpy(cat["targets"]).float()
    w = torch.from_numpy(weight)
    rng = np.random.default_rng(args.seed)
    valid_bodies = set(rng.choice(bodies, size=max(1, len(bodies) // 8), replace=False).tolist())
    is_val = torch.from_numpy(np.isin(cat["bodies"], list(valid_bodies)))
    opt = torch.optim.Adam(trunk.parameters(), lr=args.lr)

    def losses(idx):
        mu = trunk(x[idx])[:, :6].tanh()
        fit = ((mu[:, :3] - target[idx]).square().sum(1) * w[idx]).mean()
        trust = (mu - base[idx]).square().sum(1).mean() + 4.0 * (mu[:, 4] - base[idx][:, 4]).square().mean()
        return fit, trust

    train_idx = torch.nonzero(~is_val).squeeze(1)
    val_idx = torch.nonzero(is_val).squeeze(1)
    with torch.no_grad():
        f0, _ = losses(val_idx)
    print(f"[train] held-out bodies {sorted(valid_bodies)}: fit before {f0.item():.4f}", flush=True)
    for epoch in range(args.epochs):
        perm = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(perm), args.batch):
            fit, trust = losses(perm[start:start + args.batch])
            loss = fit + args.trust * trust
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vf, vt = losses(val_idx)
            tf, tt = losses(train_idx[:20000])
        print(f"[train] epoch {epoch + 1}: train fit {tf.item():.4f} trust {tt.item():.4f} | "
              f"held-out fit {vf.item():.4f} trust {vt.item():.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    new_state = dict(state)
    for k, v in trunk.state_dict().items():
        new_state["trunk." + k] = v.detach().clone()
    out = {**payload, "model_state_dict": new_state,
           "ipc_bc": dict(features=str(args.features.resolve()), states=n, bodies=len(bodies),
                          heldout_bodies=sorted(valid_bodies), epochs=args.epochs, lr=args.lr,
                          trust=args.trust, hold_weight=args.hold_weight)}
    path = args.out / "fmvp_ipc_bc.pt"
    torch.save(out, path)
    (args.out / "manifest.json").write_text(json.dumps(out["ipc_bc"], indent=1))
    print(f"[saved] {path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("features")
    f.add_argument("--datasets", type=Path, nargs="+", required=True)
    f.add_argument("--dagger", type=Path, nargs="*", default=[],
                   help="Directories of lookahead-labelled attempts (accepted or not).")
    f.add_argument("--exclude-bodies", type=int, nargs="*", default=[])
    f.add_argument("--checkpoint", type=Path, default=Path("/home/ge47gax/Desktop/fmvp_sim.pt"))
    f.add_argument("--yaw", type=float, default=267.0)
    f.add_argument("--workers", type=int, default=24)
    f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--features", type=Path, required=True)
    t.add_argument("--checkpoint", type=Path, default=Path("/home/ge47gax/Desktop/fmvp_sim.pt"))
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--trust", type=float, default=0.5)
    t.add_argument("--hold-weight", type=float, default=2.0)
    t.add_argument("--dagger-weight", type=float, default=3.0)
    t.add_argument("--init", type=Path, default=None, help="Start the trunk from this checkpoint instead.")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    features(a) if a.stage == "features" else train(a)


if __name__ == "__main__":
    main()
