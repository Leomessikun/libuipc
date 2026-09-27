"""Train the flow-matching chunk policy (``uipc_manip.flow_policy``) on the body-split rollout index.

Run under the ``curl`` environment. Two stages:

* ``features``: every state of every indexed episode through the frozen encoder (``obs_features``), with
  the gripper load and the executed action in the model frame; one ``.npz`` shard per worker.
* ``train``: rectified-flow training of the action chunk given the normalised observation summary; validates
  on the index's ``val`` split (flow-sampled first action against the executed one) and saves a checkpoint
  the bridge serves (``wang_bridge --serve`` recognises ``flow_policy``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / ".claude/worktrees/residual-rl/python"
R1 = HERE.parents[1] / "output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt"


def features_worker(job):
    episodes, out_path, checkpoint = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(PACKAGE))
    import torch
    from uipc_manip.flow_policy import force_feature, obs_features
    from uipc_manip.obs import ObsSpec
    from uipc_manip.wang_bridge import ReferencePolicy

    torch.set_num_threads(1)
    policy = ReferencePolicy(checkpoint, device="cpu", yaw_deg=267.0)
    rotation = policy.rotation
    spec = ObsSpec(768)
    store = {}
    for n, ep in enumerate(episodes):
        with np.load(ep["path"], allow_pickle=False) as d:
            obs = np.asarray(d["obs"], np.float32)
            actions = np.asarray(d["actions"], np.float32)
            grip = np.asarray(d["gripper_force"], np.float32)
        T = len(actions)
        feats = np.zeros((T, 250), np.float32)
        for t in range(T):
            pos, flags, valid, _ = spec.unpack_numpy(obs[t])
            feats[t] = obs_features(policy, pos[valid], flags[valid], torch)
        force = np.stack([force_feature(grip[t], rotation) for t in range(T)])
        act = np.concatenate([actions[:, :3] @ rotation.T, actions[:, 5:6]], 1).astype(np.float32)
        store[f"e{ep['id']}_feat"], store[f"e{ep['id']}_force"], store[f"e{ep['id']}_act"] = feats, force, act
        print(f"[features] {out_path.name} {n + 1}/{len(episodes)} T={T}", flush=True)
    np.savez(out_path, **store)
    return str(out_path)


def features(a):
    from multiprocessing import get_context

    index = json.loads(a.index.read_text())
    eps = [dict(e, id=i) for i, e in enumerate(index["episodes"])]
    a.out.mkdir(parents=True, exist_ok=True)
    done = {int(k.split("_")[0][1:]) for p in a.out.glob("shard_*.npz") for k in np.load(p).files}
    eps = [e for e in eps if e["id"] not in done]
    shards = [eps[i::a.workers] for i in range(a.workers)]
    start = len(list(a.out.glob("shard_*.npz")))
    jobs = [(s, a.out / f"shard_{start + i:03d}.npz", str(a.checkpoint)) for i, s in enumerate(shards) if s]
    with get_context("spawn").Pool(len(jobs)) as pool:
        for path in pool.imap_unordered(features_worker, jobs):
            print(f"[shard] {path}", flush=True)


def load(a):
    index = json.loads(a.index.read_text())["episodes"]
    data = {}
    for p in sorted(a.features.glob("shard_*.npz")):
        with np.load(p) as z:
            for k in z.files:
                i, kind = k.split("_")
                data.setdefault(int(i[1:]), {})[kind] = z[k]
    split = {s: [i for i, e in enumerate(index) if e["split"] == s and i in data] for s in ("train", "val", "test")}
    return index, data, split


def chunks(ep, horizon):
    """Target chunk for every state: the next ``horizon`` executed actions, zero (hold) past the end."""
    act = ep["act"]
    T = len(act)
    pad = np.concatenate([act, np.zeros((horizon, act.shape[1]), np.float32)])
    idx = np.arange(T)[:, None] + np.arange(horizon)[None]
    return pad[idx]


def train(a):
    sys.path.insert(0, str(PACKAGE))
    import torch
    from uipc_manip.flow_policy import build_net

    torch.manual_seed(a.seed)
    index, data, split = load(a)
    def stack(ids):
        cond = np.concatenate([np.concatenate([data[i]["feat"], data[i]["force"]], 1) for i in ids])
        tgt = np.concatenate([chunks(data[i], a.horizon) for i in ids])
        return cond, tgt
    ctr, ttr = stack(split["train"])
    cva, tva = stack(split["val"])
    stats = dict(cond_mean=ctr.mean(0), cond_std=ctr.std(0) + 1e-3,
                 act_mean=ttr.reshape(-1, ttr.shape[-1]).mean(0), act_std=ttr.reshape(-1, ttr.shape[-1]).std(0) + 1e-3)
    print(f"[train] train states {len(ctr)} val {len(cva)}; action mean {stats['act_mean'].round(3)} std {stats['act_std'].round(3)}", flush=True)
    dev = torch.device(a.device)
    S = {k: torch.as_tensor(v, device=dev) for k, v in stats.items()}
    C = (torch.as_tensor(ctr, device=dev) - S["cond_mean"]) / S["cond_std"]
    X = (torch.as_tensor(ttr, device=dev) - S["act_mean"]) / S["act_std"]
    Cv = (torch.as_tensor(cva, device=dev) - S["cond_mean"]) / S["cond_std"]
    Xv = torch.as_tensor(tva, device=dev)
    net = build_net(torch, a.horizon, a.width, a.depth).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def sample(cond, n_steps=10):
        x = torch.randn(len(cond), a.horizon, 4, device=dev)
        for k in range(n_steps):
            t = torch.full((len(cond),), k / n_steps, device=dev)
            x = x + net(x, t, cond) / n_steps
        return x * S["act_std"] + S["act_mean"]

    def validate():
        net.eval()
        with torch.no_grad():
            pick = torch.randperm(len(Cv), device=dev)[:8192]
            pred = sample(Cv[pick]).clamp(-1, 1)
            err_first = (pred[:, 0] - Xv[pick, 0]).square().mean(0)
            err_chunk = (pred - Xv[pick]).square().mean()
            base = (Xv[pick, 0] - S["act_mean"]).square().mean(0)      # predicting the mean action
        net.train()
        return err_first.cpu().numpy(), float(err_chunk), base.cpu().numpy()

    best = None
    for step in range(1, a.steps + 1):
        i = torch.randint(0, len(C), (a.batch,), device=dev)
        x1, c = X[i], C[i]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=dev)
        xt = x0 + t[:, None, None] * (x1 - x0)
        loss = (net(xt, t, c) - (x1 - x0)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            ef, ec, base = validate()
            print(f"[train] step {step} loss {loss.item():.4f} | val first-action MSE {ef.round(4)} (mean-action baseline "
                  f"{base.round(4)}) chunk MSE {ec:.4f}", flush=True)
            if best is None or ef.sum() < best:
                best = float(ef.sum())
                a.out.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=net.state_dict(), step=step, flow_policy=dict(
                    horizon=a.horizon, width=a.width, depth=a.depth, encoder_checkpoint=str(a.checkpoint),
                    stats={k: v.tolist() for k, v in stats.items()}, index=str(a.index), val_first_mse=ef.tolist())),
                    a.out / "flow_policy.pt")
    print(f"[train] best val first-action MSE sum {best:.4f} -> {a.out / 'flow_policy.pt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("features")
    f.add_argument("--index", type=Path, required=True)
    f.add_argument("--checkpoint", type=Path, default=R1)
    f.add_argument("--workers", type=int, default=20)
    f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--index", type=Path, required=True)
    t.add_argument("--features", type=Path, required=True)
    t.add_argument("--checkpoint", type=Path, default=R1)
    t.add_argument("--horizon", type=int, default=16)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--steps", type=int, default=30000)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    features(a) if a.stage == "features" else train(a)


if __name__ == "__main__":
    main()
