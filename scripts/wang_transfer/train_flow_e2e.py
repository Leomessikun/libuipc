"""Train the end-to-end flow policy (``uipc_manip.flow_policy.build_e2e_net``) on the body-split rollout index.

Run under the ``curl`` environment. Two stages:

* ``prepare``: every state of every indexed episode as the model-frame cloud the policy serves on
  (``e2e_cloud``: 2.5 cm voxels, at most 512 points, float16), with the gripper load and the executed
  action in the model frame; one ``.npz`` shard per worker.
* ``train``: rectified-flow training of the action chunk, with the point encoder trained jointly (not the
  frozen FMVP encoder of ``train_flow_policy.py``). Augmentation per sample: a turn about the vertical of
  up to ``--aug-yaw`` degrees applied to the cloud and the chunk's translations, 3 mm point noise, and 10 %
  point dropout (the gripper point is kept). Validation and checkpoint selection as in ``train_flow_policy``.
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


def prepare_worker(job):
    episodes, out_path, voxel, points = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(PACKAGE))
    from uipc_manip.flow_policy import e2e_cloud, force_feature
    from uipc_manip.obs import ObsSpec
    from uipc_manip.wang_bridge import up_axis_rotation

    rotation = up_axis_rotation(267.0)
    spec = ObsSpec(768)
    store = {}
    for n, ep in enumerate(episodes):
        with np.load(ep["path"], allow_pickle=False) as d:
            obs = np.asarray(d["obs"], np.float32)
            actions = np.asarray(d["actions"], np.float32)
            grip = np.asarray(d["gripper_force"], np.float32)
        T = len(actions)
        clouds = np.zeros((T, points, 6), np.float16)
        for t in range(T):
            pos, flags, valid, _ = spec.unpack_numpy(obs[t])
            clouds[t] = e2e_cloud(pos[valid], flags[valid], 267.0, voxel, points)[0]
        force = np.stack([force_feature(grip[t], rotation) for t in range(T)])
        act = np.concatenate([actions[:, :3] @ rotation.T, actions[:, 5:6]], 1).astype(np.float32)
        store[f"e{ep['id']}_cloud"], store[f"e{ep['id']}_force"], store[f"e{ep['id']}_act"] = clouds, force, act
        print(f"[prepare] {out_path.name} {n + 1}/{len(episodes)} T={T}", flush=True)
    np.savez(out_path, **store)
    return str(out_path)


def prepare(a):
    from multiprocessing import get_context

    index = json.loads(a.index.read_text())
    eps = [dict(e, id=i) for i, e in enumerate(index["episodes"])]
    a.out.mkdir(parents=True, exist_ok=True)
    done = {int(k.split("_")[0][1:]) for p in a.out.glob("shard_*.npz") for k in np.load(p).files}
    eps = [e for e in eps if e["id"] not in done]
    shards = [eps[i::a.workers] for i in range(a.workers)]
    start = len(list(a.out.glob("shard_*.npz")))
    jobs = [(s, a.out / f"shard_{start + i:03d}.npz", a.voxel, a.points) for i, s in enumerate(shards) if s]
    with get_context("spawn").Pool(len(jobs)) as pool:
        for path in pool.imap_unordered(prepare_worker, jobs):
            print(f"[shard] {path}", flush=True)


def load(a):
    index = json.loads(a.index.read_text())["episodes"]
    data = {}
    for p in sorted(a.data.glob("shard_*.npz")):
        with np.load(p) as z:
            for k in z.files:
                i, kind = k.split("_")
                data.setdefault(int(i[1:]), {})[kind] = z[k]
    split = {s: [i for i, e in enumerate(index) if e["split"] == s and i in data] for s in ("train", "val", "test")}
    if a.train_fraction < 1.0:
        bodies = sorted({index[i]["body"] for i in split["train"]})
        rng = np.random.default_rng(a.seed)
        keep = set(rng.choice(bodies, max(1, round(a.train_fraction * len(bodies))), replace=False).tolist())
        split["train"] = [i for i in split["train"] if index[i]["body"] in keep]
    return index, data, split


def chunks(act, horizon):
    T = len(act)
    pad = np.concatenate([act, np.zeros((horizon, act.shape[1]), np.float32)])
    return pad[np.arange(T)[:, None] + np.arange(horizon)[None]]


def train(a):
    sys.path.insert(0, str(PACKAGE))
    import torch
    from uipc_manip.flow_policy import build_e2e_net

    torch.manual_seed(a.seed)
    index, data, split = load(a)
    dev = torch.device(a.device)

    def stack(ids):
        cloud = np.concatenate([data[i]["cloud"] for i in ids])
        force = np.concatenate([data[i]["force"] for i in ids])
        tgt = np.concatenate([chunks(data[i]["act"], a.horizon) for i in ids])
        return cloud, force, tgt

    ctr, ftr, ttr = stack(split["train"])
    cva, fva, tva = stack(split["val"])
    stats = dict(act_mean=ttr.reshape(-1, 4).mean(0), act_std=ttr.reshape(-1, 4).std(0) + 1e-3,
                 force_mean=ftr.mean(0), force_std=ftr.std(0) + 1e-3)
    print(f"[train] train states {len(ctr)} val {len(cva)}; clouds {ctr.nbytes / 1e9:.1f} GB", flush=True)
    S = {k: torch.as_tensor(v, device=dev) for k, v in stats.items()}
    C, F, X = (torch.as_tensor(v, device=dev) for v in (ctr, ftr, ttr))
    Cv, Fv, Xv = (torch.as_tensor(v, device=dev) for v in (cva, fva, tva))
    net = build_e2e_net(torch, a.horizon, a.width, a.depth).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    def batch(Cs, Fs, Xs, i, augment):
        cloud = Cs[i].float()
        mask = cloud[..., 3:].sum(-1) > 0.5
        act = Xs[i].clone()
        force = Fs[i].clone()
        if augment:
            ang = (torch.rand(len(i), device=dev) * 2 - 1) * np.radians(a.aug_yaw)
            c, s = torch.cos(ang), torch.sin(ang)
            R = torch.zeros(len(i), 3, 3, device=dev)            # about the model frame's vertical (y)
            R[:, 0, 0], R[:, 0, 2], R[:, 1, 1], R[:, 2, 0], R[:, 2, 2] = c, s, 1, -s, c
            cloud[..., :3] = torch.einsum("bij,bpj->bpi", R, cloud[..., :3])
            act[..., :3] = torch.einsum("bij,bhj->bhi", R, act[..., :3])
            force = torch.einsum("bij,bj->bi", R, force)      # the gripper load is a model-frame vector too
            grip = cloud[..., 5] > 0.5
            cloud[..., :3] += torch.randn_like(cloud[..., :3]) * 0.003 * (~grip)[..., None]
            mask = mask & ((torch.rand(mask.shape, device=dev) > 0.1) | grip)
        force = (force - S["force_mean"]) / S["force_std"]
        return cloud, mask, force, (act - S["act_mean"]) / S["act_std"]

    def validate():
        net.eval()
        with torch.no_grad():
            pick = torch.randperm(len(Cv), device=dev)[:4096]
            cloud, mask, force, _ = batch(Cv, Fv, Xv, pick, False)
            x = torch.randn(len(pick), a.horizon, 4, device=dev)
            for k in range(10):
                t = torch.full((len(pick),), k / 10, device=dev)
                x = x + net(x, t, cloud, mask, force) / 10
            pred = (x * S["act_std"] + S["act_mean"]).clamp(-1, 1)
            err = (pred[:, 0] - Xv[pick, 0]).square().mean(0)
            base = (Xv[pick, 0] - S["act_mean"]).square().mean(0)
        net.train()
        return err.cpu().numpy(), base.cpu().numpy()

    best = None
    for step in range(1, a.steps + 1):
        i = torch.randint(0, len(C), (a.batch,), device=dev)
        cloud, mask, force, x1 = batch(C, F, X, i, True)
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=dev)
        xt = x0 + t[:, None, None] * (x1 - x0)
        loss = (net(xt, t, cloud, mask, force) - (x1 - x0)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % a.eval_every == 0 or step == a.steps:
            err, base = validate()
            print(f"[train] step {step} loss {loss.item():.4f} | val first-action MSE {err.round(4)} "
                  f"(mean-action baseline {base.round(4)})", flush=True)
            if best is None or err.sum() < best:
                best = float(err.sum())
                a.out.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=net.state_dict(), step=step, flow_e2e=dict(
                    horizon=a.horizon, width=a.width, depth=a.depth, voxel=a.voxel, points=a.points,
                    stats={k: v.tolist() for k, v in stats.items()}, index=str(a.index), val_first_mse=err.tolist())),
                    a.out / "flow_e2e.pt")
    print(f"[train] best val first-action MSE sum {best:.4f} -> {a.out / 'flow_e2e.pt'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("prepare")
    f.add_argument("--index", type=Path, required=True)
    f.add_argument("--workers", type=int, default=16)
    f.add_argument("--voxel", type=float, default=0.025)
    f.add_argument("--points", type=int, default=512)
    f.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--index", type=Path, required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--voxel", type=float, default=0.025)
    t.add_argument("--points", type=int, default=512)
    t.add_argument("--horizon", type=int, default=16)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--steps", type=int, default=60000)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--aug-yaw", type=float, default=10.0)
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--train-fraction", type=float, default=1.0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prepare(a) if a.stage == "prepare" else train(a)


if __name__ == "__main__":
    main()
