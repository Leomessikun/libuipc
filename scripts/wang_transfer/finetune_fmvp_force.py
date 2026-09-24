"""Force-conditioned fine-tune of fmvp_sim.pt: the gripper load enters FMVP's FiLM layers.

Run under the `curl` Python 3.9 environment (torch_geometric; CUDA works there). The frozen-encoder
behaviour cloning (`finetune_fmvp_bc.py`) cannot tell a jammed state from a free one that looks the
same in the voxel cloud, so lookahead labels contradict the policy's own actions. The gripper load
separates them: it reproduces per decision under identical commands and is what a wrist sensor reads.

* ``clouds``: the policy-frame point cloud of every labelled state (same episode selection and
  targets as `finetune_fmvp_bc.py`), plus the force input exactly as the collector builds it for a
  profile with ``force_source="gripper"``: load times ``--force-scale``, norm-clipped to
  ``--force-clip``, exponentially averaged with ``--force-ema``, rotated into the model frame.
* ``train``: FiLM weight matrices start at zero (biases kept), so training starts from the released
  zero-force behaviour. FiLM layers and the actor trunk train; the rest of the encoder stays frozen.
  FMVP's released FiLM averages the modulation over a batch; training applies it per cloud instead,
  which is identical at the batch size of one the bridge serves.

Deploy with the collector profile printed at the end of ``train`` (``--profiles-json``).
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from finetune_fmvp_bc import PACKAGE, accepted_episodes, dagger_episodes  # noqa: E402


def force_inputs(gripper_force: np.ndarray, rotation: np.ndarray, scale: float, clip: float, ema: float) -> np.ndarray:
    """rollout_controls.force_input over a whole episode, then ours -> model frame."""
    out, previous = [], np.zeros(3, np.float32)
    for f in np.asarray(gripper_force, np.float64):
        value = f * scale
        value = value * min(1.0, clip / max(np.linalg.norm(value), 1e-12))
        previous = (ema * value + (1 - ema) * previous).astype(np.float32)
        out.append(previous @ rotation.T)
    return np.asarray(out, np.float32)


def clouds_worker(job) -> str:
    episodes, out_path, checkpoint, yaw, scale, clip, ema = job
    sys.path.insert(0, str(PACKAGE))
    import torch
    from torch_geometric.data import Batch, Data
    from uipc_manip.obs import ObsSpec
    from uipc_manip.wang_bridge import GRIPPER, ReferencePolicy, to_reference_cloud

    torch.set_num_threads(1)
    policy = ReferencePolicy(checkpoint, device="cpu", yaw_deg=yaw)
    rotation = policy.rotation.astype(np.float32)
    spec = ObsSpec(768)
    pos_all, x_all, sizes, forces, logits, targets, bodies, kinds = [], [], [], [], [], [], [], []
    for episode in episodes:
        with np.load(episode["path"], allow_pickle=False) as data:
            obs = np.asarray(data["obs"][:-1], np.float32)
            actions = np.asarray(data["actions"], np.float32)
            controller = np.asarray(data["controller_id"], np.int8)
            grip = np.asarray(data["gripper_force"], np.float32)
        f_in = force_inputs(grip, rotation, scale, clip, ema)
        labels = episode.get("labels")
        steps = range(len(actions)) if labels is None else [t for t in labels if t < len(actions)]
        for t in steps:
            pos, flags, valid, _ = spec.unpack_numpy(obs[t])
            rp, rx = to_reference_cloud(pos[valid], flags[valid], yaw_deg=yaw, voxel=policy.voxel)
            batch = Batch.from_data_list([Data(x=torch.from_numpy(rx), pos=torch.from_numpy(rp))])
            with torch.no_grad():
                encoded, _ = policy.encoder(batch, torch.zeros(1, 3))
                out = policy.trunk(encoded[batch.x[:, GRIPPER] == 1])[0]
            pos_all.append(rp); x_all.append(rx); sizes.append(len(rp))
            forces.append(f_in[t]); logits.append(out.numpy())
            if labels is not None:
                targets.append(np.clip(actions[t, :3] @ rotation.T, -1, 1)); kinds.append(6)
            elif controller[t] == 2:
                targets.append(np.zeros(3, np.float32)); kinds.append(2)
            else:
                targets.append(np.clip(actions[t, :3] @ rotation.T, -1, 1)); kinds.append(int(controller[t]))
            bodies.append(episode["body"])
        print(f"[clouds] body={episode['body']} states={len(steps)}", flush=True)
    np.savez(out_path, pos=np.concatenate(pos_all), x=np.concatenate(x_all), sizes=np.asarray(sizes),
             forces=np.stack(forces), logits=np.stack(logits), targets=np.stack(targets),
             bodies=np.asarray(bodies), kinds=np.asarray(kinds, np.int8))
    return str(out_path)


def clouds(args) -> None:
    from multiprocessing import get_context

    episodes = accepted_episodes(args.datasets) + dagger_episodes(args.dagger)
    excluded = set(args.exclude_bodies)
    episodes = [e for e in episodes if e["body"] not in excluded]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes.json").write_text(json.dumps(episodes, indent=1))
    (args.out / "force_input.json").write_text(json.dumps(dict(scale=args.force_scale, clip=args.force_clip,
                                                               ema=args.force_ema, yaw=args.yaw)))
    shards = [episodes[i::args.workers] for i in range(args.workers)]
    jobs = [(s, args.out / f"shard_{i:02d}.npz", str(args.checkpoint), args.yaw, args.force_scale,
             args.force_clip, args.force_ema) for i, s in enumerate(shards) if s]
    with get_context("spawn").Pool(len(jobs)) as pool:
        for path in pool.imap_unordered(clouds_worker, jobs):
            print(f"[shard] {path}", flush=True)
    print(f"[clouds] {len(episodes)} episodes from {len({e['body'] for e in episodes})} bodies", flush=True)


def per_cloud_film_forward(self, x, pos, batch, force_vector, visual=False):
    """FMVP's Net.forward with the FiLM modulation applied per cloud instead of batch-averaged."""
    import torch.nn.functional as F

    def film(layer, h, owner):
        gammas, betas = layer(force_vector).chunk(2, dim=1)
        return gammas[owner] * h + betas[owner]

    sa_out = (x, pos, batch)
    sa_outs = [sa_out]
    for i in range(self.num_layer):
        sa_out = self.sa_module_list[i](*sa_out)
        if i == self.num_layer - 1:
            x, pos, batch, indices = sa_out
            x = film(self.film_layers[0], x, batch)
            sa_out = x, pos, batch
        sa_outs.append(sa_out)
    x, pos_skip, batch_skip = self.fp_module_list[0](*sa_outs[-1], *sa_outs[-2])
    fp_out = film(self.film_layers[1], x, batch_skip), pos_skip, batch_skip
    for i in range(1, self.num_layer):
        x, pos_skip, batch_skip = self.fp_module_list[i](*fp_out, *sa_outs[-(i + 2)])
        fp_out = film(self.film_layers[i + 1], x, batch_skip), pos_skip, batch_skip
    x, _, _ = fp_out
    for layer in self.lin_layers:
        x = F.relu(layer(x))
    x = self.out_layer(x) if not self.residual_learning else x @ self.out_w
    return x, indices


def train(args) -> None:
    sys.path.insert(0, str(PACKAGE))
    import torch
    from torch_geometric.data import Batch, Data
    from uipc_manip.wang_bridge import GRIPPER, ReferencePolicy

    device = torch.device(args.device)
    shards = sorted(args.clouds.glob("shard_*.npz"))
    parts = [np.load(p) for p in shards]
    pos = [np.split(p["pos"], np.cumsum(p["sizes"])[:-1]) for p in parts]
    xs = [np.split(p["x"], np.cumsum(p["sizes"])[:-1]) for p in parts]
    pos = [c for part in pos for c in part]
    xs = [c for part in xs for c in part]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("forces", "logits", "targets", "bodies", "kinds")}
    n = len(pos)
    bodies, counts = np.unique(cat["bodies"], return_counts=True)
    per_body = dict(zip(bodies.tolist(), counts.tolist()))
    weight = np.array([1.0 / per_body[b] for b in cat["bodies"]], np.float32)
    weight *= np.where(cat["kinds"] == 2, args.hold_weight, 1.0) * np.where(cat["kinds"] == 6, args.dagger_weight, 1.0)
    weight /= weight.mean()
    norms = np.linalg.norm(cat["forces"], axis=1)
    print(f"[train] {n} states, {len(bodies)} bodies, lookahead labels {int((cat['kinds'] == 6).sum())}, "
          f"force input norm p50/p90/max {np.percentile(norms, 50):.2f}/{np.percentile(norms, 90):.2f}/{norms.max():.2f}", flush=True)

    policy = ReferencePolicy(args.checkpoint, device="cpu", yaw_deg=args.yaw)
    if not policy.film:
        raise ValueError("The checkpoint has no FiLM layers; use fmvp_sim.pt or fmvp_real.pt")
    net = policy.encoder.pointnet2
    net.forward = types.MethodType(per_cloud_film_forward, net)
    with torch.no_grad():
        for layer in net.film_layers:
            layer.weight.zero_()                       # start from the zero-force behaviour
    encoder, trunk = policy.encoder.to(device), policy.trunk.to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in net.film_layers.parameters():
        p.requires_grad_(True)
    params = list(net.film_layers.parameters()) + list(trunk.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    encoder.eval(); trunk.train()

    rng = np.random.default_rng(args.seed)
    val_bodies = set(rng.choice(bodies, size=max(1, len(bodies) // 8), replace=False).tolist())
    is_val = np.isin(cat["bodies"], list(val_bodies))
    train_idx, val_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    forces = torch.from_numpy(cat["forces"]).float()
    base = torch.from_numpy(cat["logits"]).float()[:, :6].tanh()
    target = torch.from_numpy(cat["targets"]).float()
    w = torch.from_numpy(weight)

    def run(idx):
        batch = Batch.from_data_list([Data(x=torch.from_numpy(xs[i]), pos=torch.from_numpy(pos[i])) for i in idx]).to(device)
        encoded, _ = encoder(batch, forces[idx].to(device))
        mu = trunk(encoded[batch.x[:, GRIPPER] == 1])[:, :6].tanh()
        b, t, ww = base[idx].to(device), target[idx].to(device), w[idx].to(device)
        fit = ((mu[:, :3] - t).square().sum(1) * ww).mean()
        trust = (mu - b).square().sum(1).mean() + 4.0 * (mu[:, 4] - b[:, 4]).square().mean()
        return fit, trust

    def evaluate(idx):
        with torch.no_grad():
            fs, ts = [], []
            for s in range(0, len(idx), 256):
                f, t = run(idx[s:s + 256]); fs.append(f.item()); ts.append(t.item())
        return float(np.mean(fs)), float(np.mean(ts))

    print(f"[train] held-out bodies {sorted(val_bodies)}: fit/trust before {evaluate(val_idx)}", flush=True)
    for epoch in range(args.epochs):
        perm = rng.permutation(train_idx)
        for s in range(0, len(perm), args.batch):
            fit, trust = run(perm[s:s + args.batch])
            loss = fit + args.trust * trust
            opt.zero_grad(); loss.backward(); opt.step()
        film_norm = float(sum(l.weight.norm() for l in net.film_layers))
        print(f"[train] epoch {epoch + 1}: train {evaluate(perm[:4096])} held-out {evaluate(val_idx)} "
              f"film |W| {film_norm:.3f}", flush=True)

    payload = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    state = dict(payload["model_state_dict"])
    for k, v in net.film_layers.state_dict().items():
        state["encoder.pointnet2.film_layers." + k] = v.detach().cpu().clone()
    for k, v in trunk.state_dict().items():
        state["trunk." + k] = v.detach().cpu().clone()
    config = json.loads((args.clouds / "force_input.json").read_text())
    profile = [dict(name="gripper_force", force_source="gripper", force_scale=config["scale"],
                    force_clip=config["clip"], force_ema=config["ema"])]
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "model_state_dict": state,
                "ipc_force_ft": dict(clouds=str(args.clouds.resolve()), states=n, heldout=sorted(val_bodies),
                                     epochs=args.epochs, lr=args.lr, trust=args.trust, profile=profile[0])},
               args.out / "fmvp_ipc_force.pt")
    (args.out / "profile.json").write_text(json.dumps(profile, indent=1) + "\n")
    print(f"[saved] {args.out / 'fmvp_ipc_force.pt'}; collector profile {args.out / 'profile.json'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    c = sub.add_parser("clouds")
    c.add_argument("--datasets", type=Path, nargs="+", required=True)
    c.add_argument("--dagger", type=Path, nargs="*", default=[])
    c.add_argument("--exclude-bodies", type=int, nargs="*", default=[])
    c.add_argument("--checkpoint", type=Path, default=Path("/home/ge47gax/Desktop/fmvp_sim.pt"))
    c.add_argument("--yaw", type=float, default=267.0)
    c.add_argument("--force-scale", type=float, default=0.01)
    c.add_argument("--force-clip", type=float, default=3.0)
    c.add_argument("--force-ema", type=float, default=0.3)
    c.add_argument("--workers", type=int, default=24)
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--clouds", type=Path, required=True)
    t.add_argument("--checkpoint", type=Path, default=Path("/home/ge47gax/Desktop/fmvp_sim.pt"))
    t.add_argument("--yaw", type=float, default=267.0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--trust", type=float, default=0.5)
    t.add_argument("--hold-weight", type=float, default=2.0)
    t.add_argument("--dagger-weight", type=float, default=3.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    clouds(a) if a.stage == "clouds" else train(a)


if __name__ == "__main__":
    main()
