"""Bounded cloth-motion representation pretraining, inspired by PointZero.

This is not a reproduction of PointZero. It reuses the FQL actor's exact encoder
and trains its tool-point readout to support short action-conditioned material
tracks. Only that encoder transfers; the decoder, privileged query locations,
future actions and track targets never enter the deployed policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .fql import FQLAgent, FQLConfig
from .obs import EXTRA_DIM, FLAG_DEFORMABLE, ObsSpec


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track_windows(observations, actions, positions, spec, *, horizon=5,
                  queries=32, seed=17, max_distance=.03):
    """Correspond material vertices, not independently voxelized observation rows.

    At each start frame, associate visible cloth centroids with nearest material
    vertices within max_distance, deduplicate, and uniformly sample query IDs.
    Labels follow those SAME IDs through the entire window. The decoder query is
    the exact starting vertex, in the initial tool frame; the policy encoder still
    receives only the original observation. Query selection never sees the future.
    """
    from scipy.spatial import cKDTree
    observations, actions, positions = map(np.asarray, (observations, actions, positions))
    n = len(actions)
    if (horizon < 1 or queries < 1 or n < horizon or max_distance <= 0
            or observations.shape != (n, spec.dim) or positions.ndim != 3
            or positions.shape[0] != n + 1 or positions.shape[2] != 3):
        raise ValueError("Track windows require aligned observations, commands and T+1 material states")
    if any(not np.isfinite(a).all() for a in (observations, actions, positions)):
        raise ValueError("Nonfinite motion data")
    pos, feat, valid, extra = spec.unpack_numpy(observations)
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("obs", "actions", "query", "delta", "mask", "vertex_ids", "start", "mapping_distance")}
    for t in range(n - horizon + 1):
        visible = pos[t][valid[t] & (feat[t, :, FLAG_DEFORMABLE] > .5)] + extra[t, :3]
        distance, nearest = cKDTree(positions[t]).query(visible)
        accepted = distance <= max_distance
        ids = np.unique(nearest[accepted])
        if not len(ids):
            continue
        ids = rng.permutation(ids)[:queries]
        count = len(ids)
        padded = np.pad(ids, (0, queries - count), constant_values=ids[0])
        delta = positions[t + 1:t + horizon + 1, padded] - positions[t, padded][None]
        rows["obs"].append(observations[t])
        rows["actions"].append(actions[t:t + horizon])
        rows["query"].append(positions[t, padded] - extra[t, :3])
        rows["delta"].append(delta.transpose(1, 0, 2))
        rows["mask"].append(np.arange(queries) < count)
        rows["vertex_ids"].append(padded)
        rows["start"].append(t)
        rows["mapping_distance"].extend(distance[accepted].tolist())
    if not rows["obs"]:
        raise ValueError("No visible material queries")
    return {k: np.asarray(v, dtype=np.int64 if k in ("vertex_ids", "start") else np.float32)
            for k, v in rows.items()}


def build(args):
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError(args.out)
    reference = json.loads((args.rl_dataset / "manifest.json").read_text())
    spec = ObsSpec(reference["point_budget"])
    groups = {"train": [], "validation": []}
    inventory, seen = [], set()
    # Rewards are unused for prediction; permit only this explicit non-physics
    # difference and run/configuration identifiers. Everything else must match.
    excluded = {"cells", "human", "garments", "workspace", "seed", "show_viewer", "logging_level", "reward"}
    for root in args.sources:
        result = json.loads((root / "result.json").read_text())
        if not result.get("completed"):
            raise ValueError(f"Incomplete geometry rollout: {root}")
        diff = [k for k in reference["env"] if k not in excluded
                and reference["env"][k] != result["env"].get(k)]
        if diff:
            raise ValueError(f"Motion/RL environment mismatch: {diff}")
        with np.load(root / "policy.npz", allow_pickle=False) as tape:
            obs, actions, anchors = tape["observations"], tape["actions"], tape["anchors"]
        for slot, (garment, human) in enumerate(result["cells"]):
            path = root / f"positions_{slot}.npz"
            with np.load(path, allow_pickle=False) as tape:
                positions = tape["positions"]
            trace = [r for r in result["trace"] if r["slot"] == slot]
            n = len(actions)
            if (len(trace) != n or any(r.get("sim_error", False) for r in trace)
                    or [r["episode_step"] for r in trace] != list(range(1, n + 1))
                    or any(r.get("time_limit", False) for r in trace[:-1])):
                raise ValueError("Geometry tape crosses a reset, is incomplete or contains simulator errors")
            if not np.allclose(obs[:, slot, -EXTRA_DIM:-EXTRA_DIM + 3], anchors[:-1, slot], atol=1e-6):
                raise ValueError("Geometry/observation tool frames differ")
            source_id = sha256(path)
            if source_id in seen:
                raise ValueError("Duplicate material trajectory")
            seen.add(source_id)
            split = "validation" if int(human) in args.validation_bodies else "train"
            rows = track_windows(obs[:, slot], actions[:, slot], positions, spec,
                                 horizon=args.horizon, queries=args.queries,
                                 max_distance=args.max_distance, seed=args.seed + len(inventory))
            distances = rows.pop("mapping_distance")
            groups[split].append(rows)
            inventory.append(dict(source=str(root), slot=slot, garment=garment, human=int(human), split=split,
                                  sha256=source_id, policy_sha256=sha256(root / "policy.npz"),
                                  result_sha256=sha256(root / "result.json"), windows=len(rows["obs"]),
                                  mean_queries=float(rows["mask"].sum(1).mean()),
                                  mapping_distance_mean_m=float(distances.mean()),
                                  mapping_distance_max_m=float(distances.max())))
    if any(not g for g in groups.values()):
        raise ValueError("Motion data needs separate training and validation bodies")
    args.out.mkdir(parents=True)
    for split, parts in groups.items():
        np.savez_compressed(args.out / f"{split}.npz", **{
            k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    manifest = dict(format="dressing_material_windows_v1", completed=True, seed=args.seed,
                    horizon=args.horizon, queries=args.queries, point_budget=spec.point_budget,
                    action_dim=reference["action_dim"], validation_bodies=args.validation_bodies,
                    env=reference["env"], rl_dataset=str(args.rl_dataset), inventory=inventory,
                    max_mapping_distance_m=args.max_distance, seconds=time.perf_counter() - started,
                    source_collection_seconds=sum(json.loads((p / "result.json").read_text())["seconds"] for p in args.sources))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


class TrackDecoder(nn.Module):
    """Disposable decoder; geometry control gets no target coordinates/actions."""

    def __init__(self, latent_dim, action_dim, horizon, queries):
        super().__init__()
        self.horizon = horizon
        self.slots = nn.Parameter(torch.randn(queries, 16) * .02)
        self.net = nn.Sequential(nn.Linear(latent_dim + horizon * action_dim + 3 + 16, 256),
                                 nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, horizon * 3))

    def forward(self, z, commands, query, *, objective):
        b, q, _ = query.shape
        if objective == "geometry":
            commands, query = torch.zeros_like(commands), torch.zeros_like(query)
        inputs = torch.cat((z[:, None].expand(-1, q, -1),
                            commands.flatten(1)[:, None].expand(-1, q, -1),
                            query, self.slots[None].expand(b, -1, -1)), -1)
        return self.net(inputs).reshape(b, q, self.horizon, 3)


def prediction_loss(prediction, query, delta, mask, objective):
    if objective == "motion":
        error = (prediction - delta / .05).square().mean((-1, -2))
        return ((error * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
    if objective != "geometry":
        raise ValueError(objective)
    # Reconstruct the unordered CURRENT query cloud, without feeding the decoder
    # its target coordinates. All output slots participate, only valid targets do.
    pred = prediction.mean(2)
    distance = (pred[:, :, None] - query[:, None] / .5).square().sum(-1)
    forward = distance.masked_fill(~mask[:, None].bool(), torch.inf).min(2).values.mean(1)
    backward = (distance.min(1).values * mask).sum(1) / mask.sum(1).clamp_min(1)
    return .5 * (forward + backward).mean()


def initialize_actor_encoder(agent, path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != "dressing_motion_encoder_v1"
            or payload["encoder_config"] != asdict(agent.cfg.encoder)
            or payload["point_budget"] != agent.spec.point_budget
            or payload["action_dim"] != agent.action_dim):
        raise ValueError("Incompatible motion encoder checkpoint")
    agent.actor.encoder.load_state_dict(payload["encoder"], strict=True)
    return dict(path=str(path), sha256=sha256(path), objective=payload["objective"],
                steps=payload["steps"], source_seed=payload["seed"],
                validation_bodies=payload["validation_bodies"],
                training_bodies=payload["training_bodies"], env=payload["env"],
                pretraining_seconds=payload["seconds"], transfer="actor_encoder_only")


@torch.no_grad()
def measure(agent, decoder, data, objective, batch_size):
    sums = np.zeros(4)
    for start in range(0, len(data["obs"]), batch_size):
        b = {k: v[start:start + batch_size] for k, v in data.items()}
        z = agent.actor._frame_latent(agent.unpack(b["obs"]))
        pred = decoder(z, b["actions"], b["query"], objective=objective)
        loss = prediction_loss(pred, b["query"], b["delta"], b["mask"], objective)
        # Non-oracle diagnostics, averaged per window; zero motion is a physical baseline.
        if objective == "motion":
            def mde(x):
                error = (x - b["delta"]).norm(dim=-1).mean(-1)
                return ((error * b["mask"]).sum(1) / b["mask"].sum(1)).mean()
            shuffled = decoder(z, b["actions"].roll(1, 0), b["query"], objective=objective)
            values = torch.stack([loss, mde(pred * .05), mde(torch.zeros_like(pred)), mde(shuffled * .05)])
        else:
            values = torch.stack([loss, loss.new_zeros(()), loss.new_zeros(()), loss.new_zeros(())])
        sums += len(b["obs"]) * values.cpu().numpy()
    values = sums / len(data["obs"])
    names = ["loss", "track_mde_m", "zero_motion_mde_m", "shuffled_action_mde_m"]
    return dict(zip(names if objective == "motion" else names[:1], values.tolist()))


def pretrain(args):
    from .sac import SACAgent, SACConfig
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.perf_counter()
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    if not manifest.get("completed") or manifest.get("format") != "dressing_material_windows_v1":
        raise ValueError("Expected completed material-track dataset")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    base = SACConfig.from_dict(SACAgent.read_checkpoint(args.reference)["sac_config"])
    cfg = FQLConfig(encoder=base.encoder, hidden_dim=base.hidden_dim, trunk_blocks=base.trunk_blocks,
                    discount=base.discount, alpha=100.)
    agent = FQLAgent(ObsSpec(manifest["point_budget"]), manifest["action_dim"], cfg, args.device)
    decoder = TrackDecoder(agent.actor.encoder.feature_dim + EXTRA_DIM, agent.action_dim,
                           manifest["horizon"], manifest["queries"]).to(agent.device)
    optimizer = torch.optim.Adam(list(agent.actor.encoder.parameters()) + list(decoder.parameters()),
                                 lr=args.learning_rate, fused=agent.device.type == "cuda")
    data = {}
    for split in ("train", "validation"):
        with np.load(args.dataset / f"{split}.npz", allow_pickle=False) as tape:
            data[split] = {k: torch.as_tensor(tape[k], device=agent.device)
                           for k in ("obs", "actions", "query", "delta", "mask")}
    args.out.mkdir(parents=True)
    result = dict(completed=False, objective=args.objective, seed=args.seed, steps=args.steps,
                  dataset=str(args.dataset), dataset_manifest_sha256=sha256(args.dataset / "manifest.json"),
                  train_windows=len(data["train"]["obs"]), validation_windows=len(data["validation"]["obs"]),
                  batch_size=args.batch_size, learning_rate=args.learning_rate,
                  initial={s: measure(agent, decoder, d, args.objective, args.batch_size) for s, d in data.items()})
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            idx = torch.randint(len(data["train"]["obs"]), (args.batch_size,), device=agent.device)
            b = {k: v[idx] for k, v in data["train"].items()}
            z = agent.actor._frame_latent(agent.unpack(b["obs"]))
            pred = decoder(z, b["actions"], b["query"], objective=args.objective)
            loss = prediction_loss(pred, b["query"], b["delta"], b["mask"], args.objective)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                row = dict(step=step, train_loss=float(loss.detach()), seconds=time.perf_counter() - started,
                           validation=measure(agent, decoder, data["validation"], args.objective, args.batch_size))
                if not np.isfinite([row["train_loss"], *row["validation"].values()]).all():
                    raise FloatingPointError(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
    result.update(completed=True, final={s: measure(agent, decoder, d, args.objective, args.batch_size)
                                        for s, d in data.items()}, seconds=time.perf_counter() - started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated() if agent.device.type == "cuda" else 0)
    payload = dict(format="dressing_motion_encoder_v1", encoder=agent.actor.encoder.state_dict(),
                   encoder_config=asdict(agent.cfg.encoder), point_budget=agent.spec.point_budget,
                   action_dim=agent.action_dim, objective=args.objective, steps=args.steps, seed=args.seed,
                   seconds=result["seconds"], validation_bodies=manifest["validation_bodies"], env=manifest["env"],
                   training_bodies=sorted({r["human"] for r in manifest["inventory"] if r["split"] == "train"}),
                   decoder=decoder.state_dict(), optimizer=optimizer.state_dict(),
                   dataset_manifest_sha256=result["dataset_manifest_sha256"])
    torch.save(payload, args.out / "encoder.pt")
    result["seconds"] = time.perf_counter() - started
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", type=Path, nargs="+", required=True)
    b.add_argument("--rl-dataset", type=Path, required=True)
    b.add_argument("--validation-bodies", type=int, nargs="+", default=[14048, 14049])
    b.add_argument("--horizon", type=int, default=5)
    b.add_argument("--queries", type=int, default=32)
    b.add_argument("--max-distance", type=float, default=.03)
    t = sub.add_parser("pretrain")
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--reference", type=Path, required=True)
    t.add_argument("--objective", choices=["motion", "geometry"], required=True)
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--device", default="cuda")
    for parser in (b, t):
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    if args.mode == "pretrain" and min(args.steps, args.batch_size, args.log_every) < 1:
        p.error("Positive training counts required")
    (build if args.mode == "build" else pretrain)(args)


if __name__ == "__main__":
    main()
