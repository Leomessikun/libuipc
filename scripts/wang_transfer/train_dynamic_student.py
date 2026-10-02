"""Cache frozen r1 features, then fit matched history/current-only students on CPU.

Run with curl's Python. Training uses all query labels, including failed task
episodes; it never filters by success. Non-query states anchor the released
r1 actions, with zero targets for the shared completion hold. Invalid-physics
episodes are excluded. Validation is by explicit whole body IDs, not frames.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from dynamic_student import adapter, encode, sha256
from motion_observation import history_inputs

PACKAGE = Path("/home/ge47gax/kun/libuipc/.claude/worktrees/residual-rl/python")
BASE = Path("/home/ge47gax/kun/libuipc/output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt")


def features(args):
    sys.path.insert(0, str(args.package_root))
    import torch
    from uipc_manip.wang_bridge import ReferencePolicy
    from uipc_manip.obs import ObsSpec
    torch.set_num_threads(1)
    policy = ReferencePolicy(args.base, device="cpu", yaw_deg=args.yaw)
    spec = ObsSpec(768)
    args.out.mkdir(parents=True, exist_ok=False)
    records, seen = [], set()
    for directory in args.episodes:
        run = json.loads((directory / "run.json").read_text())
        if run["checkpoint_sha256"] != sha256(args.base):
            raise ValueError(f"Mixed base checkpoints in {directory}")
        for metric in json.loads((directory / "metrics.json").read_text()):
            condition = metric["condition"]
            if condition != "observed":
                continue
            failure = metric.get("failure")
            if failure and failure["kind"] == "invalid_physics":
                raise ValueError(f"Invalid physics episode requires diagnosis: {directory}")
            source = directory / f"{condition}_trajectory.npz"
            content_hash = sha256(source)
            if content_hash in seen:
                continue
            seen.add(content_hash)
            with np.load(source, allow_pickle=False) as d:
                obs = d["obs"][:-1].copy()
                query = d["queried"].copy()
                targets = np.where(query[:, None], d["teacher_actions"], d["nominal_actions"])
                holds = d["completion_hold"].copy()
                targets[holds] = 0.
            if len(obs) != len(targets) or not query.any():
                raise ValueError(f"No aligned teacher labels in {source}")
            feats, logits, tools = [], [], []
            for observation in obs:
                pos, flags, valid, extra = spec.unpack_numpy(observation)
                feat, logit = encode(policy, pos[valid], flags[valid])
                feats.append(feat); logits.append(logit); tools.append(extra[:3] @ policy.rotation.T)
            file = args.out / f"episode_{len(records):04d}.npz"
            np.savez_compressed(file, features=np.asarray(feats), logits=np.asarray(logits), tools=np.asarray(tools),
                                targets=targets, queried=query, holds=holds)
            record = dict(path=str(file.resolve()), source=str(source.resolve()), sha256=content_hash,
                          body=run["motion"]["body_id"], motion_source_sha256=run["motion"]["source_sha256"],
                          subject=run["motion"]["source_info"]["sbj_id"], frames=len(obs), queries=int(query.sum()),
                          final_success=metric["final_success"], data_mode=run["data_mode"])
            records.append(record)
            print("[features] " + json.dumps(record), flush=True)
    if not records:
        raise ValueError("No dynamic teacher episodes")
    manifest = dict(episodes=records, base_checkpoint=str(args.base.resolve()), base_sha256=sha256(args.base),
                    yaw=args.yaw, feature_dim=50, feature_rng_seed=0,
                    extraction_sha256=sha256(Path(__file__).with_name("dynamic_student.py")))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def train(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    manifest = json.loads((args.features / "manifest.json").read_text())
    val_bodies = set(args.validation_bodies)
    parts = {k: [] for k in ("x", "logits", "targets", "query", "val", "episode")}
    for i, record in enumerate(manifest["episodes"]):
        with np.load(record["path"], allow_pickle=False) as d:
            x = history_inputs(d["features"], d["tools"], args.frames, current_only=args.current_only)
            parts["x"].append(x); parts["logits"].append(d["logits"])
            parts["targets"].append(d["targets"]); parts["query"].append(d["queried"])
            parts["val"].append(np.full(len(x), record["body"] in val_bodies))
            parts["episode"].append(np.full(len(x), i))
    data = {k: np.concatenate(v) for k, v in parts.items()}
    train_index, val_index = np.flatnonzero(~data["val"]), np.flatnonzero(data["val"])
    if min(len(train_index), len(val_index)) == 0:
        raise ValueError("Explicit disjoint training and validation bodies are required")
    # Equal episode contribution; equal relative emphasis on correction labels.
    weight = np.ones(len(data["x"]), np.float32)
    for ep in np.unique(data["episode"]):
        mask = data["episode"] == ep
        weight[mask] = 1. / mask.sum()
    weight *= np.where(data["query"], args.query_weight, 1.)
    weight /= weight[train_index].mean()
    x = torch.from_numpy(data["x"]).float()
    base = torch.from_numpy(data["logits"]).float()
    target = torch.from_numpy(data["targets"]).float()
    weights = torch.from_numpy(weight)
    sys.path.insert(0, str(args.package_root))
    from uipc_manip.wang_bridge import up_axis_rotation
    rotation = torch.from_numpy(up_axis_rotation(manifest["yaw"])).float()
    model = adapter(args.frames)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def losses(index):
        residual = model(x[index])
        action = (base[index] + residual).tanh()
        world = torch.cat((action[:, :3] @ rotation, action[:, 3:] @ rotation), dim=1).clamp(-1, 1)
        error = (world - target[index]).square().mean(1)
        fit = (error * weights[index]).mean()
        trust = residual.square().mean()
        return fit, trust, error

    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "train.jsonl").open("w") as log:
        for update in range(args.updates + 1):
            if update:
                index = rng.choice(train_index, args.batch, replace=len(train_index) < args.batch)
                fit, trust, _ = losses(index)
                loss = fit + args.trust * trust
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite student loss")
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            if update % 100 == 0 or update == args.updates:
                with torch.no_grad():
                    train_fit, _, _ = losses(train_index)
                    val_fit, _, error = losses(val_index)
                    query_error = error[torch.from_numpy(data["query"][val_index])].mean()
                row = dict(update=update, train_fit=float(train_fit), validation_fit=float(val_fit),
                           validation_query_mse=float(query_error))
                if not np.isfinite(list(row.values())).all():
                    raise FloatingPointError("Invalid validation metric or missing validation queries")
                log.write(json.dumps(row) + "\n"); log.flush()
                print("[train] " + json.dumps(row), flush=True)
    config = dict(base_checkpoint=manifest["base_checkpoint"], base_sha256=manifest["base_sha256"],
                  yaw=manifest["yaw"], frames=args.frames, current_only=args.current_only, seed=args.seed,
                  updates=args.updates, batch=args.batch, learning_rate=args.lr, trust=args.trust,
                  query_weight=args.query_weight, train_states=len(train_index), val_states=len(val_index),
                  train_queries=int(data["query"][train_index].sum()), validation_bodies=sorted(val_bodies),
                  feature_manifest_sha256=sha256(args.features / "manifest.json"),
                  train_code_sha256=sha256(__file__), runtime_sha256=sha256(Path(__file__).with_name("dynamic_student.py")),
                  parameters=sum(p.numel() for p in model.parameters()), policy="frozen r1 + history residual adapter",
                  selection="fixed update budget; no test-set checkpoint selection")
    torch.save(dict(dynamic_student=config, adapter_state_dict=model.state_dict()), args.out / "student.pt")
    # This deployment-equivalent r1 control isolates fixed FPS / bridge effects
    # from learned corrections. Its action adapter is exactly zero.
    torch.save(dict(dynamic_student={**config, "updates": 0, "policy": "r1 with zero adapter and fixed FPS"},
                    adapter_state_dict=initial_state), args.out / "initial_student.pt")
    (args.out / "manifest.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    f = sub.add_parser("features")
    f.add_argument("--episodes", type=Path, nargs="+", required=True)
    f.add_argument("--base", type=Path, default=BASE)
    f.add_argument("--yaw", type=float, default=267.)
    t = sub.add_parser("train")
    t.add_argument("--features", type=Path, required=True)
    t.add_argument("--validation-bodies", type=int, nargs="+", required=True)
    t.add_argument("--frames", type=int, default=4)
    t.add_argument("--current-only", action="store_true")
    t.add_argument("--updates", type=int, default=1000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--trust", type=float, default=.01)
    t.add_argument("--query-weight", type=float, default=3.)
    t.add_argument("--seed", type=int, default=0)
    for command in (f, t):
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--package-root", type=Path, default=PACKAGE)
    a = p.parse_args()
    if a.stage == "features":
        features(a)
    else:
        train(a)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
