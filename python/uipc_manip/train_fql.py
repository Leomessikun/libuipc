"""Train/evaluate the dressing FQL reference on explicit IPC transitions.

Training and native evaluation are separate processes: Genesis must initialize
before any CUDA matrix work. Existing SAC checkpoints provide configuration,
never learned weights or optimizer state, to a fresh FQL learner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .fql import FQLAgent, FQLConfig
from .obs import ObsSpec


def load_transitions(source, validation_bodies):
    """Episode/body split; failures retained, reset successors never fabricated."""
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest.get("completed") or manifest.get("transition_schema") != "explicit_successors_v1":
        raise ValueError("FQL requires a completed dataset with explicit successor observations")
    records = json.loads((source / "episode_metrics.json").read_text())
    groups = {"train": [], "validation": []}
    inventory, source_splits = [], {}
    for record in records:
        split = "validation" if int(record["human"]) in validation_bodies else "train"
        source_id = record.get("source_sha256", record["path"])
        if source_id in source_splits and source_splits[source_id] != split:
            raise ValueError("Repeated source episode appears on both sides of the split")
        source_splits[source_id] = split
        path = source / record["path"]
        with np.load(path, allow_pickle=False) as tape:
            obs, nxt = tape["obs"], tape["next_obs"]
            actions, rewards = tape["actions"], tape["rewards"][:, None]
            metrics = [json.loads(str(x)) for x in tape["step_metrics"]]
        n = len(actions)
        if obs.shape != nxt.shape or obs.shape != (n, manifest["obs_dim"]) or len(metrics) != n:
            raise ValueError(f"Inconsistent transition shapes: {path}")
        if actions.shape != (n, manifest["action_dim"]) or rewards.shape != (n, 1):
            raise ValueError(f"Inconsistent action/reward shapes: {path}")
        # Native sim errors return reset observations; exclude that row and any suffix.
        error = next((i for i, m in enumerate(metrics) if m.get("sim_error", False)), n)
        if any(m.get("time_limit", False) for m in metrics[:max(0, error - 1)]):
            raise ValueError(f"A tape crosses an episode boundary: {path}")
        arrays = (obs[:error], actions[:error], rewards[:error], nxt[:error], np.ones((error, 1), np.float32))
        if any(not np.isfinite(a).all() for a in arrays) or (np.abs(actions[:error]) > 1.00001).any():
            raise ValueError(f"Invalid transition values: {path}")
        if error:
            groups[split].append(arrays)
        inventory.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              source_id=source_id, split=split, rows=error,
                              garment=record["garment"], human=record["human"],
                              successful=bool(record.get("kept", False))))
    if any(not rows for rows in groups.values()):
        raise ValueError("Both body-disjoint training and validation data are required")
    arrays = {name: tuple(np.concatenate(parts).astype(np.float32) for parts in zip(*rows, strict=True))
              for name, rows in groups.items()}
    return arrays, manifest, inventory


@torch.no_grad()
def validation(agent, arrays, batch_size):
    sums = torch.zeros(4, device=agent.device)
    count = 0
    # Validation cannot change training's random stream or choose individual noisy actions.
    devices = [agent.device.index or 0] if agent.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(9301)
        for start in range(0, len(arrays[0]), batch_size):
            flat, actions, rewards, nxt, masks = (a[start:start + batch_size] for a in arrays)
            obs = agent.unpack(flat)
            noise = torch.randn_like(actions)
            prior = agent.flow_actions(agent.behavior._frame_latent(obs), noise)
            actor = agent.actor(obs, noise).clamp(-1, 1)
            q1, q2 = agent.critic(obs, actions)
            q = .5 * (q1 + q2)
            values = torch.stack(((prior - actions).square().mean(),
                                  (actor - actions).square().mean(),
                                  (prior - actor).square().mean(), q.mean()))
            sums += len(actions) * values
            count += len(actions)
    return dict(zip(("prior_action_mse", "actor_action_mse", "actor_prior_mse", "data_q"),
                    (sums / count).cpu().tolist(), strict=True))


def train(args):
    from .sac import SACAgent, SACConfig
    started = time.perf_counter()
    if args.out.exists():
        raise FileExistsError("Use a new run directory, including when resuming")
    arrays, manifest, inventory = load_transitions(args.dataset, set(args.validation_bodies))
    payload = SACAgent.read_checkpoint(args.reference)
    base = SACConfig.from_dict(payload["sac_config"])
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    cfg = FQLConfig(encoder=base.encoder, hidden_dim=base.hidden_dim,
                    trunk_blocks=base.trunk_blocks, discount=base.discount,
                    alpha=args.alpha, learning_rate=args.learning_rate)
    identity = dict(dataset=str(args.dataset), validation_bodies=args.validation_bodies,
                    inventory=inventory, env=manifest["env"], reference=str(args.reference))
    if args.resume:
        agent, old = FQLAgent.load(args.resume, args.device, resume=True)
        if any(old[k] != identity[k] for k in identity) or agent.cfg != cfg:
            raise ValueError("Resume data, environment or FQL configuration differs")
    else:
        agent = FQLAgent(ObsSpec(manifest["point_budget"]), manifest["action_dim"], cfg, args.device)
    encoder_init = None
    if args.encoder_init:
        from .motion_pretrain import initialize_actor_encoder
        encoder_init = initialize_actor_encoder(agent, args.encoder_init)
        if (encoder_init["validation_bodies"] != args.validation_bodies
                or set(encoder_init["training_bodies"]) & set(args.validation_bodies)
                or encoder_init["env"] != manifest["env"]):
            raise ValueError("Encoder pretraining violates the RL environment/body split")
    elif args.resume:
        encoder_init = old.get("encoder_initialization")
    data = {name: tuple(torch.as_tensor(a, device=agent.device) for a in rows) for name, rows in arrays.items()}
    args.out.mkdir(parents=True)
    metadata = {**identity, "seed": args.seed, "fresh_weights": not bool(args.resume or args.encoder_init),
                "fresh_rl_optimizer": not bool(args.resume),
                "encoder_initialization": encoder_init,
                "resumed_from": str(args.resume) if args.resume else None,
                "train_rows": len(data["train"][0]), "validation_rows": len(data["validation"][0]),
                "preparation_seconds": time.perf_counter() - started,
                "source_preparation_seconds": manifest.get("seconds"),
                "reward_scale": 1.0, "heldout_role": "development bodies, not untouched final test",
                "config": vars(args) | {"out": str(args.out), "dataset": str(args.dataset),
                                         "reference": str(args.reference), "resume": str(args.resume) if args.resume else None,
                                         "encoder_init": str(args.encoder_init) if args.encoder_init else None}}
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: metadata[k] for k in ("train_rows", "validation_rows", "preparation_seconds")}), flush=True)
    learning_started = time.perf_counter()
    last = {}
    with (args.out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            idx = torch.randint(len(data["train"][0]), (args.batch_size,), device=agent.device)
            stats = agent.update(tuple(a[idx] for a in data["train"]))
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                values = torch.stack(list(stats.values())).cpu().tolist()
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite FQL update {step}: {values}")
                last = dict(step=agent.updates, train=dict(zip(stats, values, strict=True)),
                            elapsed_seconds=time.perf_counter() - learning_started)
                if step == 1 or step % args.validation_every == 0 or step == args.steps:
                    last["validation"] = validation(agent, data["validation"], args.batch_size)
                    if not np.isfinite(list(last["validation"].values())).all():
                        raise FloatingPointError("Nonfinite validation metrics")
                    agent.save(args.out / f"checkpoint_{agent.updates:06d}.pt", metadata)
                log.write(json.dumps(last) + "\n")
                log.flush()
                print(json.dumps(last), flush=True)
    agent.save(args.out / "final.pt", metadata)
    result = dict(completed=True, updates=agent.updates, requested_updates=args.steps,
                  learning_seconds=time.perf_counter() - learning_started,
                  total_seconds=time.perf_counter() - started,
                  peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(agent.device) if agent.device.type == "cuda" else 0,
                  last=last)
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def evaluate(args):
    from . import train_sac
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_live import LiveCellFactory
    from .expert_baseline import run_world
    from .sac import SACAgent
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "dressing_fql_v1":
        raise ValueError("Evaluation requires a FQL checkpoint")
    metadata = payload["metadata"]
    reference = SACAgent.read_checkpoint(metadata["reference"])
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], reference)
    train_sac.resolve_defaults(targs)
    targs._resume_env = metadata["env"]
    cfg = train_sac.restore_env_config(train_sac.dressing_config(targs), targs)
    cells = tuple((g, int(b)) for g, b in (c.rsplit(":", 1) for c in args.cells))
    cfg = replace(cfg, cells=cells, workspace=str(args.out / "assets"), decision_watchdog=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, len(cells), LiveCellFactory(cfg.live))
    agent, _ = FQLAgent.load(args.checkpoint, args.device)
    agent.modules.eval()
    args.out.mkdir(parents=True, exist_ok=True)
    result = dict(checkpoint=str(args.checkpoint), cells=cells, env=cfg.to_dict(),
                  build_seconds=time.perf_counter() - started, rounds=args.rounds,
                  policies=args.policies, evaluations=[], completed=False)
    try:
        for rep in range(args.rounds):
            for policy in args.policies[::1 if rep % 2 == 0 else -1]:
                torch.manual_seed(args.seed + rep)
                before = time.perf_counter()
                dest = args.out / f"{policy}_round{rep}"
                callback = None if policy == "scripted" else lambda obs: agent.act(obs, behavior=policy == "behavior")
                records = run_world(env, cells, cfg.horizon, args.seed + rep * 100, dest, 0,
                                    save_observations=False, policy=callback)
                valid = [not r["sim_error"] and r["final_upperarm_ratio"] >= cfg.reward.success_upperarm_ratio
                         and r["max_tracking_error"] <= env.grasp_tracking_tolerance_m for r in records]
                row = dict(policy=policy, round=rep, records=records, seconds=time.perf_counter() - before,
                           valid_successes=sum(valid), episodes=len(records),
                           mean_coverage=float(np.mean([r["final_upperarm_ratio"] for r in records])))
                result["evaluations"].append(row)
                result["total_seconds"] = time.perf_counter() - started
                (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items() if k != "records"}), flush=True)
        result["completed"] = True
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    finally:
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--reference", type=Path, required=True)
    initialization = t.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--encoder-init", type=Path,
                                help="Initialize only the actor encoder from motion/geometry pretraining")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--validation-bodies", type=int, nargs="+", default=[14048, 14049])
    t.add_argument("--alpha", type=float, default=100.)
    t.add_argument("--learning-rate", type=float, default=3e-4)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--validation-every", type=int, default=500)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--cells", nargs="+", default=["tshirt_26:14046", "tshirt_68:14046", "tshirt_26:14048", "tshirt_68:14049"])
    e.add_argument("--rounds", type=int, default=2)
    e.add_argument("--policies", nargs="+", choices=["actor", "behavior", "scripted"], default=["actor", "behavior"])
    for parser in (t, e):
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--seed", type=int, default=17)
    args = p.parse_args(argv)
    if args.mode == "train":
        if min(args.steps, args.batch_size, args.log_every, args.validation_every) < 1:
            p.error("Training counts must be positive")
        if args.validation_every % args.log_every:
            p.error("validation-every must be a multiple of log-every")
        train(args)
    else:
        if args.rounds < 1:
            p.error("rounds must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
