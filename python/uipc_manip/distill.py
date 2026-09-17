"""Distil a visual policy from explicitly admitted teacher rollouts.

Appendix A.1 of FMVP clones the filtered trajectories into the Wang RSS 2023
segmentation-PointNet++ actor with Adam at 1e-4, batch 128, negative log
likelihood of the executed action, and 40,000 updates. This port keeps those
settings and swaps the encoder for whichever one the teacher used (the set
transformer by default here). The student is saved as a full SAC agent
checkpoint whose critic is untouched, so ``train_sac --eval-only --resume``
evaluates it without a second code path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .models import EncoderConfig
from .obs import ObsSpec
from .sac import SACAgent, SACConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dirs", type=str, nargs="+", required=True, help="Rollout directories written by uipc_manip.collect_rollouts.")
    p.add_argument("--work-dir", type=str, default="output/uipc_manip")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--teacher-checkpoint", type=str, default=None, help="Copy the actor architecture from this SAC checkpoint.")
    p.add_argument("--init-actor", type=str, default=None,
                   help="Initialize actor weights only from a compatible checkpoint; critic and optimizers stay fresh.")
    p.add_argument("--actor", choices=("wang-flow", "flat"), default="wang-flow")
    p.add_argument("--encoder", choices=("pointnet2", "transformer"), default="transformer")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--steps", type=int, default=40_000, help="Gradient updates; FMVP uses 40,000.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of kept episodes held out, split by episode.")
    p.add_argument("--validation-bodies", type=int, nargs="+", default=None,
                   help="Explicit body-disjoint split; overrides val-ratio and requires human IDs in records.")
    p.add_argument("--preload-to-device", action="store_true", help="Keep the finite demonstration arrays on the training device.")
    p.add_argument("--loss", choices=("nll", "mse", "nll_mse"), default="nll", help="nll is the paper's; use mse for bang-bang scripted data whose atanh is unbounded.")
    p.add_argument("--mse-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=10_000)
    p.add_argument("--max-train-transitions", type=int, default=0, help="Cap on training transitions loaded into memory; 0 loads every kept episode.")
    p.add_argument("--min-upperarm-ratio", type=float, default=None, help="Tighten the filter on the kept episodes to this final ratio; only kept episodes carry data, so it cannot loosen it.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    return p


def episode_ok(record: dict, *, min_upperarm_ratio: float | None) -> bool:
    if record.get("path") is None or record.get("sim_error", False) or not record.get("kept", False):
        return False
    if min_upperarm_ratio is not None:
        return float(record["final_upperarm_ratio"]) >= float(min_upperarm_ratio)
    return True


def load_dataset(source_dirs: list[str], *, val_ratio: float, seed: int, max_train_transitions: int,
                 min_upperarm_ratio: float | None, validation_bodies: list[int] | None = None):
    """Episode-level split of the kept rollouts into flat observation and action arrays."""
    episodes: list[tuple[Path, dict]] = []
    manifests: list[dict] = []
    for d in source_dirs:
        d = Path(d)
        manifests.append(json.loads((d / "manifest.json").read_text()))
        for record in json.loads((d / "episode_metrics.json").read_text()):
            if episode_ok(record, min_upperarm_ratio=min_upperarm_ratio):
                episodes.append((d / record["path"], record))
    if not episodes:
        raise SystemExit("No episode passes the filter; nothing to distil")
    dims = {(int(m["obs_dim"]), int(m["action_dim"]), int(m["point_budget"])) for m in manifests}
    if len(dims) != 1:
        raise SystemExit(f"Source rollouts disagree on observation or action layout: {sorted(dims)}")
    # Equal tensor dimensions do not establish compatible physics or camera inputs.
    ignore = {"cells", "human", "garments", "workspace", "seed", "show_viewer", "logging_level",
              "decision_watchdog", "contact_force_readout"}
    contracts = [{k: v for k, v in m.get("env", {}).items() if k not in ignore} for m in manifests]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Source environment contracts differ; use an explicit transfer experiment")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0,1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    if validation_bodies is not None:
        if any("human" not in record for _, record in episodes):
            raise ValueError("Body-disjoint splitting requires a human ID for every episode")
        held = set(validation_bodies)
        val_idx = np.asarray([k for k in order if int(episodes[k][1]["human"]) in held], dtype=int)
        train_idx = np.asarray([k for k in order if int(episodes[k][1]["human"]) not in held], dtype=int)
        if not len(val_idx) or not len(train_idx):
            raise ValueError("Body-disjoint split requires nonempty training and validation sets")
    else:
        n_val = min(len(episodes) - 1, int(round(val_ratio * len(episodes))))
        val_idx, train_idx = order[:n_val], order[n_val:]
    n_val = len(val_idx)

    def stack(indices, limit=0):
        obs, act = [], []
        total = 0
        for k in indices:
            with np.load(episodes[k][0]) as data:
                o, a = data["obs"], data["actions"]
            obs_dim, action_dim, _ = next(iter(dims))
            if (o.ndim != 2 or a.shape != (len(o), action_dim) or o.shape[1] != obs_dim
                    or not len(o) or not np.isfinite(o).all() or not np.isfinite(a).all()
                    or np.abs(a).max() > 1 + 1e-6):
                raise ValueError(f"Invalid observation/action alignment or values: {episodes[k][0]}")
            if limit > 0 and total + len(o) > limit:
                o, a = o[: limit - total], a[: limit - total]
            obs.append(o)
            act.append(a)
            total += len(o)
            if limit > 0 and total >= limit:
                break
        return np.concatenate(obs), np.concatenate(act)

    train_obs, train_act = stack(train_idx, max_train_transitions)
    val_obs, val_act = stack(val_idx) if n_val > 0 else (train_obs[: min(256, len(train_obs))], train_act[: min(256, len(train_act))])
    summary = {
        "source_dirs": [str(d) for d in source_dirs],
        "episodes": len(episodes),
        "train_episodes": int(len(train_idx)),
        "val_episodes": int(n_val),
        "train_transitions": int(len(train_obs)),
        "val_transitions": int(len(val_obs)),
        "val_from_train": bool(n_val == 0),
        "split_kind": "body" if validation_bodies is not None else "episode",
        "train_paths": [str(episodes[k][0]) for k in train_idx],
        "val_paths": [str(episodes[k][0]) for k in val_idx],
        "train_cells": sorted({(r.get("garment", "unknown"), r.get("human", -1)) for k in train_idx for r in [episodes[k][1]]}),
        "val_cells": sorted({(r.get("garment", "unknown"), r.get("human", -1)) for k in val_idx for r in [episodes[k][1]]}),
        "obs_dim": int(next(iter(dims))[0]),
        "action_dim": int(next(iter(dims))[1]),
        "point_budget": int(next(iter(dims))[2]),
    }
    return train_obs, train_act, val_obs, val_act, manifests, summary


def student_config(args, manifests: list[dict]) -> SACConfig:
    if args.teacher_checkpoint:
        return SACConfig.from_dict(SACAgent.read_checkpoint(args.teacher_checkpoint)["sac_config"])
    for m in manifests:
        if "sac_config" in m:
            return SACConfig.from_dict(m["sac_config"])
    return SACConfig(actor_type=args.actor, hidden_dim=args.hidden_dim, encoder=EncoderConfig(kind=args.encoder))


def initialize_actor(agent, checkpoint):
    """Copy only compatible actor weights; never adopt an untrained BC critic."""
    payload = SACAgent.read_checkpoint(checkpoint)
    saved = payload.get("protocol", {})
    ours = agent.protocol()
    if any(saved.get(k) != v for k, v in ours.items() if k in saved):
        raise ValueError("Actor initialization checkpoint protocol differs")
    agent.actor.load_state_dict(payload["actor"], strict=True)


def bc_loss(agent: SACAgent, obs_np, act_np, loss_kind: str, mse_weight: float, *, report: bool = True):
    import torch
    import torch.nn.functional as F

    flat = torch.as_tensor(obs_np, device=agent.device)
    target = torch.as_tensor(act_np, device=agent.device)
    batch = agent._unpack(flat)
    mu, _, _, _ = agent.actor(batch, compute_pi=False, compute_log_pi=False)
    mse = F.mse_loss(mu, target)
    metrics = {"mse": float(mse.detach().item()), "max_abs": float((mu - target).abs().max().item())} if report else {}
    if loss_kind == "mse":
        return mse, metrics
    nll = -agent.actor.action_log_prob(batch, target).mean()
    if report:
        metrics["nll"] = float(nll.detach().item())
    return (nll if loss_kind == "nll" else nll + float(mse_weight) * mse), metrics


def main(argv: list[str] | None = None) -> None:
    import torch

    args = build_parser().parse_args(argv)
    if min(args.steps, args.batch_size, args.eval_every) < 1 or args.lr <= 0:
        raise ValueError("Positive training steps, batch size, evaluation interval and learning rate required")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_obs, train_act, val_obs, val_act, manifests, summary = load_dataset(
        args.source_dirs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_train_transitions=args.max_train_transitions,
        min_upperarm_ratio=args.min_upperarm_ratio,
        validation_bodies=args.validation_bodies,
    )
    cfg = student_config(args, manifests)
    if int(cfg.history_length) != 1:
        raise SystemExit("The teacher is a history-aware policy; causal sequence behaviour cloning is not implemented, "
                         "so a student cannot be distilled from single-frame demonstrations")
    agent = SACAgent(ObsSpec(summary["point_budget"]), summary["action_dim"], cfg, args.device)
    if args.init_actor:
        initialize_actor(agent, args.init_actor)
    run_name = args.run_name or f"distill_{cfg.actor_type}_{cfg.encoder.kind}_seed{args.seed}"
    run_dir = Path(args.work_dir) / run_name
    if run_dir.exists():
        raise FileExistsError(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"stage": "expert_behavior_cloning", "dataset": summary, "args": vars(args), "sac_config": cfg.to_dict(),
                "task": "dressing", "env": manifests[0]["env"],
                "actor_initialization": "checkpoint_actor_only" if args.init_actor else "random_same_architecture",
                "critic_trained": False,
                "cells": summary["train_cells"], "heldout_cells": summary["val_cells"],
                "admission_rules": [m.get("admission_rule", "paper_filter") for m in manifests]}
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Save the optimizer actually used by BC in the SAC-format checkpoint.
    optimizer = agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=float(args.lr),
                                                        fused=agent.device.type == "cuda")
    if args.preload_to_device:
        train_obs, train_act, val_obs, val_act = [torch.as_tensor(x, device=agent.device)
                                                for x in (train_obs, train_act, val_obs, val_act)]
    generator = torch.Generator(device=agent.device).manual_seed(args.seed)
    print(f"[distill] {json.dumps(summary)}", flush=True)
    log_path = run_dir / "distill_log.csv"
    log_path.write_text("step,train_loss,train_mse,val_loss,val_mse,val_max_abs,elapsed_s\n")
    best_val = float("inf")
    t0 = time.time()

    def evaluate() -> tuple[float, dict]:
        agent.actor.eval()
        total_loss, total_mse, maximum = 0., 0., 0.
        with torch.no_grad():
            for start in range(0, len(val_obs), args.batch_size):
                o, a = val_obs[start:start + args.batch_size], val_act[start:start + args.batch_size]
                loss, metrics = bc_loss(agent, o, a, args.loss, args.mse_weight)
                total_loss += len(o) * float(loss.item())
                total_mse += len(o) * metrics["mse"]
                maximum = max(maximum, metrics["max_abs"])
        agent.actor.train()
        return total_loss / len(val_obs), {"mse": total_mse / len(val_obs), "max_abs": maximum}

    for step in range(1, int(args.steps) + 1):
        report = step == 1 or step % int(args.eval_every) == 0 or step == int(args.steps)
        idx = (torch.randint(len(train_obs), (args.batch_size,), device=agent.device, generator=generator)
               if args.preload_to_device else rng.integers(0, len(train_obs), size=args.batch_size))
        loss, metrics = bc_loss(agent, train_obs[idx], train_act[idx], args.loss, args.mse_weight, report=report)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), float(args.grad_clip))
        optimizer.step()
        if report:
            val_loss, val_metrics = evaluate()
            with log_path.open("a") as handle:
                handle.write(f"{step},{loss.item():.6f},{metrics['mse']:.6f},{val_loss:.6f},{val_metrics['mse']:.6f},{val_metrics['max_abs']:.4f},{time.time() - t0:.1f}\n")
            print(f"[distill] step={step} train_loss={loss.item():.5f} train_mse={metrics['mse']:.5f} val_loss={val_loss:.5f} val_mse={val_metrics['mse']:.5f}", flush=True)
            if val_loss < best_val:
                best_val = val_loss
                agent.save(ckpt_dir / "actor_best.pt", step, {**metadata, "val_loss": val_loss, "val_metrics": val_metrics})
        if int(args.save_every) > 0 and step % int(args.save_every) == 0:
            agent.save(ckpt_dir / f"actor_{step:07d}.pt", step, {**metadata, "best_val_loss": best_val})
    final_val, final_metrics = evaluate()
    agent.save(ckpt_dir / "actor_final.pt", int(args.steps), {**metadata, "val_loss": final_val, "val_metrics": final_metrics})
    print(f"[distill] done best_val={best_val:.5f} final_val={final_val:.5f} checkpoints={ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
