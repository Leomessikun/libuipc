"""Step 0 of the training-infrastructure proposal: the expert's region baseline and its demonstrations.

Two things nobody has measured come out of one run.

* **The bar.** What the scripted expert scores on a region under the evaluation metric, the final
  upper-arm ratio of :func:`~uipc_manip.cellplan.summarize`, on the held-out configurations the
  teacher is scored on and on the training ones it learns from. The 22 of 40 on record is not that
  number: it is the highest reading of an episode, on bodies 0 to 7, which are not a region's poses.
* **The fuel.** Every decision's privileged state, action and reward, one file per episode, which
  seed the demonstration half of an expert-anchored teacher's replay. Failed episodes are kept too:
  their rewards are what a critic learns the task's shape from.

Records carry the fields :func:`uipc_manip.train_sac.evaluate` writes, so the summary here and an
evaluation round of ``pretrain_wang`` are the same numbers.

Usage (inside the Genesis environment, from the repository root)::

    PYTHONPATH=python python -m uipc_manip.expert_baseline --region 13 --poses heldout
    PYTHONPATH=python python -m uipc_manip.expert_baseline --region 13 --poses train

Every unrecognised flag is passed to ``pretrain_wang teacher``, so the worlds, the horizon and the
physics are the teacher's, whose bar this run measures.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import pretrain_wang
from .curriculum import WANG_GARMENT_ORDER
from .dressing_env import GenesisIPCDressingEnv
from .dressing_live import LiveCellFactory
from .genesis_env import _ensure_genesis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the scripted expert over a region and record its episodes.")
    p.add_argument("--region", type=int, required=True, help="The arm-pose region (0-26) to measure.")
    p.add_argument("--poses", choices=("heldout", "train", "all"), default="heldout",
                   help="Held-out poses 45-49 (the teacher's bar), the training poses 0-44, or both.")
    p.add_argument("--garments", nargs="+", default=list(WANG_GARMENT_ORDER), help="Garments of the distribution.")
    p.add_argument("--num-envs", type=int, default=24, help="Configurations a world holds at once.")
    p.add_argument("--max-cells", type=int, default=None, help="Stop after this many configurations; for smoke tests.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default=None, help="Defaults to <work-dir>/expert_r<region>_<poses>_s<seed>.")
    p.add_argument("--work-dir", type=str, default="output/uipc_manip")
    p.add_argument("--save-observations", action="store_true",
                   help="Also store the point-cloud observation of every decision; 5,383 floats a step.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Play this SAC checkpoint deterministically instead of the expert, on the checkpoint's own "
                        "physics and observation. This is how a voided evaluation round is replayed offline.")
    return p


class EpisodeTape:
    """One slot's decisions and the record its episode ends on, in ``train_sac.evaluate``'s shape."""

    def __init__(self, metric_keys: tuple[str, ...], save_observations: bool) -> None:
        self.metric_keys = tuple(metric_keys)
        self.save_observations = bool(save_observations)
        self.reset()

    def reset(self) -> None:
        self.privileged: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.observations: list[np.ndarray] = []
        self.stages: list[str] = []
        self.ret = 0.0
        self.max_tracking = 0.0
        self.early_turn = False
        self.running_max = {k: -np.inf for k in self.metric_keys}
        self.last_seen = {k: float("nan") for k in self.metric_keys}

    def step(self, privileged, action, obs, stage: str, reward: float, info: dict) -> None:
        self.privileged.append(np.asarray(privileged, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.stages.append(str(stage))
        if self.save_observations:
            self.observations.append(np.asarray(obs, dtype=np.float32))
        self.ret += float(reward)
        self.max_tracking = max(self.max_tracking, float(info.get("tracking_error", 0.0)))
        self.early_turn |= bool(info.get("early_turn", False))
        for k in self.metric_keys:
            if k in info:
                self.running_max[k] = max(self.running_max[k], float(info[k]))
                self.last_seen[k] = float(info[k])

    def record(self, slot: int, info: dict, cell: tuple[str, int]) -> dict:
        out = {
            "slot": int(slot),
            "success": bool(info.get("success", False)),
            "distance": float(info.get("distance", np.nan)),
            "return": float(self.ret),
            "max_tracking_error": float(self.max_tracking),
            "sim_error": bool(info.get("sim_error", False)),
            "early_turn": bool(self.early_turn),
            "length": len(self.actions),
            "final_stage": self.stages[-1] if self.stages else "none",
        }
        # FMVP Appendix A.1 keeps a trajectory when it ends dressed and never cut the elbow.
        out["paper_filter"] = bool(out["success"] and not out["early_turn"])
        for k in self.metric_keys:
            out[f"final_{k}"] = float(info[k]) if k in info else float(self.last_seen[k])
            out[f"max_{k}"] = float(self.running_max[k])
        out["garment"], out["human"] = str(cell[0]), int(cell[1])
        out["cell"] = str(info.get("cell", f"{cell[0]}|human={cell[1]}"))
        return out

    def save(self, path: Path, record: dict) -> None:
        arrays = {
            "privileged": np.stack(self.privileged) if self.privileged else np.zeros((0, 0), dtype=np.float32),
            "actions": np.stack(self.actions) if self.actions else np.zeros((0, 0), dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "stages": np.asarray(self.stages),
            "record": np.asarray(json.dumps(record)),
        }
        if self.save_observations and self.observations:
            arrays["obs"] = np.stack(self.observations)
        np.savez_compressed(path, **arrays)


def run_world(env, cells, horizon: int, seed_base: int, out_dir: Path, index: int, *,
              save_observations: bool = False, policy=None) -> list[dict]:
    """Play one episode on every slot of a built world; return one record per slot.

    ``policy`` maps observations to actions; without one the environment's scripted expert plays.
    """
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)
    tapes = [EpisodeTape(env.metric_keys, save_observations) for _ in range(env.num_envs)]
    obs = env.reset([seed_base + i for i in range(env.num_envs)])
    records: list[dict] = []
    for _ in range(int(horizon)):
        privileged = env.privileged()
        stages = env.scripted_stage_names() if policy is None else ["policy"] * env.num_envs
        actions = np.asarray(env.scripted_actions() if policy is None else policy(obs), dtype=np.float32)
        next_obs, rewards, dones, infos = env.step(actions)
        for i, info in enumerate(infos):
            tapes[i].step(privileged[i], actions[i], obs[i], stages[i], float(rewards[i]), info)
        obs = next_obs
        if bool(np.any(dones)):
            break
    for i, (tape, cell) in enumerate(zip(tapes, cells, strict=True)):
        record = tape.record(i + index, infos[i], cell)
        path = out_dir / "episodes" / f"episode_{i + index:05d}.npz"
        tape.save(path, record)
        record["path"] = str(path.relative_to(out_dir))
        records.append(record)
        print(f"[expert] {record['cell']:34s} final upper {record['final_upperarm_ratio']:.3f} "
              f"(max {record['max_upperarm_ratio']:.3f}) forearm {record['final_forearm_ratio']:.3f} "
              f"stage {record['final_stage']:10s} early_turn={record['early_turn']} sim_error={record['sim_error']}", flush=True)
    return records


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, extra = build_parser().parse_known_args(argv)
    poses = {
        "heldout": list(pretrain_wang.EVAL_POSES),
        "train": list(pretrain_wang.TRAIN_POSES),
        "all": sorted({*pretrain_wang.TRAIN_POSES, *pretrain_wang.EVAL_POSES}),
    }[args.poses]
    garments = list(dict.fromkeys(str(g) for g in args.garments))
    # The teacher's own command line decides the physics, the horizon and the observation; this run
    # only replaces its policy, so the bar it measures is the bar the teacher is scored against.
    teacher_argv = ["teacher", "--region", str(args.region), "--garments", *garments,
                    "--num-envs", str(args.num_envs), "--seed", str(args.seed), "--work-dir", str(args.work_dir), *extra]
    protocol_args, targs, _ = pretrain_wang.prepare(teacher_argv)
    from . import train_sac

    payload, step = None, None
    if args.checkpoint is not None:
        from .sac import SACAgent

        payload = SACAgent.read_checkpoint(args.checkpoint)
        saved_task = payload.get("metadata", {}).get("task")
        if saved_task not in (None, "dressing"):
            raise ValueError(f"Checkpoint task {saved_task!r} is not dressing")
        # Playback inherits the checkpoint's physics and observation, as ``collect_rollouts`` does.
        targs.eval_only = True
        train_sac.restore_resume_args(targs, pretrain_wang.trainer_argv(protocol_args, extra), payload)
        step = int(payload.get("step", 0))
    base_cfg = train_sac.dressing_config(targs)
    default_name = (f"expert_r{args.region}_{args.poses}_s{args.seed}" if payload is None
                    else f"replay_r{args.region}_{args.poses}_{step:08d}")
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.work_dir) / default_name
    (out_dir / "episodes").mkdir(parents=True, exist_ok=True)

    # Genesis must come up before anything runs matrix work on the GPU: generating a body does, and so
    # does building the agent, and cuBLAS first leaves Quadrants unable to start (``dressing_env``).
    _ensure_genesis(base_cfg.logging_level)
    policy = None
    if payload is not None:
        import torch

        from .obs import ObsSpec
        from .sac import SACConfig

        torch.manual_seed(int(args.seed))
        agent = SACAgent(ObsSpec(targs.point_budget), GenesisIPCDressingEnv.action_dim,
                         SACConfig.from_dict(payload["sac_config"]), targs.device)
        agent.load(args.checkpoint, load_optimizers=False)
        agent.train(False)
        policy = lambda obs: agent.act(obs, deterministic=True)  # noqa: E731

    factory = LiveCellFactory(base_cfg.live)
    wanted = pretrain_wang.region_configs([int(args.region)], garments, poses)
    if args.max_cells is not None:
        wanted = wanted[: int(args.max_cells)]
    cells, dropped = [], []
    for garment, body in wanted:
        try:
            factory.clearance_for(garment, body)
            cells.append((str(garment), int(body)))
        except Exception as exc:  # NoClearPlacement and anything else the placement refuses
            print(f"[expert] {garment} on body {body} cannot be placed: {exc}", flush=True)
            dropped.append({"garment": str(garment), "body": int(body), "reason": str(exc)})
    if not cells:
        raise RuntimeError("No configuration of this region can be placed")

    t0, records, size = time.time(), [], max(1, int(args.num_envs))
    played = "the expert" if payload is None else f"checkpoint {Path(args.checkpoint).name} (step {step})"
    print(f"[expert] region {args.region} {args.poses} poses: {len(cells)} configurations in worlds of {size}, "
          f"horizon {targs.horizon}, no decision watchdog, playing {played}", flush=True)
    for start in range(0, len(cells), size):
        chunk = cells[start:start + size]
        # Without the watchdog a slow decision costs time only; a trip would void the whole world.
        cfg = replace(base_cfg, cells=tuple(chunk), decision_watchdog=False)
        env = GenesisIPCDressingEnv(cfg, num_envs=len(chunk), cell_factory=factory)
        try:
            records += run_world(env, chunk, targs.horizon, int(args.seed) * 1000 + start, out_dir, start,
                                 save_observations=bool(args.save_observations), policy=policy)
        finally:
            env.close()
            gc.collect()
        elapsed = time.time() - t0
        print(f"[expert] {len(records)}/{len(cells)} configurations, {elapsed:.0f}s", flush=True)

    summary = pretrain_wang.eval_summary(records, cells)
    summary["mean_length"] = float(np.mean([r["length"] for r in records]))
    manifest = {
        "policy": "expert" if payload is None else "checkpoint",
        "checkpoint": None if payload is None else str(args.checkpoint),
        "checkpoint_step": step,
        "region": int(args.region),
        "poses": args.poses,
        "garments": garments,
        "horizon": int(targs.horizon),
        "num_envs": int(args.num_envs),
        "seed": int(args.seed),
        "cells": [[g, b] for g, b in cells],
        "dropped": dropped,
        "env": base_cfg.to_dict(),
        "elapsed_s": time.time() - t0,
        "summary": summary,
    }
    (out_dir / "records.json").write_text(json.dumps(records, indent=1, default=float) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float) + "\n")
    print(f"[expert] mean final upper-arm ratio {summary['mean_final_upperarm_ratio']:.3f}, "
          f"success {summary['success_rate']:.3f} over {summary['cell_count']} configurations, "
          f"mean max upper-arm {summary['mean_max_upperarm_ratio']:.3f}, "
          f"paper filter {summary['paper_filter_rate']:.3f}, {manifest['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
