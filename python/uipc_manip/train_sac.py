"""Point-cloud SAC pretraining on the Genesis + libuipc manipulation tasks.

Usage (from the repository root, inside the Genesis environment)::

    PYTHONPATH=python python -m uipc_manip.train_sac --task cloth_drag --num-envs 4 \
        --total-transitions 20000 --work-dir output/uipc_manip --run-name cloth_drag_seed1

``--policy heuristic --eval-only`` runs the scripted reachability check that
every task must pass before SAC is worth running. ``--resume`` continues from a
checkpoint directory, and ``--eval-only`` with ``--resume`` plays a checkpoint
without modifying it.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .curriculum import WANG_GARMENT_ORDER, curriculum_order, garment_curriculum_stage
from .dressing_env import DEFAULT_GARMENTS, DressingConfig, GenesisIPCDressingEnv
from .genesis_env import EnvConfig, GenesisIPCManipEnv, ViewerClosed
from .obs import ObsSpec, goal_rel, marker_centroid_rel
from .sac import (
    SACConfig,
    gradient_update_budget,
    wang_equivalent_alpha_lr,
    wang_equivalent_discount,
    wang_equivalent_reward_scale,
)
from .tasks import TASKS, heuristic_action


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=[*sorted(TASKS), "dressing"], default="dressing")
    p.add_argument("--human", type=int, default=0, help="dressing: cached human whose cells fill every slot (a regional teacher).")
    p.add_argument("--garments", type=str, nargs="+", default=list(DEFAULT_GARMENTS), help="dressing: garments cycled across slots.")
    p.add_argument("--anchor-count", type=int, default=12, help="dressing: cuff vertices held by the picker.")
    p.add_argument("--cuff-strength", type=float, default=1.0e4, help="dressing: soft position constraint strength_rate of the held cuff; 100 lets the garment detach from the tool.")
    p.add_argument("--no-obs-augment", action="store_true", help="dressing: disable camera jitter and dropout.")
    p.add_argument("--garment-curriculum-interval", type=int, default=0, help="dressing: vector steps between admitting one more garment's slots to replay, easiest first (Wang's curriculum_update_freq); 0 trains on every garment from the start.")
    p.add_argument("--garment-curriculum-order", type=str, default=",".join(WANG_GARMENT_ORDER), help="dressing: comma-separated garment names, easiest first; garments not named are appended.")
    p.add_argument("--num-envs", type=int, default=32, help="Deformable and robot copies solved together in one IPC world.")
    p.add_argument("--horizon", type=int, default=None, help="Decisions per episode; 900 for dressing, 150 otherwise.")
    p.add_argument("--action-repeat", type=int, default=None, help="Simulation steps per decision; 1 for dressing (the reference decides at 60 Hz), 5 otherwise. The tool speed cap is held over the whole decision.")
    p.add_argument("--max-translation", type=float, default=0.006)
    p.add_argument("--point-budget", type=int, default=None, help="Points per observation; 768 for dressing, 256 otherwise.")
    p.add_argument("--friction", type=float, default=0.6)
    p.add_argument("--settle-steps", type=int, default=40)
    p.add_argument("--vis", action="store_true", help="Open the Genesis viewer (forces a single environment).")
    p.add_argument("--policy", choices=("sac", "heuristic", "random"), default="sac")
    p.add_argument("--total-transitions", type=int, default=20_000)
    p.add_argument("--init-steps", type=int, default=0, help="Vector steps of uniform random actions before the policy acts.")
    p.add_argument("--updates-per-step", type=int, default=0, help="Gradient updates per vector step; 0 = one per collected transition.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--discount", type=float, default=None, help="Override the horizon-equivalent Wang discount.")
    p.add_argument("--alpha-lr", type=float, default=None)
    p.add_argument("--actor-lr", type=float, default=1.0e-4)
    p.add_argument("--critic-lr", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--actor", choices=("wang-flow", "flat"), default="wang-flow", help="wang-flow is the reference tool-point actor.")
    p.add_argument("--algo", choices=("sac", "flashsac"), default="sac", help="Scalar reference critic or bounded categorical critic.")
    p.add_argument("--encoder", choices=("pointnet2", "transformer"), default="pointnet2")
    p.add_argument("--num-bins", type=int, default=51, help="flashsac: value atoms per critic head.")
    p.add_argument("--min-v", type=float, default=-50.0, help="flashsac: lowest value atom.")
    p.add_argument("--max-v", type=float, default=50.0, help="flashsac: highest value atom.")
    p.add_argument("--sa-neighbors", type=int, nargs="+", default=[8, 16], help="Ball-query neighbours per set-abstraction level.")
    p.add_argument("--point-jitter", type=float, default=0.0, help="Per-point jitter [m] applied to replay samples.")
    p.add_argument("--grad-clip-max-norm", type=float, default=0.0)
    p.add_argument("--min-alpha", type=float, default=0.0)
    p.add_argument("--init-temperature", type=float, default=0.1, help="Initial SAC temperature; the reference uses 0.1 at a 150-step horizon.")
    p.add_argument("--eval-freq", type=int, default=500, help="Vector steps between evaluations (0 disables).")
    p.add_argument("--num-eval-episodes", type=int, default=4)
    p.add_argument("--checkpoint-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--work-dir", type=str, default="output/uipc_manip", help="Runs land here; the repository ignores output/.")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint file to load (weights, temperature, optimizers).")
    p.add_argument("--resume-replay", type=str, default=None, help="Replay snapshot directory saved with the checkpoint.")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--save-trajectories", action="store_true", help="Store evaluation trajectories as .npz files.")
    return p


def resolve_defaults(args) -> None:
    """Fill the task-dependent launcher defaults in place."""
    if args.horizon is None:
        args.horizon = 900 if args.task == "dressing" else 150
    if args.action_repeat is None:
        args.action_repeat = 1 if args.task == "dressing" else 5
    if args.point_budget is None:
        args.point_budget = 768 if args.task == "dressing" else 256


def build_sac_config(args) -> SACConfig:
    """The reference SAC settings with the horizon-equivalent discount and temperature learning rate."""
    discount = wang_equivalent_discount(args.horizon) if args.discount is None else float(args.discount)
    alpha_lr = wang_equivalent_alpha_lr(args.horizon) if args.alpha_lr is None else float(args.alpha_lr)
    sac_cfg = SACConfig(
        discount=discount,
        alpha_lr=alpha_lr,
        init_temperature=args.init_temperature,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        grad_clip_max_norm=args.grad_clip_max_norm,
        min_alpha=args.min_alpha,
        point_jitter_scale=args.point_jitter,
        actor_type=args.actor,
        algo=args.algo,
        num_bins=args.num_bins,
        min_v=args.min_v,
        max_v=args.max_v,
    )
    neighbors = [int(n) for n in args.sa_neighbors]
    if len(neighbors) == 1:
        neighbors = neighbors * 2
    sac_cfg.encoder = replace(sac_cfg.encoder, kind=args.encoder, sa_neighbors=neighbors)
    return sac_cfg


def make_env(args):
    if args.task == "dressing":
        cfg = DressingConfig(
            human=args.human,
            garments=tuple(args.garments),
            horizon=args.horizon,
            action_repeat=args.action_repeat,
            point_budget=args.point_budget,
            anchor_count=args.anchor_count,
            constraint_strength=args.cuff_strength,
            seed=args.seed,
            augment_obs=not args.no_obs_augment,
            show_viewer=bool(args.vis),
        )
        return GenesisIPCDressingEnv(cfg, num_envs=args.num_envs)
    return GenesisIPCManipEnv(env_config(args), num_envs=args.num_envs)


def env_config(args) -> EnvConfig:
    return EnvConfig(
        task=args.task,
        horizon=args.horizon,
        action_repeat=args.action_repeat,
        max_translation=args.max_translation,
        point_budget=args.point_budget,
        seed=args.seed,
        friction=args.friction,
        settle_steps=args.settle_steps,
        show_viewer=bool(args.vis),
    )


def heuristic_actions(obs: np.ndarray, spec: ObsSpec, max_translation: float) -> np.ndarray:
    return np.stack([heuristic_action(marker_centroid_rel(o, spec), goal_rel(o, spec), max_translation) for o in obs])


def evaluate(env, policy, spec: ObsSpec, args, episodes: int, trajectory_dir: Path | None = None) -> dict:
    """Play deterministic episodes and report success and distance statistics."""
    obs = env.reset([args.seed * 1000 + i for i in range(env.num_envs)])
    finished: list[dict] = []
    returns = np.zeros(env.num_envs)
    max_tracking = np.zeros(env.num_envs)
    early_turn_seen = np.zeros(env.num_envs, dtype=bool)
    metric_keys = tuple(getattr(env, "metric_keys", ()))
    running_max = {k: np.full(env.num_envs, -np.inf) for k in metric_keys}
    trajectories = [[] for _ in range(env.num_envs)] if trajectory_dir is not None else None
    episode_index = 0
    while len(finished) < episodes:
        actions = policy(obs, True)
        if trajectories is not None:
            for i, state in enumerate(env.states()):
                trajectories[i].append(state)
        obs, rewards, dones, infos = env.step(actions)
        returns += rewards
        for i, info in enumerate(infos):
            max_tracking[i] = max(max_tracking[i], float(info.get("tracking_error", 0.0)))
            early_turn_seen[i] |= bool(info.get("early_turn", False))
            for k in metric_keys:
                if k in info:
                    running_max[k][i] = max(running_max[k][i], float(info[k]))
            if dones[i]:
                record = {
                    "success": bool(info.get("success", False)),
                    "distance": float(info.get("distance", np.nan)),
                    "return": float(returns[i]),
                    "max_tracking_error": float(max_tracking[i]),
                    "sim_error": bool(info.get("sim_error", False)),
                    "early_turn": bool(early_turn_seen[i]),
                }
                # FMVP Appendix A.1 keeps a trajectory when it ends dressed and never cut the elbow.
                record["paper_filter"] = bool(record["success"] and not record["early_turn"])
                for k in metric_keys:
                    record[f"final_{k}"] = float(info.get(k, np.nan))
                    record[f"max_{k}"] = float(running_max[k][i])
                    running_max[k][i] = -np.inf
                if "garment" in info:
                    record["garment"] = str(info["garment"])
                if len(finished) < episodes:
                    finished.append(record)
                    if trajectories is not None:
                        static = None
                        if hasattr(env, "arm_vertices"):
                            static = {"vertices": env.arm_vertices, "faces": env.arm_faces}
                        _save_trajectory(trajectory_dir, episode_index, trajectories[i], env.descriptions[i], static)
                        episode_index += 1
                returns[i] = 0.0
                max_tracking[i] = 0.0
                early_turn_seen[i] = False
                if trajectories is not None:
                    trajectories[i] = []
    distances = np.array([r["distance"] for r in finished])
    summary = {
        "episodes": len(finished),
        "success_rate": float(np.mean([r["success"] for r in finished])),
        "mean_final_distance": float(np.nanmean(distances)),
        "mean_return": float(np.mean([r["return"] for r in finished])),
        "max_tracking_error": float(max(r["max_tracking_error"] for r in finished)),
        "sim_errors": int(sum(r["sim_error"] for r in finished)),
        "early_turn_rate": float(np.mean([r["early_turn"] for r in finished])),
        "paper_filter_rate": float(np.mean([r["paper_filter"] for r in finished])),
    }
    for k in metric_keys:
        summary[f"mean_final_{k}"] = float(np.nanmean([r[f"final_{k}"] for r in finished]))
        summary[f"mean_max_{k}"] = float(np.nanmean([r[f"max_{k}"] for r in finished]))
    garments = sorted({r["garment"] for r in finished if "garment" in r})
    for g in garments:
        rows = [r for r in finished if r.get("garment") == g]
        summary[f"success_rate_{g}"] = float(np.mean([r["success"] for r in rows]))
    summary["records"] = finished
    return summary


def _save_trajectory(directory: Path, index: int, states: list[dict], description: dict, static: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    static = static or {}
    np.savez_compressed(
        directory / f"episode_{index:03d}.npz",
        **{f"static_{k}": np.asarray(v) for k, v in static.items()},
        positions=np.stack([s["positions"] for s in states]),
        tcp=np.stack([s["tcp"] for s in states]),
        goal=np.stack([s["goal"] for s in states]),
        marker_centroid=np.stack([s["marker_centroid"] for s in states]),
        qpos=np.stack([s["qpos"] for s in states]),
        faces=np.asarray(description["faces"], dtype=np.int32).reshape(-1, 3),
        edges=np.asarray(description["edges"], dtype=np.int32).reshape(-1, 2),
        radius=float(description["radius"]),
        deformable=str(description["deformable"]),
        task=str(description["task"]),
    )


def _set_viewer_caption(env, text: str) -> None:
    try:
        env.scene.viewer._pyrender_viewer.set_caption(text)
    except Exception:  # viewer backends without a caption API
        pass


def _hold_viewer(env, text: str) -> None:
    """Keep the Genesis viewer open after evaluation so the scene can be inspected."""
    _set_viewer_caption(env, text)
    viewer = getattr(env.scene, "viewer", None)
    if viewer is None:
        return
    while viewer.is_alive():
        viewer.update(force=True)
        time.sleep(0.03)


def checkpoint_score(metrics: dict) -> tuple:
    """Lexicographic selection key: FMVP's final-ratio success first, then how far the arm was dressed."""
    return (
        metrics["success_rate"],
        -metrics["mean_final_distance"],
        metrics.get("mean_max_upperarm_ratio", 0.0),
        metrics["mean_return"],
    )


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fields: list[str] | None = None

    def log(self, row: dict) -> None:
        if self._fields is None:
            self._fields = list(row)
            with self.path.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self._fields).writeheader()
        with self.path.open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=self._fields, extrasaction="ignore").writerow(
                {k: row.get(k, "") for k in self._fields}
            )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.vis:
        args.num_envs = 1
    np.random.seed(args.seed)
    run_name = args.run_name or f"{args.task}_{args.policy}_seed{args.seed}"
    run_dir = Path(args.work_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed * 100 + i for i in range(args.num_envs)]
    resolve_defaults(args)
    env = make_env(args)
    spec = ObsSpec(args.point_budget)
    description = env.descriptions[0]
    print(
        f"[uipc-manip] task={args.task} envs={env.num_envs} obs_dim={env.obs_dim} build={description['build_seconds']:.1f}s "
        f"settle_displacement={description['settle_displacement_m']:.4f} m",
        flush=True,
    )
    (run_dir / "env.json").write_text(json.dumps(description, indent=2) + "\n")

    if args.policy == "heuristic":
        if args.task == "dressing":
            policy = lambda obs, deterministic: env.scripted_actions()  # noqa: E731
        else:
            policy = lambda obs, deterministic: heuristic_actions(obs, spec, args.max_translation)  # noqa: E731
        agent = None
    elif args.policy == "random":
        policy = lambda obs, deterministic: np.random.uniform(-1, 1, size=(obs.shape[0], env.action_dim))  # noqa: E731
        agent = None
    else:
        import torch

        from .sac import SACAgent

        torch.manual_seed(args.seed)
        sac_cfg = build_sac_config(args)
        agent = SACAgent(spec, env.action_dim, sac_cfg, args.device)
        if args.resume:
            payload = agent.load(args.resume, load_optimizers=not args.eval_only)
            saved_task = payload["metadata"].get("task")
            if saved_task is not None and saved_task != args.task:
                raise ValueError(f"Checkpoint was trained on task {saved_task!r}, not {args.task!r}")
            print(f"[uipc-manip] resumed {args.resume} at step {payload['step']}", flush=True)
        policy = lambda obs, deterministic: agent.act(obs, deterministic)  # noqa: E731

    trajectory_dir = run_dir / "trajectories" if args.save_trajectories else None
    if args.eval_only or agent is None:
        if args.vis:
            _set_viewer_caption(env, f"uipc_manip {args.task} | {args.policy} policy | blue: deformable, green: goal, red: marker centroid")
        try:
            metrics = evaluate(env, policy, spec, args, args.num_eval_episodes, trajectory_dir)
        except ViewerClosed:
            print("[uipc-manip] viewer closed; exiting", flush=True)
            return
        (run_dir / "eval.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "records"}, indent=2), flush=True)
        if args.vis:
            _hold_viewer(env, f"uipc_manip {args.task} | done: success {metrics['success_rate']:.0%} | drag to orbit, close window to exit")
        env.close()
        return

    from .replay import FlatReplayBuffer

    replay = FlatReplayBuffer(env.obs_dim, env.action_dim, args.replay_capacity, args.batch_size, args.device)
    reward_scale = wang_equivalent_reward_scale(agent.cfg.discount)
    start_step = 0
    if args.resume:
        start_step = int(agent.read_checkpoint(args.resume)["step"])
        if args.resume_replay:
            replay.load(args.resume_replay)
    metadata = {
        "task": args.task,
        "env": description["config"],
        "sac_config": agent.cfg.to_dict(),
        "reward_scale": reward_scale,
        "seed": args.seed,
        "num_envs": env.num_envs,
    }
    (run_dir / "config.json").write_text(json.dumps({"args": vars(args), **metadata}, indent=2) + "\n")
    logger = CsvLogger(run_dir / "train_log.csv")
    eval_logger = CsvLogger(run_dir / "eval_log.csv")
    total_vector_steps = int(np.ceil(args.total_transitions / env.num_envs))
    print(
        f"[uipc-manip] discount={agent.cfg.discount:.6f} alpha_lr={agent.cfg.alpha_lr:.2e} reward_scale={reward_scale:.3f} "
        f"vector_steps={total_vector_steps}",
        flush=True,
    )

    # Wang's garment curriculum: slots of garments not yet admitted keep stepping but do not feed replay.
    slot_garments = [str(d.get("garment", "")) for d in env.descriptions]
    order = curriculum_order(args.garment_curriculum_order.split(","), [g for g in slot_garments if g]) if any(slot_garments) else []
    slot_rank = np.array([order.index(g) if g else 0 for g in slot_garments], dtype=np.int64)
    curriculum_interval = int(args.garment_curriculum_interval) if order else 0
    active_garments = -1
    if curriculum_interval > 0:
        print(f"[uipc-manip] garment curriculum every {curriculum_interval} vector steps, order={order}", flush=True)

    obs = env.reset(seeds)
    episode_return = np.zeros(env.num_envs)
    updates_started = False
    best_score = None
    recent_returns: list[float] = []
    recent_success: list[float] = []
    t_start = time.time()
    stats: dict = {}
    for vector_step in range(start_step + 1, start_step + total_vector_steps + 1):
        if vector_step <= args.init_steps:
            actions = np.random.uniform(-1.0, 1.0, size=(env.num_envs, env.action_dim)).astype(np.float32)
        else:
            actions = agent.act(obs, deterministic=False).astype(np.float32)
        next_obs, rewards, dones, infos = env.step(actions)
        stage = garment_curriculum_stage(vector_step, interval=curriculum_interval, garment_count=max(1, len(order)))
        if stage != active_garments:
            active_garments = stage
            if curriculum_interval > 0:
                print(f"[uipc-manip] curriculum step={vector_step} active={stage}/{len(order)} garments={order[:stage]}", flush=True)
        added = 0
        for i, info in enumerate(infos):
            if info.get("sim_error"):
                episode_return[i] = 0.0
                continue
            if slot_rank[i] >= stage:
                continue
            # Time limits are not terminal states: bootstrap from the true final observation.
            terminal_obs = info.get("terminal_obs", None)
            replay.add(obs[i], actions[i], float(rewards[i]) * reward_scale, next_obs[i] if terminal_obs is None else terminal_obs, False)
            added += 1
            episode_return[i] += float(rewards[i])
            if dones[i]:
                recent_returns.append(float(episode_return[i]))
                recent_success.append(float(info.get("success", False)))
                episode_return[i] = 0.0
        obs = next_obs
        budget, updates_started = gradient_update_budget(
            transitions_added=added,
            replay_size=replay.size,
            batch_size=args.batch_size,
            updates_started=updates_started,
            updates_per_step=args.updates_per_step,
        )
        if vector_step > args.init_steps:
            for _ in range(budget):
                stats = agent.update(replay)
        if vector_step % args.log_interval == 0:
            elapsed = time.time() - t_start
            row = {
                "step": vector_step,
                "transitions": replay.total_added,
                "updates": agent.updates,
                "elapsed_s": round(elapsed, 1),
                "episode_return": float(np.mean(recent_returns[-20:])) if recent_returns else float("nan"),
                "episode_success": float(np.mean(recent_success[-20:])) if recent_success else float("nan"),
                "active_garments": int(active_garments),
                **{k: round(v, 5) for k, v in stats.items()},
            }
            logger.log(row)
            print("[uipc-manip] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        do_eval = args.eval_freq > 0 and vector_step % args.eval_freq == 0
        do_ckpt = args.checkpoint_interval > 0 and vector_step % args.checkpoint_interval == 0
        if do_eval or do_ckpt or vector_step == start_step + total_vector_steps:
            ckpt_dir = run_dir / "checkpoints"
            path = agent.save(ckpt_dir / f"checkpoint_{vector_step:07d}.pt", vector_step, metadata)
            replay.save(ckpt_dir / f"replay_{vector_step:07d}", metadata={"step": vector_step})
            if do_eval or vector_step == start_step + total_vector_steps:
                agent.train(False)
                metrics = evaluate(env, policy, spec, args, args.num_eval_episodes, trajectory_dir)
                agent.train(True)
                summary = {k: v for k, v in metrics.items() if k != "records"}
                eval_logger.log({"step": vector_step, **summary})
                print(f"[uipc-manip] eval step={vector_step} " + json.dumps(summary), flush=True)
                score = checkpoint_score(metrics)
                if best_score is None or score > best_score:
                    best_score = score
                    agent.save(ckpt_dir / "best.pt", vector_step, {**metadata, "eval": summary})
                obs = env.reset(seeds)
                episode_return[:] = 0.0
            print(f"[uipc-manip] saved {path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
