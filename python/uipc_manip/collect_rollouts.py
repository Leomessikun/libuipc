"""FMVP Stage I-B: roll out a frozen dressing policy and keep the paper-filtered episodes.

FMVP (Appendix A.1) rolls its simulation teachers out for more than 8,000
episodes and keeps a trajectory when its final upper-arm dressed ratio is at
least 0.7 and the gripper never made an early turn at the elbow; 2,514 survive
and are distilled by behaviour cloning. This collector applies the same two
filters to a SAC checkpoint, the scripted expert, or a random policy, stores
each kept episode as one compressed ``episode_*.npz`` of training-time
observations and executed actions, and records every attempt, kept or not, in
``episode_metrics.json``.

Observations are collected with the training-time camera jitter and dropout
on, and every episode draws fresh randomness, so the student sees the
distribution the teacher acted in rather than a handful of fixed evaluation
seeds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .train_sac import build_parser, make_env, resolve_defaults, restore_resume_args


class EpisodeCollector:
    """Per-slot bookkeeping of the FMVP trajectory filter; independent of the simulator."""

    def __init__(self, num_envs: int, out_dir: Path, *, min_upperarm_ratio: float, target_kept: int) -> None:
        self.out_dir = Path(out_dir)
        (self.out_dir / "episodes").mkdir(parents=True, exist_ok=True)
        self.min_upperarm_ratio = float(min_upperarm_ratio)
        self.target_kept = int(target_kept)
        self.records: list[dict] = []
        self.kept = 0
        self._obs = [[] for _ in range(num_envs)]
        self._actions = [[] for _ in range(num_envs)]
        self._rewards = [[] for _ in range(num_envs)]
        self._max_upper = np.zeros(num_envs)
        self._max_fore = np.zeros(num_envs)
        self._early_turn = np.zeros(num_envs, dtype=bool)
        self._tracking = np.zeros(num_envs)

    @property
    def attempts(self) -> int:
        return len(self.records)

    def step(self, i: int, obs: np.ndarray, action: np.ndarray, reward: float, info: dict) -> dict | None:
        """Record one transition of slot ``i``; return the episode record when it ends."""
        self._obs[i].append(np.asarray(obs, dtype=np.float32))
        self._actions[i].append(np.asarray(action, dtype=np.float32))
        self._rewards[i].append(float(reward))
        self._max_upper[i] = max(self._max_upper[i], float(info.get("upperarm_ratio", 0.0)))
        self._max_fore[i] = max(self._max_fore[i], float(info.get("forearm_ratio", 0.0)))
        self._early_turn[i] |= bool(info.get("early_turn", False))
        self._tracking[i] = max(self._tracking[i], float(info.get("tracking_error", 0.0)))
        if not (info.get("time_limit", False) or info.get("sim_error", False)):
            return None
        return self._finish(i, info)

    def _finish(self, i: int, info: dict) -> dict:
        sim_error = bool(info.get("sim_error", False))
        final_upper = float(info.get("upperarm_ratio", 0.0))
        paper_success = (not sim_error) and final_upper >= self.min_upperarm_ratio and not self._early_turn[i]
        keep = paper_success and self.kept < self.target_kept
        record = {
            "index": len(self.records),
            "garment": str(info.get("garment", "")),
            "length": len(self._actions[i]),
            "return": float(sum(self._rewards[i])),
            "final_upperarm_ratio": final_upper,
            "max_upperarm_ratio": float(self._max_upper[i]),
            "final_forearm_ratio": float(info.get("forearm_ratio", 0.0)),
            "max_forearm_ratio": float(self._max_fore[i]),
            "early_turn": bool(self._early_turn[i]),
            "max_tracking_error": float(self._tracking[i]),
            "sim_error": sim_error,
            "paper_filter_success": bool(paper_success),
            "kept": bool(keep),
            "path": None,
        }
        if keep:
            path = self.out_dir / "episodes" / f"episode_{self.kept:05d}.npz"
            np.savez_compressed(
                path,
                obs=np.stack(self._obs[i]),
                actions=np.stack(self._actions[i]),
                rewards=np.asarray(self._rewards[i], dtype=np.float32),
            )
            record["path"] = str(path.relative_to(self.out_dir))
            self.kept += 1
        self.records.append(record)
        self._obs[i], self._actions[i], self._rewards[i] = [], [], []
        self._max_upper[i] = self._max_fore[i] = self._tracking[i] = 0.0
        self._early_turn[i] = False
        return record

    def summary(self) -> dict:
        rows = self.records
        n = max(1, len(rows))
        return {
            "attempted_episodes": len(rows),
            "kept_episodes": self.kept,
            "paper_filter_rate": float(sum(r["paper_filter_success"] for r in rows) / n),
            "early_turn_rate": float(sum(r["early_turn"] for r in rows) / n),
            "sim_errors": int(sum(r["sim_error"] for r in rows)),
            "mean_final_upperarm_ratio": float(np.mean([r["final_upperarm_ratio"] for r in rows])) if rows else 0.0,
            "mean_max_upperarm_ratio": float(np.mean([r["max_upperarm_ratio"] for r in rows])) if rows else 0.0,
            "kept_per_garment": {
                g: int(sum(1 for r in rows if r["kept"] and r["garment"] == g)) for g in sorted({r["garment"] for r in rows})
            },
        }


def add_collection_args(parser):
    parser.add_argument("--checkpoint", type=str, default=None, help="SAC checkpoint to roll out; omit to use --policy heuristic or random.")
    parser.add_argument("--out-dir", type=str, default=None, help="Rollout directory; defaults to <work-dir>/<run-name>/rollouts.")
    parser.add_argument("--target-kept-episodes", type=int, default=2514, help="Stop once this many episodes pass the paper filter (FMVP kept 2,514).")
    parser.add_argument("--max-episodes", type=int, default=8000, help="Stop after this many attempts (FMVP rolled out more than 8,000).")
    parser.add_argument("--min-upperarm-ratio", type=float, default=0.7, help="Final upper-arm ratio the paper filter requires; lower it only for labelled smoke tests.")
    parser.add_argument("--stochastic", action="store_true", help="Sample the SAC actor instead of playing its mean.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = add_collection_args(build_parser())
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    payload = None
    if args.checkpoint is not None:
        from .sac import SACAgent

        payload = SACAgent.read_checkpoint(args.checkpoint)
        # Collection is policy playback: inherit the teacher's environment by
        # default while allowing explicitly requested evaluation variants.
        args.eval_only = True
        restore_resume_args(args, argv, payload)
        saved_task = payload.get("metadata", {}).get("task")
        if saved_task is not None and saved_task != args.task:
            raise ValueError(f"Checkpoint task {saved_task!r} does not match collection task {args.task!r}")
    resolve_defaults(args)
    np.random.seed(args.seed)
    run_name = args.run_name or f"rollouts_{args.task}_{args.policy if args.checkpoint is None else 'sac'}_seed{args.seed}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.work_dir) / run_name / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    description = env.descriptions[0]
    manifest = {"args": vars(args), "env": description["config"], "obs_dim": int(env.obs_dim), "action_dim": int(env.action_dim), "point_budget": int(args.point_budget)}

    if args.checkpoint is not None:
        import torch

        from .obs import ObsSpec
        from .sac import SACAgent, SACConfig

        torch.manual_seed(args.seed)
        cfg = SACConfig.from_dict(payload["sac_config"])
        agent = SACAgent(ObsSpec(args.point_budget), env.action_dim, cfg, args.device)
        agent.load(args.checkpoint, load_optimizers=False)
        agent.train(False)
        policy = lambda obs: agent.act(obs, deterministic=not args.stochastic)  # noqa: E731
        manifest["checkpoint"] = str(args.checkpoint)
        manifest["sac_config"] = cfg.to_dict()
        manifest["protocol"] = agent.protocol()
        manifest["teacher_step"] = int(payload.get("step", 0))
    elif args.policy == "heuristic":
        policy = lambda obs: env.scripted_actions()  # noqa: E731
    else:
        policy = lambda obs: np.random.uniform(-1.0, 1.0, size=(obs.shape[0], env.action_dim))  # noqa: E731
    manifest["policy"] = "sac" if args.checkpoint is not None else str(args.policy)
    if args.checkpoint is None and args.policy == "sac":
        raise SystemExit("--policy sac needs --checkpoint; pass --policy heuristic or random to collect without one")

    collector = EpisodeCollector(env.num_envs, out_dir, min_upperarm_ratio=args.min_upperarm_ratio, target_kept=args.target_kept_episodes)
    obs = env.reset([args.seed * 100 + i for i in range(env.num_envs)])
    t0 = time.time()
    print(f"[collect] policy={manifest['policy']} envs={env.num_envs} horizon={args.horizon} target_kept={args.target_kept_episodes} max_attempts={args.max_episodes} min_ratio={args.min_upperarm_ratio}", flush=True)
    while collector.kept < args.target_kept_episodes and collector.attempts < args.max_episodes:
        actions = np.asarray(policy(obs), dtype=np.float32)
        next_obs, rewards, dones, infos = env.step(actions)
        for i, info in enumerate(infos):
            record = collector.step(i, obs[i], actions[i], float(rewards[i]), info)
            if record is not None:
                print(
                    f"[collect] episode={record['index']} garment={record['garment']} final_upper={record['final_upperarm_ratio']:.3f} "
                    f"max_upper={record['max_upperarm_ratio']:.3f} early_turn={record['early_turn']} kept={record['kept']} "
                    f"({collector.kept}/{args.target_kept_episodes}) elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
        obs = next_obs
    manifest["summary"] = collector.summary()
    manifest["elapsed_s"] = time.time() - t0
    (out_dir / "episode_metrics.json").write_text(json.dumps(collector.records, indent=1) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[collect] " + json.dumps(manifest["summary"]), flush=True)
    env.close()


if __name__ == "__main__":
    main()
