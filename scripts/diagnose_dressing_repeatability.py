"""Separate repeated observations/inference, fixed-action dynamics and policy feedback.

This bounded diagnostic never trains a policy or changes solver tolerances.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from uipc_manip import physics_gradient_probe as probe
from uipc_manip import train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.dressing_live import LiveCellFactory
from uipc_manip.physics_gradient_actor import load_agent
from uipc_manip.sac import SACAgent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=1097)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if min(args.step, args.horizon) < 1 or args.repeats < 2:
        p.error("Positive step/horizon and at least two repeats required")
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    g, b = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((g, int(b)),),
                  decision_watchdog=False, workspace=str(args.out / "assets"))
    if args.step + args.horizon >= cfg.horizon or cfg.augment_obs:
        raise ValueError("Use unaugmented observations and a window before auto-reset")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=1, cell_factory=LiveCellFactory(cfg.live))
    result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config=cfg.to_dict(), transitions=0, runs=[])
    positions, observations = [], []

    def save():
        result["seconds"] = time.perf_counter() - start
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        if agent.cfg.history_length != 1:
            raise ValueError("This audit currently supports single-frame policies")
        obs = env.reset([args.seed])
        for _ in range(args.step):
            obs, _, done, infos = env.step(agent.act(obs, deterministic=True))
            result["transitions"] += 1
            if done[0] or infos[0].get("sim_error"):
                raise RuntimeError(f"Approach terminated: {infos[0]}")
        snap = probe.take_snapshot(env, "feedback", args.step)
        result["initial"] = snap["measure"]
        cached = obs.copy()
        actions = np.stack([agent.act(cached, deterministic=True)[0] for _ in range(32)])
        result["cached_observation_action_max_spread"] = float(np.ptp(actions, axis=0).max())
        repeated_obs = []
        for _ in range(args.repeats):
            error = probe.restore(env, snap)
            if error > 1e-5:
                raise RuntimeError(f"Restore error {error} exceeds 10 micrometres")
            repeated_obs.append(env.observation()[0])
        result["restored_observation_max_spread"] = float(np.ptp(np.stack(repeated_obs), axis=0).max())
        # Only the first reference run supplies the fixed action sequence.
        fixed = None
        for rep in range(args.repeats):
            order = ("closed_loop", "fixed_actions") if rep % 2 == 0 else ("fixed_actions", "closed_loop")
            for mode in order:
                before = time.perf_counter()
                error = probe.restore(env, snap)
                obs = cached.copy()
                rows, xs, os, commands = [], [], [], []
                for t in range(args.horizon):
                    a = fixed[t].copy() if mode == "fixed_actions" else agent.act(obs, deterministic=True)[0]
                    commands.append(a.copy())
                    obs, _, done, infos = env.step(a[None])
                    result["transitions"] += 1
                    if done[0] or infos[0].get("sim_error"):
                        raise RuntimeError(f"Continuation terminated: {infos[0]}")
                    info = infos[0]
                    rows.append({k: info[k] for k in ("upperarm_ratio", "forearm_ratio", "grasp_valid",
                                 "tracking_error", "collision_rejected_substeps", "tether_rejected_substeps")})
                    xs.append(env.positions()[0].copy())
                    os.append(obs[0].copy())
                commands = np.stack(commands)
                if fixed is None:
                    fixed = commands.copy()
                if mode == "fixed_actions":
                    assert np.array_equal(commands, fixed)
                result["runs"].append(dict(mode=mode, repeat=rep, restore_error_m=error, trace=rows,
                                            actions=commands.tolist(), seconds=time.perf_counter() - before))
                positions.append(np.stack(xs))
                observations.append(np.stack(os))
                save()
                print(json.dumps(dict(mode=mode, repeat=rep, final_coverage=rows[-1]["upperarm_ratio"])), flush=True)
        x, o = np.stack(positions), np.stack(observations)
        a = np.asarray([r["actions"] for r in result["runs"]])
        np.savez_compressed(args.out / "traces.npz", positions=x, observations=o, actions=a,
                            snapshot_positions=snap["positions"][0], snapshot_observation=cached[0])
        result["spread"] = {}
        for mode in ("fixed_actions", "closed_loop"):
            indices = [i for i, r in enumerate(result["runs"]) if r["mode"] == mode]
            coverage = np.asarray([[s["upperarm_ratio"] for s in result["runs"][i]["trace"]] for i in indices])
            result["spread"][mode] = dict(position_max_m_by_step=np.ptp(x[indices], axis=0).max(axis=(1, 2)).tolist(),
                                           action_max_by_step=np.ptp(a[indices], axis=0).max(axis=1).tolist(),
                                           observation_max_by_step=np.ptp(o[indices], axis=0).max(axis=1).tolist(),
                                           coverage_by_step=np.ptp(coverage, axis=0).tolist(), final_coverages=coverage[:, -1].tolist())
        result["completed"] = True
        save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
