"""Predictability horizon of garment-arm contact: divergence of continuations from one restored state.

The scripted dressing expert drives one slot to each requested decision; the world is
snapshotted there and continued several times with (a) exactly the expert's recorded
commands and (b) the same commands with a small perturbation of the first command only.
Group (a) measures the growth of numerical non-reproducibility, group (b) the growth of a
controlled perturbation. Positions, opening centres and progress are recorded per decision
so that a finite-time growth rate can be fitted afterwards. No policy is trained and no
solver setting is changed.
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
from uipc_manip.sac import SACAgent


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Supplies the environment configuration only")
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--snapshot-steps", default="40,90,140")
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--perturbations", default="0,1e-4,1e-3,1e-2", help="Normalized action units on the first command")
    p.add_argument("--seed", type=int, default=1097)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    eps_list = parse_floats(args.perturbations)
    if args.horizon < 2 or args.repeats < 2 or not steps or min(steps) < 1:
        p.error("Need horizon >= 2, repeats >= 2 and positive snapshot steps")
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    g, b = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((g, int(b)),), decision_watchdog=False,
                  contact_force_readout=False, workspace=str(args.out / "assets"))
    if steps[-1] + args.horizon >= cfg.horizon or cfg.augment_obs:
        raise ValueError("Snapshots plus horizon must end before auto-reset; observations must be unaugmented")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=1, cell_factory=LiveCellFactory(cfg.live))
    result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config=cfg.to_dict(), decision_seconds=float(cfg.dt) * int(cfg.action_repeat),
                  max_translation_m=float(cfg.max_translation), max_rotation_rad=float(cfg.max_rotation),
                  transitions=0, snapshots=[], completed=False)
    traces = {}
    rng = np.random.default_rng(args.seed)

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def measure(info):
        priv = env.privileged()[0]
        return dict(upperarm_ratio=float(info["upperarm_ratio"]), forearm_ratio=float(info["forearm_ratio"]),
                    tracking_error=float(info["tracking_error"]), grasp_valid=bool(info["grasp_valid"]),
                    opening_center=[float(x) for x in priv[13:16]], tool=[float(x) for x in priv[3:6]])

    def run(commands, first_delta=None):
        """Replay ``commands`` from the restored state; returns rows, positions and executed commands."""
        rows, xs, executed = [], [], []
        for t, a in enumerate(commands):
            a = a.copy()
            if t == 0 and first_delta is not None:
                a = np.clip(a + first_delta, -1.0, 1.0)
            executed.append(a.copy())
            _, _, done, infos = env.step(a[None])
            result["transitions"] += 1
            if done[0] or infos[0].get("sim_error"):
                raise RuntimeError(f"Continuation terminated: {infos[0]}")
            rows.append(measure(infos[0]))
            xs.append(env.positions()[0].astype(np.float32))
        return rows, np.stack(xs), np.stack(executed)

    try:
        env.reset([args.seed])
        step = 0
        for target in steps:
            while step < target:
                _, _, done, infos = env.step(env.scripted_actions())
                result["transitions"] += 1
                step += 1
                if done[0] or infos[0].get("sim_error"):
                    raise RuntimeError(f"Approach terminated at {step}: {infos[0]}")
            snap = probe.take_snapshot(env, f"step{target}", target)
            entry = dict(step=target, stage=env.scripted_stage_names()[0], initial=measure(infos[0]),
                         restore_errors=[], runs=[])
            # The expert's own continuation supplies the command sequence to replay.
            reference_rows, reference_x, reference_a = [], [], []
            for _ in range(args.horizon):
                a = env.scripted_actions()[0].copy()
                _, _, done, infos = env.step(a[None])
                result["transitions"] += 1
                if done[0] or infos[0].get("sim_error"):
                    raise RuntimeError(f"Expert continuation terminated: {infos[0]}")
                reference_rows.append(measure(infos[0]))
                reference_x.append(env.positions()[0].astype(np.float32))
                reference_a.append(a)
            fixed = np.stack(reference_a)
            entry["runs"].append(dict(group="expert_closed_loop", eps=0.0, repeat=0, trace=reference_rows))
            traces[f"s{target}_expert_closed_loop"] = np.stack(reference_x)
            for eps in eps_list:
                for rep in range(args.repeats):
                    error = probe.restore(env, snap)
                    entry["restore_errors"].append(error)
                    if error > 1e-5:
                        raise RuntimeError(f"Restore error {error} m")
                    delta = None
                    if eps > 0:
                        u = rng.standard_normal(env.action_dim)
                        delta = eps * u / np.linalg.norm(u)
                    before = time.perf_counter()
                    rows, xs, executed = run(fixed, delta)
                    if eps == 0:
                        assert np.array_equal(executed, fixed)
                    entry["runs"].append(dict(group="identical" if eps == 0 else "perturbed", eps=float(eps),
                                              repeat=rep, restore_error_m=error, seconds=time.perf_counter() - before,
                                              first_delta=None if delta is None else delta.tolist(), trace=rows))
                    traces[f"s{target}_eps{eps:g}_r{rep}"] = xs
                    print(json.dumps(dict(step=target, eps=eps, repeat=rep, final_upperarm=rows[-1]["upperarm_ratio"],
                                          restore_error_m=error)), flush=True)
            traces[f"s{target}_fixed_commands"] = fixed
            np.savez_compressed(args.out / "traces.npz", **traces)
            result["snapshots"].append(entry)
            save()
            # Continue the approach from the snapshot with the expert in closed loop.
            error = probe.restore(env, snap)
            entry["restore_errors"].append(error)
            step = target
        result["completed"] = True
        save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
