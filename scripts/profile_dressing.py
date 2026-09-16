"""Measure actual dressing throughput without changing solver tolerances or action timing.

Run with the Genesis Python environment and PYTHONPATH=build_raw/python/src:python.
Homogeneous garment/body batches isolate scaling; they do not predict a mixed training world.
Optional native timers run after the throughput window because their synchronizations add cost.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from uipc_manip import pretrain_wang, train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.dressing_live import LiveCellFactory
from uipc_manip.genesis_env import _ensure_genesis
from uipc_manip.physics_gradient_probe import frame_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 8, 24])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--timer-steps", type=int, default=0)
    parser.add_argument("--garment", default="tshirt_26")
    parser.add_argument("--body", type=int, default=14049)
    parser.add_argument("--checkpoint", help="Profile this deterministic policy instead of scripted actions.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.envs) < 1 or args.steps < 1 or args.timer_steps < 0 or args.steps + args.timer_steps > 300:
        parser.error("Positive environment counts/steps and nonnegative timer steps required, at most 300 decisions total")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for n in args.envs:
        _, targs, _ = pretrain_wang.prepare(["teacher", "--region", "13", "--garments", args.garment,
                                            "--num-envs", str(n), "--obs-mode", "wang_static_arm", "--no-obs-augment"])
        cfg = train_sac.dressing_config(targs)
        _ensure_genesis(cfg.logging_level)
        cfg = replace(cfg, cells=tuple((args.garment, args.body) for _ in range(n)), contact_force_readout=False,
                      workspace=str(args.out.parent / (args.out.stem + "_assets")))
        t0 = time.perf_counter()
        env = GenesisIPCDressingEnv(cfg, num_envs=n, cell_factory=LiveCellFactory(cfg.live))
        build_s = time.perf_counter() - t0
        try:
            obs = env.reset(list(range(n)))
            agent = None
            if args.checkpoint:
                from uipc_manip.physics_gradient_actor import load_agent
                agent = load_agent(Path(args.checkpoint), env.spec.point_budget, env.action_dim, "cuda")
            timers, steps, actions_s = {}, [], []

            def wrap(name):
                original = getattr(env, name)

                def call(*a, **kw):
                    start = time.perf_counter()
                    try:
                        return original(*a, **kw)
                    finally:
                        timers[name] = timers.get(name, 0.0) + time.perf_counter() - start

                setattr(env, name, call)

            for name in ("_sim_step", "observation", "_progress", "_update_targets", "positions"):
                wrap(name)
            report = {"n": n, "build_s": build_s, "arguments": vars(args)}
            report["measurement_start_s"] = time.perf_counter()
            for i in range(args.steps + args.timer_steps):
                if i == args.steps:
                    import uipc
                    uipc.Timer.report_as_json()  # Clear setup and uninstrumented frames.
                    uipc.Timer.enable_all()
                t0 = time.perf_counter()
                action = env.scripted_actions() if agent is None else agent.act(obs, deterministic=True)
                actions_s.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                obs, _, _, infos = env.step(action)
                duration = time.perf_counter() - t0
                if any(info.get("sim_error") for info in infos):
                    raise RuntimeError(str(infos))
                if i < args.steps:
                    steps.append(duration)
                if i == args.steps - 1:
                    report.update(step_s=steps, action_s=sum(actions_s), timers_inclusive=dict(timers),
                                  transitions_per_s=n * len(steps) / sum(steps), last_frame_stats=frame_stats(env),
                                  final_upperarm=[info.get("upperarm_ratio") for info in infos],
                                  measurement_end_s=time.perf_counter())
                if i >= args.steps:
                    report.setdefault("native_timers", []).append(uipc.Timer.report_as_json())
            print(f"{n} environments: {report['transitions_per_s']:.3f} simulator transitions/s", flush=True)
            reports.append(report)
            args.out.write_text(json.dumps(reports, indent=2, default=str) + "\n")
        finally:
            if args.timer_steps:
                import uipc
                uipc.Timer.disable_all()
            env.close()
            del env
            gc.collect()


if __name__ == "__main__":
    main()
