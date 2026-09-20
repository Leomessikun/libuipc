"""How different must two actions be before this simulator can tell them apart?

ADR 0009 proposes a policy-improvement operator that compares candidate actions from
one exactly restored state and only takes a step when the comparison resolves above the
environment's own noise. That operator is worthless if nothing resolves, so this
measures the resolution directly, before anything is implemented.

At a state the policy visits, the world is snapshotted. The base action is the policy's
own; a candidate is that action displaced by ``sigma`` in a random direction and held
for ``hold`` decisions before the policy resumes. Every candidate, the base included, is
repeated with identical commands from the identical restored state, so the spread of a
candidate against itself is the noise floor and the gap between two candidates' means is
the signal. ``sigma = 0`` repeats the base action and measures the floor alone.

Four scores are read from the same branches, because the choice of score is a design
variable and the task metric is not obviously the best one:

* the task metric, sustained upper-arm coverage;
* the opening's geometric progress along the arm axis, which is continuous and needs no
  ray cast;
* whether the opening threaded the arm within the window, the milestone probability the
  ADR proposes as the potential;
* the summed task reward, for reference.

What comes out per score is a curve of margin over noise against sigma. The smallest
sigma whose margin clears the noise is the action resolution of this environment under
that score, and it sizes the operator.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import physics_gradient_probe as probe  # noqa: E402
from uipc_manip import train_sac  # noqa: E402

OPENING_CENTER = slice(13, 16)
"""Where the opening centre sits in the 35-float privileged vector (dressing_privileged.py)."""


def axis_progress(center: np.ndarray, finger: np.ndarray, shoulder: np.ndarray) -> float:
    """Distance of the opening centre along the finger-to-shoulder axis, in metres.

    Continuous everywhere, including across the threading event that the task reward is
    flat over, and free of the ray cast whose single-decision reading is coarse.
    """
    axis = np.asarray(shoulder, dtype=np.float64) - np.asarray(finger, dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length < 1e-9:
        return 0.0
    return float((np.asarray(center, dtype=np.float64) - finger) @ (axis / length))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--snapshot-steps", default="100,140,180")
    p.add_argument("--sigmas", default="0,0.25,0.5,1.0", help="Action displacement, normalized units")
    p.add_argument("--holds", default="1,4", help="Decisions the candidate action is held")
    p.add_argument("--window", type=int, default=16, help="Decisions per branch, inside the predictability horizon")
    p.add_argument("--repeats", type=int, default=6)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=7701)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    sigmas = [float(s) for s in args.sigmas.split(",") if s.strip()]
    holds = [int(h) for h in args.holds.split(",") if h.strip()]
    if 0.0 not in sigmas:
        raise ValueError("sigma 0 is the noise floor and must be measured")
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.physics_gradient_actor import load_agent
    from uipc_manip.sac import SACAgent

    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False, workspace=str(args.out / "assets"))
    if steps[-1] + args.window >= cfg.horizon:
        raise ValueError("The last branch must end before the episode's time limit")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=args.slots, cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, checkpoint=str(args.checkpoint), cell=args.cell, env=cfg.to_dict(),
                  snapshot_steps=steps, sigmas=sigmas, holds=holds, window=args.window,
                  repeats=args.repeats, slots=args.slots, seed=args.seed,
                  physical_decisions=0, states=[], records=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        rng = np.random.default_rng(args.seed)
        obs = env.reset([args.seed + i for i in range(args.slots)])
        step = 0
        for target in steps:
            while step < target:
                obs, _, done, info = env.step(agent.act(obs, deterministic=True))
                result["physical_decisions"] += args.slots
                step += 1
                if np.any(done) or any(r.get("sim_error") for r in info):
                    raise RuntimeError(f"Approach terminated at decision {step}")
            snapshot = probe.take_snapshot(env, f"step{target}", target)
            base_action = np.asarray(agent.act(obs, deterministic=True), dtype=np.float64)
            priv0 = env.privileged().copy()
            fingers = [np.asarray(c.finger, dtype=np.float64) for c in env.cells]
            shoulders = [np.asarray(c.shoulder, dtype=np.float64) for c in env.cells]
            start_axis = [axis_progress(priv0[i][OPENING_CENTER], fingers[i], shoulders[i])
                          for i in range(args.slots)]
            # One displacement direction per state, shared by every sigma, so the curve
            # against sigma is a curve along one line and not a cloud of directions.
            direction = rng.standard_normal((args.slots, env.action_dim))
            direction /= np.linalg.norm(direction, axis=1, keepdims=True)
            ids = [f"{garment}__{body}__seed{args.seed + i}__step{target}" for i in range(args.slots)]
            for i in range(args.slots):
                result["states"].append(dict(state=ids[i], slot=i, step=target,
                                             upperarm_ratio=float(info[i]["upperarm_ratio"]),
                                             tracking_error=float(info[i]["tracking_error"]),
                                             axis_progress=start_axis[i],
                                             base_action=[float(x) for x in base_action[i]],
                                             direction=[float(x) for x in direction[i]]))
            for hold in holds:
                for sigma in sigmas:
                    for repeat in range(args.repeats):
                        error = probe.restore(env, snapshot)
                        if error > 1e-5:
                            raise RuntimeError(f"Restore error {error} m")
                        branch = env.observation()
                        candidate = np.clip(base_action + sigma * direction, -1.0, 1.0)
                        traces = [[] for _ in range(args.slots)]
                        for t in range(args.window):
                            action = candidate if t < hold else agent.act(branch, deterministic=True)
                            branch, _, done, rows = env.step(
                                np.clip(action, -1, 1).astype(np.float32), reset_on_done=False)
                            result["physical_decisions"] += args.slots
                            if np.any(done) or any(r.get("sim_error") for r in rows):
                                raise RuntimeError(f"Branch sigma={sigma} hold={hold} ended at {t}")
                            priv = env.privileged()
                            for i in range(args.slots):
                                traces[i].append(dict(
                                    upperarm=float(rows[i]["upperarm_ratio"]),
                                    threaded=float(rows[i]["threaded"]),
                                    task_reward=float(rows[i]["task_reward"]) if "task_reward" in rows[i] else float("nan"),
                                    axis=axis_progress(priv[i][OPENING_CENTER], fingers[i], shoulders[i])))
                        tail = min(12, args.window)
                        for i in range(args.slots):
                            upper = np.asarray([r["upperarm"] for r in traces[i]])
                            axis = np.asarray([r["axis"] for r in traces[i]])
                            thread = np.asarray([r["threaded"] for r in traces[i]])
                            task = np.asarray([r["task_reward"] for r in traces[i]])
                            result["records"].append(dict(
                                state=ids[i], slot=i, step=target, hold=hold, sigma=sigma, repeat=repeat,
                                sustained_coverage=float(upper[-tail:].min()),
                                final_upperarm=float(upper[-1]),
                                axis_progress=float(axis[-1]),
                                axis_gain=float(axis[-1] - start_axis[i]),
                                threaded_within=float(thread.max()),
                                threaded_fraction=float(thread.mean()),
                                task_reward_sum=float(np.nansum(task))))
                        print(json.dumps(dict(step=target, hold=hold, sigma=sigma, repeat=repeat,
                                              axis_gain=[round(r["axis_gain"], 5)
                                                         for r in result["records"][-args.slots:]])), flush=True)
                        save()
            probe.restore(env, snapshot)
            obs = env.observation()
            step = target
        result["completed"] = True
        save()
        print(json.dumps(dict(states=len(result["states"]), branches=len(result["records"]),
                              decisions=result["physical_decisions"],
                              minutes=round(result["seconds"] / 60, 1))), flush=True)
    finally:
        env.close()
        gc.collect()


if __name__ == "__main__":
    main()
