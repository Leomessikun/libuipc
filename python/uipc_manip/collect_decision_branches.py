"""Collect counterfactual recovery branches at states a dressing policy visits.

For every snapshot step the policy is driven to that decision, the world is
snapshotted, and each macro of :data:`uipc_manip.decision_branches.MACROS` is executed
for a short window and then handed back to the *same* policy for the rest of the
branch. Every macro is repeated with identical commands, so the spread of a macro's
own outcome is measured alongside the difference between macros.

Slots share one world and one garment/body cell but start from different seeds, so
one run yields ``slots`` distinct decision states per snapshot step. Nothing is
trained here and no solver, reward or controller setting is changed.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import decision_branches as db
from . import physics_gradient_probe as probe


def state_id(cell: tuple[str, int], seed: int, step: int) -> str:
    return f"{cell[0]}__{cell[1]}__seed{seed}__step{step}"


def collect(cfg, checkpoint: Path, out: Path, *, snapshot_steps: list[int], window: int, follow: int,
            repeats: int, slots: int, seed: int, step_m: float, history: int, macros=db.MACROS,
            sustained: int | None = None) -> dict:
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_live import LiveCellFactory
    from .physics_gradient_actor import load_agent

    sustained = min(12, window + follow) if sustained is None else int(sustained)
    out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(replace(cfg, workspace=str(out / "assets")), num_envs=slots,
                                cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, env=cfg.to_dict(), checkpoint=str(checkpoint),
                  macros=[m["name"] for m in macros], snapshot_steps=snapshot_steps, window=window,
                  follow=follow, repeats=repeats, slots=slots, seed=seed, macro_step_m=step_m,
                  history=history, sustained_decisions=sustained, physical_decisions=0, restore_errors=[], states=[], records=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        agent = load_agent(checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        if agent.cfg.history_length != 1:
            raise ValueError("This collector drives a single-frame actor")
        seeds = [seed + i for i in range(slots)]
        obs = env.reset(seeds)
        past_obs: deque[np.ndarray] = deque(maxlen=history)
        past_actions: deque[np.ndarray] = deque(maxlen=history)
        arrays: dict[str, np.ndarray] = {}
        step = 0
        for target in snapshot_steps:
            while step < target:
                action = agent.act(obs, deterministic=True)
                past_obs.append(obs.copy())
                past_actions.append(action.copy())
                obs, _, done, info = env.step(action)
                result["physical_decisions"] += slots
                step += 1
                if np.any(done) or any(r.get("sim_error") for r in info):
                    raise RuntimeError(f"Policy approach terminated at decision {step}")
            if len(past_obs) < history:
                raise ValueError("Snapshot steps must leave room for the history window")
            snapshot = probe.take_snapshot(env, f"step{target}", target)
            privileged = env.privileged().copy()
            tools = env._anchor.copy()
            directions = []
            for i, cell in enumerate(env.cells):
                directions.append(db.slot_directions(tools[i], cell.finger, cell.elbow, cell.shoulder))
            ids = []
            for i in range(slots):
                sid = state_id((env.cells[i].garment, env.cells[i].human), seeds[i], target)
                ids.append(sid)
                result["states"].append(dict(state=sid, slot=i, seed=seeds[i], step=target,
                                             garment=env.cells[i].garment, human=int(env.cells[i].human),
                                             upperarm_ratio=float(info[i]["upperarm_ratio"]),
                                             forearm_ratio=float(info[i]["forearm_ratio"]),
                                             tracking_error=float(info[i]["tracking_error"]),
                                             tool=[float(x) for x in tools[i]],
                                             directions={k: [float(x) for x in v] for k, v in directions[i].items()}))
                arrays[f"{sid}__observation"] = obs[i].astype(np.float32)
                arrays[f"{sid}__privileged"] = privileged[i].astype(np.float32)
                arrays[f"{sid}__history_observations"] = np.stack(past_obs)[:, i].astype(np.float32)
                arrays[f"{sid}__history_actions"] = np.stack(past_actions)[:, i].astype(np.float32)
            np.savez_compressed(out / "states.npz", **arrays)
            for repeat in range(repeats):
                for macro in macros:
                    error = probe.restore(env, snapshot)
                    result["restore_errors"].append(error)
                    if error > 1e-5:
                        raise RuntimeError(f"Restore error {error} m exceeds ten micrometres")
                    branch_obs = env.observation()
                    traces: list[list[dict]] = [[] for _ in range(slots)]
                    commands = []
                    for t in range(window + follow):
                        policy_action = agent.act(branch_obs, deterministic=True)
                        if t < window:
                            action = np.stack([db.macro_action(macro, t, window, directions[i], policy_action[i],
                                                               step_m, cfg.max_translation) for i in range(slots)])
                        else:
                            action = policy_action
                        action = np.clip(action, -1.0, 1.0).astype(np.float32)
                        commands.append(action.copy())
                        branch_obs, _, done, rows = env.step(action, reset_on_done=False)
                        result["physical_decisions"] += slots
                        if np.any(done) or any(r.get("sim_error") for r in rows):
                            raise RuntimeError(f"Branch {macro['name']} terminated at step {t}")
                        for i in range(slots):
                            traces[i].append({k: rows[i][k] for k in
                                              ("upperarm_ratio", "forearm_ratio", "tracking_error", "grasp_valid",
                                               "collision_rejected_substeps", "tether_rejected_substeps",
                                               "commanded_translation_m", "accepted_anchor_translation_m")})
                    for i in range(slots):
                        record = dict(state=ids[i], macro=macro["name"], repeat=repeat, slot=i, step=target,
                                      restore_error_m=error, **db.branch_summary(traces[i], sustained))
                        record["trace"] = traces[i]
                        result["records"].append(record)
                    arrays[f"{ids[0]}__commands__{macro['name']}__r{repeat}"] = np.stack(commands)[:, 0].astype(np.float32)
                    print(json.dumps(dict(step=target, macro=macro["name"], repeat=repeat,
                                          sustained=[round(r["sustained_coverage"], 3) for r in result["records"][-slots:]])),
                          flush=True)
                    save()
            np.savez_compressed(out / "states.npz", **arrays)
            error = probe.restore(env, snapshot)
            result["restore_errors"].append(error)
            obs = env.observation()
            step = target
        result["completed"] = True
        save()
        return result
    finally:
        env.close()
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--snapshot-steps", default="100,140,180")
    p.add_argument("--window", type=int, default=8, help="Decisions executed by the macro")
    p.add_argument("--follow", type=int, default=32, help="Decisions returned to the base policy")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=3301)
    p.add_argument("--macro-step-m", type=float, default=0.008)
    p.add_argument("--history", type=int, default=5)
    # The solver's convergence settings decide how large the run-to-run seed is, and so
    # how fine a difference between two macros a repeat can resolve; they are an
    # independent variable of this collection, not a fixed property of the task.
    p.add_argument("--newton-tolerance", type=float, default=None)
    p.add_argument("--linear-tolerance", type=float, default=None)
    p.add_argument("--newton-max-iterations", type=int, default=None)
    return p


def main():
    from . import train_sac
    from .sac import SACAgent

    args = build_parser().parse_args()
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    if args.out.exists():
        raise FileExistsError(args.out)
    if min(args.window, args.follow, args.repeats, args.slots, args.history) < 1 or len(steps) < 1:
        raise ValueError("Positive window, follow, repeats, slots, history and snapshot steps required")
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    overrides = {k: v for k, v in (("newton_tolerance", args.newton_tolerance),
                                   ("linear_system_tolerance", args.linear_tolerance),
                                   ("newton_max_iterations", args.newton_max_iterations))
                 if v is not None}
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False, **overrides)
    if cfg.augment_obs:
        raise ValueError("Unaugmented observations are required so that a branch input is reproducible")
    if steps[-1] + args.window + args.follow >= cfg.horizon:
        raise ValueError("The last branch must end before the episode's time limit")
    if min(steps) < args.history:
        raise ValueError("The first snapshot must leave room for the history window")
    result = collect(cfg, args.checkpoint, args.out, snapshot_steps=steps, window=args.window,
                     follow=args.follow, repeats=args.repeats, slots=args.slots, seed=args.seed,
                     step_m=args.macro_step_m, history=args.history)
    print(json.dumps(dict(states=len(result["states"]), branches=len(result["records"]),
                          physical_decisions=result["physical_decisions"], seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    main()
