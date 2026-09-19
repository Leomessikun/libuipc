"""How much is closed-loop control worth, and where along a dressing episode?

Every method tried on this task has been a policy that reads the observation at all
three hundred decisions. Nothing has measured whether the task needs that. This does,
by putting a plan and a policy through the identical disturbance and letting only one
of them see what happened.

At a decision the policy visits, the world is snapshotted and continued once without
interference; the commands of that continuation are the *plan*. From the same snapshot
four arms then take an identical kick — a random direction at the controller's step,
for a few decisions. ``open`` executes the rest of the plan regardless. ``closed``
hands back to the policy, which sees the disturbed state. ``undo`` spends the same
number of decisions commanding the kick's negation before handing back, so it is a
response that *knows* what the disturbance was. ``hold`` spends those decisions
commanding nothing, which is the control that separates reversing the disturbance
from merely spending decisions away from the plan.

The three arms bracket the value of feedback at that decision, and neither bound is a
statement about the policy alone:

* ``closed - open`` is a **lower bound** — what this learner recovers. It is small
  either because the task needs no feedback here or because the learner uses none.
* ``undo - open`` is an **upper-ish bound** — what an informed response recovers,
  available to no deployable policy but measurable here. ``undo - hold`` says how
  much of that bound is the reversal itself rather than the pause.
* ``undo - closed`` is the **headroom**: what a better decision at this moment would
  be worth. A decision-sparse learner can only gain where this is positive.

All three are read against a control in which the plan is replayed from the same
snapshot with no kick at all, which measures the simulator's own irreproducibility,
and against the kick's own effect ``plan - open``: where the kick does not move the
outcome beyond the noise floor there is nothing to recover from, and the value of
feedback at that decision is undefined rather than zero.
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

from uipc_manip import decision_branches as db  # noqa: E402
from uipc_manip import physics_gradient_probe as probe  # noqa: E402
from uipc_manip import train_sac  # noqa: E402

TRACE_KEYS = ("upperarm_ratio", "forearm_ratio", "tracking_error", "grasp_valid",
              "collision_rejected_substeps", "tether_rejected_substeps",
              "commanded_translation_m", "accepted_anchor_translation_m")


def kick_commands(directions: np.ndarray, scale: float, action_dim: int) -> np.ndarray:
    """One normalized command per slot: a translation along ``directions``, no rotation."""
    out = np.zeros((len(directions), action_dim), dtype=np.float64)
    out[:, :3] = directions * scale
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--snapshot-steps", default="20,50,80,110,140,170,200,230")
    p.add_argument("--kick", type=int, default=4, help="Decisions the disturbance lasts")
    p.add_argument("--kick-m", type=float, default=0.008, help="Commanded metres per kicked decision")
    p.add_argument("--follow", type=int, default=60, help="Decisions after the kick")
    p.add_argument("--repeats", type=int, default=2, help="Independent kick directions per state")
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=5501)
    p.add_argument("--sustained", type=int, default=12)
    p.add_argument("--shoulder-extension-m", type=float, default=0.05)
    args = p.parse_args()
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    if args.out.exists():
        raise FileExistsError(args.out)
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.physics_gradient_actor import load_agent
    from uipc_manip.sac import SACAgent

    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    base = train_sac.dressing_config(targs)
    cfg = replace(base, cells=((garment, int(body)),) * args.slots, contact_force_readout=False,
                  decision_watchdog=False, workspace=str(args.out / "assets"),
                  reward=replace(base.reward, upperarm_extension_m=args.shoulder_extension_m))
    if cfg.augment_obs:
        raise ValueError("Unaugmented observations are required for a reproducible branch input")
    if steps[-1] + 2 * args.kick + args.follow >= cfg.horizon:
        raise ValueError("The last branch must end before the episode's time limit")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=args.slots, cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, checkpoint=str(args.checkpoint), cell=args.cell, env=cfg.to_dict(),
                  snapshot_steps=steps, kick=args.kick, kick_m=args.kick_m, follow=args.follow,
                  repeats=args.repeats, slots=args.slots, seed=args.seed, sustained=args.sustained,
                  physical_decisions=0, states=[], records=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    scale = float(args.kick_m) / float(cfg.max_translation)
    if scale > 1.0 + 1e-9:
        raise ValueError("The kick exceeds the controller's per-decision limit")
    try:
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        if agent.cfg.history_length != 1:
            raise ValueError("This profile drives a single-frame actor")
        rng = np.random.default_rng(args.seed)
        seeds = [args.seed + i for i in range(args.slots)]
        obs = env.reset(seeds)
        step = 0

        def continuation(snapshot, kick_dirs, plan, label, response=None):
            """Run one branch from ``snapshot``.

            ``kick_dirs`` disturbs the first ``kick`` decisions; ``response`` spends the
            next ``kick`` either reversing it or commanding nothing; ``plan`` replaces the
            policy for the rest. The branch is always ``kick + follow`` decisions long,
            whichever arm it is, so the arms are matched in decisions as well as in state.
            """
            error = probe.restore(env, snapshot)
            if error > 1e-5:
                raise RuntimeError(f"Restore error {error} m")
            branch = env.observation()
            traces = [[] for _ in range(args.slots)]
            commands = []
            for t in range(args.kick + args.follow):
                if kick_dirs is not None and t < args.kick:
                    action = kick_commands(kick_dirs, scale, env.action_dim)
                elif response == "undo" and t < 2 * args.kick:
                    action = kick_commands(-kick_dirs, scale, env.action_dim)
                elif response == "hold" and t < 2 * args.kick:
                    action = np.zeros((args.slots, env.action_dim))
                elif plan is not None:
                    action = plan[t]
                else:
                    action = agent.act(branch, deterministic=True)
                action = np.clip(action, -1.0, 1.0).astype(np.float32)
                commands.append(action.copy())
                branch, _, done, rows = env.step(action, reset_on_done=False)
                result["physical_decisions"] += args.slots
                if np.any(done) or any(r.get("sim_error") for r in rows):
                    raise RuntimeError(f"Branch {label} terminated at step {t}")
                for i in range(args.slots):
                    traces[i].append({k: rows[i][k] for k in TRACE_KEYS})
            return traces, np.stack(commands)

        def record(traces, target, ids, arm, repeat, kicked):
            for i in range(args.slots):
                result["records"].append(dict(state=ids[i], slot=i, step=target, arm=arm, repeat=repeat,
                                              kicked=bool(kicked),
                                              **db.branch_summary(traces[i], args.sustained)))
            rows = result["records"][-args.slots:]
            print(json.dumps(dict(step=target, arm=arm, repeat=repeat, kicked=bool(kicked),
                                  sustained=[round(r["sustained_coverage"], 3) for r in rows],
                                  tracking_cm=[round(100 * r["max_tracking_error"], 2) for r in rows],
                                  grasp_held=int(sum(r["whole_branch_grasp_valid"] for r in rows)))),
                  flush=True)
            save()

        for target in steps:
            while step < target:
                obs, _, done, info = env.step(agent.act(obs, deterministic=True))
                result["physical_decisions"] += args.slots
                step += 1
                if np.any(done) or any(r.get("sim_error") for r in info):
                    raise RuntimeError(f"Approach terminated at decision {step}")
            snapshot = probe.take_snapshot(env, f"step{target}", target)
            ids = [f"{garment}__{body}__seed{seeds[i]}__step{target}" for i in range(args.slots)]
            for i in range(args.slots):
                result["states"].append(dict(state=ids[i], slot=i, seed=seeds[i], step=target,
                                             garment=garment, human=int(body),
                                             upperarm_ratio=float(info[i]["upperarm_ratio"]),
                                             forearm_ratio=float(info[i]["forearm_ratio"]),
                                             tracking_error=float(info[i]["tracking_error"])))
            # The plan: what the policy itself would have done from here, undisturbed.
            plan_traces, plan = continuation(snapshot, None, None, "plan")
            record(plan_traces, target, ids, "plan", -1, False)
            # The control: the same plan replayed from the same snapshot, no disturbance.
            # Its distance from the plan is the simulator's irreproducibility, nothing else.
            control_traces, _ = continuation(snapshot, None, plan, "replay")
            record(control_traces, target, ids, "replay", -1, False)
            for repeat in range(args.repeats):
                kick_dirs = rng.standard_normal((args.slots, 3))
                kick_dirs /= np.linalg.norm(kick_dirs, axis=1, keepdims=True)
                for i in range(args.slots):
                    result["states"][-args.slots + i].setdefault("kicks", []).append(
                        [float(x) for x in kick_dirs[i]])
                open_loop, _ = continuation(snapshot, kick_dirs, plan, f"open r{repeat}")
                record(open_loop, target, ids, "open", repeat, True)
                closed, _ = continuation(snapshot, kick_dirs, None, f"closed r{repeat}")
                record(closed, target, ids, "closed", repeat, True)
                undone, _ = continuation(snapshot, kick_dirs, None, f"undo r{repeat}", response="undo")
                record(undone, target, ids, "undo", repeat, True)
                held, _ = continuation(snapshot, kick_dirs, None, f"hold r{repeat}", response="hold")
                record(held, target, ids, "hold", repeat, True)
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
