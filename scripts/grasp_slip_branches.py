"""What can be done at the moment the grip starts to slip?

The grasp is the only condition measured to separate a successful dressing from a
failed one: over the teacher's own episodes, a tracking error above 1.5 cm occurs on
0.0 % of the decisions of the episodes that succeed and 40.2 % of those that fail, and
six of its sixteen failures are complete dressings disqualified by that error alone
(`2026-09-19-teacher-supervision.md`). Two supervisors that acted on it failed because
their response was permanent: once the error is above the threshold it stays there, so
the correction applied for the rest of the episode and suppressed the dressing.

This asks the narrower question the branch machinery can answer: at the first decision
where the error crosses a lower threshold — before the 2 cm limit, when the grip is
only beginning to move — does any bounded response keep the grip *and* the dressing?
The responses are bounded in time by construction, and the teacher resumes afterwards.
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

from uipc_manip import dressing_supervisor as ds  # noqa: E402
from uipc_manip import physics_gradient_probe as probe  # noqa: E402
from uipc_manip import train_sac  # noqa: E402

RESPONSES = (
    dict(name="continue", kind="teacher"),
    dict(name="pause", kind="hold"),
    dict(name="slow_half", kind="scale", factor=0.5),
    dict(name="slow_quarter", kind="scale", factor=0.25),
    dict(name="back_off", kind="reverse", factor=1.0),
    dict(name="axial_only", kind="project"),
)
"""Bounded answers to a slipping grip, all lasting the same window: carry on, stand
still, move at a fraction of the commanded speed, retrace the last command, or keep
only the component along the arm."""


def response_action(response: dict, teacher_action: np.ndarray, previous: np.ndarray,
                    axis: np.ndarray) -> np.ndarray:
    action = np.asarray(teacher_action, dtype=np.float64).copy()
    kind = response["kind"]
    if kind == "teacher":
        return action
    if kind == "hold":
        return np.zeros_like(action)
    if kind == "scale":
        return action * float(response["factor"])
    if kind == "reverse":
        out = np.zeros_like(action)
        out[:3] = -float(response["factor"]) * np.asarray(previous, dtype=np.float64)[:3]
        return out
    if kind == "project":
        action[:3] = float(action[:3] @ axis) * axis
        return action
    raise ValueError(f"Unknown response {kind}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Supplies the environment contract")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--garments", default="hospital_gown,tshirt_26,tshirt_4")
    p.add_argument("--bodies", default="14045,14046,14047,14048")
    p.add_argument("--trigger-cm", type=float, default=1.0, help="Tracking error that opens the branch")
    p.add_argument("--window", type=int, default=12, help="Decisions the response lasts")
    p.add_argument("--follow", type=int, default=60, help="Decisions the teacher gets back afterwards")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shoulder-extension-m", type=float, default=0.05)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    bodies = [int(b) for b in args.bodies.split(",") if b.strip()]
    garments = [g for g in args.garments.split(",") if g.strip()]
    cells = tuple((g, b) for g in garments for b in bodies)
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.sac import SACAgent

    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    base = train_sac.dressing_config(targs)
    cfg = replace(base, cells=cells, decision_watchdog=False, contact_force_readout=False,
                  workspace=str(args.out / "assets"),
                  reward=replace(base.reward, upperarm_extension_m=args.shoulder_extension_m))
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=len(cells), cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, env=cfg.to_dict(), cells=[list(c) for c in cells],
                  trigger_cm=args.trigger_cm, window=args.window, follow=args.follow,
                  repeats=args.repeats, responses=[r["name"] for r in RESPONSES],
                  physical_decisions=0, triggers=[], records=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        env.reset([args.seed + i for i in range(len(cells))])
        # Drive every slot until each has crossed the trigger, snapshotting the world the
        # first time any slot does; a slot that never slips contributes nothing.
        snapshot, triggered, previous = None, np.zeros(len(cells), dtype=bool), np.zeros((len(cells), 6))
        for step in range(cfg.horizon - args.window - args.follow):
            actions = env.scripted_actions()
            _, _, done, info = env.step(actions, reset_on_done=False)
            result["physical_decisions"] += len(cells)
            previous = np.asarray(actions, dtype=np.float64)
            if any(r.get("sim_error") for r in info):
                raise RuntimeError("Simulator failure during the approach")
            crossing = np.asarray([row["tracking_error"] * 100 >= args.trigger_cm for row in info])
            newly = crossing & ~triggered
            if newly.any() and snapshot is None:
                snapshot = probe.take_snapshot(env, "first_slip", step + 1)
                result["triggers"] = [dict(cell=f"{g}/{b}", slot=i, decision=step + 1,
                                           tracking_cm=float(info[i]["tracking_error"] * 100),
                                           upperarm_ratio=float(info[i]["upperarm_ratio"]),
                                           slipping=bool(crossing[i]))
                                      for i, (g, b) in enumerate(cells)]
                result["snapshot_decision"] = step + 1
                result["previous_action"] = previous.tolist()
                break
            triggered |= crossing
        if snapshot is None:
            raise RuntimeError("No slot crossed the trigger before the branch window")
        print(json.dumps(dict(snapshot=result["snapshot_decision"],
                              slipping=[t["cell"] for t in result["triggers"] if t["slipping"]])), flush=True)
        axes = [ds.sleeve_position(row)["axis"] for row in env.privileged()]
        for repeat in range(args.repeats):
            for response in RESPONSES:
                error = probe.restore(env, snapshot)
                if error > 1e-5:
                    raise RuntimeError(f"Restore error {error} m")
                traces: list[list[dict]] = [[] for _ in cells]
                for t in range(args.window + args.follow):
                    teacher = np.asarray(env.scripted_actions(), dtype=np.float64)
                    if t < args.window:
                        actions = np.stack([response_action(response, teacher[i], previous[i], axes[i])
                                            for i in range(len(cells))])
                    else:
                        actions = teacher
                    _, _, done, rows = env.step(np.clip(actions, -1, 1).astype(np.float32), reset_on_done=False)
                    result["physical_decisions"] += len(cells)
                    if any(r.get("sim_error") for r in rows):
                        raise RuntimeError(f"Simulator failure in {response['name']}")
                    for i in range(len(cells)):
                        traces[i].append({k: float(rows[i][k]) for k in
                                          ("upperarm_ratio", "forearm_ratio", "tracking_error")})
                for i, (garment, body) in enumerate(cells):
                    upper = np.asarray([row["upperarm_ratio"] for row in traces[i]])
                    tracking = np.asarray([row["tracking_error"] for row in traces[i]])
                    result["records"].append(dict(
                        cell=f"{garment}/{body}", slot=i, response=response["name"], repeat=repeat,
                        slipping=bool(result["triggers"][i]["slipping"]),
                        final_coverage=float(upper[-1]), sustained_coverage=float(upper[-12:].min()),
                        max_coverage=float(upper.max()), peak_tracking_cm=float(tracking.max() * 100),
                        end_tracking_cm=float(tracking[-1] * 100),
                        grasp_held=bool(tracking.max() <= 0.02)))
                print(json.dumps(dict(repeat=repeat, response=response["name"],
                                      coverage=[round(r["sustained_coverage"], 3) for r in result["records"][-len(cells):]],
                                      peak_tracking=[round(r["peak_tracking_cm"], 2) for r in result["records"][-len(cells):]])),
                      flush=True)
                save()
        result["completed"] = True
        save()
    finally:
        env.close()
        gc.collect()


if __name__ == "__main__":
    main()
