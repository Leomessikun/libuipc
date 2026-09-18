"""Does repairing the teacher's measured failures make it dress more cells?

Runs the scripted dressing expert over a matrix of garment and body cells twice from the
same seeds: once as it is, and once with the supervisor that overrides it while it
stalls, drifts off the arm's centreline or strains the grasp. Nothing is trained.

The two arms share the world, the seeds, the controller and the reward, so the only
difference is the override. Both are scored by the project's coverage-and-grasp rule.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import dressing_supervisor as ds  # noqa: E402
from uipc_manip import train_sac  # noqa: E402
from uipc_manip.dressing_env import GenesisIPCDressingEnv  # noqa: E402
from uipc_manip.dressing_live import LiveCellFactory  # noqa: E402
from uipc_manip.recovery_intervention import episode_summary  # noqa: E402
from uipc_manip.sac import SACAgent  # noqa: E402

GARMENTS = ("hospital_gown", "tshirt_26", "tshirt_68", "tshirt_4", "tshirt_392")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Supplies the environment contract only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bodies", default="14045,14046,14047,14048,14049")
    p.add_argument("--garments", default=",".join(GARMENTS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decisions", type=int, default=300)
    p.add_argument("--shoulder-extension-m", type=float, default=0.05,
                   help="The corrected upper-arm ray origin the expert dataset was scored with")
    p.add_argument("--stall-window", type=int, default=20)
    p.add_argument("--stall-arc", type=float, default=0.02)
    p.add_argument("--containment", type=float, default=0.6)
    p.add_argument("--grasp-cm", type=float, default=1.2)
    p.add_argument("--hold", type=int, default=8)
    p.add_argument("--arms", default="teacher,supervised")
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    bodies = [int(b) for b in args.bodies.split(",") if b.strip()]
    garments = [g for g in args.garments.split(",") if g.strip()]
    cells = tuple((g, b) for g in garments for b in bodies)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    base = train_sac.dressing_config(targs)
    cfg = replace(base, cells=cells, contact_force_readout=False, decision_watchdog=False,
                  workspace=str(args.out / "assets"),
                  reward=replace(base.reward, upperarm_extension_m=args.shoulder_extension_m))
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=len(cells), cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, env=cfg.to_dict(), cells=[list(c) for c in cells], seed=args.seed,
                  decisions=args.decisions, episodes=[],
                  supervisor=dict(stall_window=args.stall_window, stall_arc=args.stall_arc,
                                  containment=args.containment, grasp_cm=args.grasp_cm, hold=args.hold))

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        for arm in [a for a in args.arms.split(",") if a.strip()]:
            env.reset([args.seed + i for i in range(len(cells))])
            supervisors = [ds.TeacherSupervisor(stall_window=args.stall_window, stall_arc=args.stall_arc,
                                                containment=args.containment, grasp_cm=args.grasp_cm,
                                                hold=args.hold) for _ in cells]
            traces: list[list[dict]] = [[] for _ in cells]
            reasons: list[list[str]] = [[] for _ in cells]
            for _ in range(args.decisions):
                actions = env.scripted_actions()
                if arm == "supervised":
                    privileged = env.privileged()
                    for i, supervisor in enumerate(supervisors):
                        actions[i], reason = supervisor.command(actions[i], privileged[i])
                        reasons[i].append(reason)
                _, _, done, infos = env.step(actions, reset_on_done=False)
                if any(r.get("sim_error") for r in infos):
                    raise RuntimeError(f"Simulator failure in the {arm} arm")
                for i, info in enumerate(infos):
                    traces[i].append({k: float(info[k]) for k in ("upperarm_ratio", "forearm_ratio", "tracking_error")}
                                     | dict(grasp_valid=bool(info["grasp_valid"])))
            for i, (garment, body) in enumerate(cells):
                summary = episode_summary(traces[i])
                counts = {r: reasons[i].count(r) for r in sorted(set(reasons[i]))} if reasons[i] else {}
                result["episodes"].append(dict(arm=arm, cell=f"{garment}/{body}", garment=garment, human=body,
                                               overrides=counts, **summary, trace=traces[i]))
            rows = [e for e in result["episodes"] if e["arm"] == arm]
            print(json.dumps(dict(arm=arm, episodes=len(rows), successes=sum(e["success"] for e in rows),
                                  dressed=sum(e["sustained_coverage"] >= 0.7 for e in rows),
                                  grasp_valid=sum(e["whole_episode_grasp_valid"] for e in rows),
                                  mean_sustained=round(float(np.mean([e["sustained_coverage"] for e in rows])), 4))),
                  flush=True)
            save()
        result["completed"] = True
        save()
    finally:
        env.close()


if __name__ == "__main__":
    main()
