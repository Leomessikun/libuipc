"""Cross the existing actor prefixes with frozen recovery continuations.

Reuse the verified geometric teacher and trained BC checkpoints. No search,
gradient query or policy training. The old teacher result is a positive
control; a failed reproduction cannot be called a policy-transfer defect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from uipc_manip import train_sac
from uipc_manip.recovery_teacher import branch_world
from uipc_manip.sac import SACAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/uipc_manip/recovery_teacher_20260917"))
    parser.add_argument("--original", type=Path, default=Path("output/uipc_manip/expert_pretrain_20260917/bc/checkpoints/actor_final.pt"))
    parser.add_argument("--student", type=Path, default=Path("output/uipc_manip/recovery_teacher_20260917/learning/recovery_bc/checkpoints/actor_final.pt"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4197)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=1500)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if min(args.slots, args.repeats, args.max_seconds) <= 0:
        parser.error("Positive slots, repeats and wall budget required")
    source_path = args.source / "result.json"
    source = json.loads(source_path.read_text())
    verification = json.loads((args.source / "verification/result.json").read_text())
    if not source["completed"] or not source["admitted"] or not verification["completed"]:
        raise ValueError("This experiment requires the already-admitted recovery teacher")
    selected = source["selected"].copy()
    approach = int(verification["approach"])
    cell = tuple(verification["env"]["cells"][0])
    payload = SACAgent.read_checkpoint(args.original)
    student_payload = SACAgent.read_checkpoint(args.student)
    if payload["sac_config"] != student_payload["sac_config"]:
        raise ValueError("Actor architectures/configurations differ")
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    cfg = replace(train_sac.dressing_config(targs), cells=(cell,) * args.slots,
                  contact_force_readout=False, decision_watchdog=False)
    for key in ("dt", "action_repeat", "horizon", "max_translation", "max_rotation",
                "constraint_strength", "arm_erosion_m", "newton_tolerance", "linear_system_tolerance"):
        if cfg.to_dict()[key] != verification["env"][key]:
            raise ValueError(f"Teacher's environment contract changed: {key}")
    routes = (
        dict(name="original_bc", kind="checkpoint", checkpoint=str(args.original)),
        dict(name="recovery_bc", kind="checkpoint", checkpoint=str(args.student)),
        selected,
    )
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    paths = [source_path, args.source / "verification/result.json", args.original, args.student,
             Path(__file__), Path("python/uipc_manip/recovery_teacher.py")]
    result = dict(completed=False, cell=cell, approach=approach, seed=args.seed,
                  slots=args.slots, repeats=args.repeats, physical_decisions=0, prefixes={},
                  source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  protocol="Two frozen actors supply separate prefixes; from each restored state compare both actors "
                           "and the previously selected teacher through decision 300. Same continuation command caps. "
                           "No fitting, search, SAC update, or success-based checkpoint selection.")

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        for name, checkpoint in (("original_bc", args.original), ("recovery_bc", args.student)):
            run = branch_world(cfg, checkpoint, args.out / name, routes, approach, args.slots, args.seed,
                               repeats=args.repeats, deadline=started + args.max_seconds)
            result["prefixes"][name] = dict(initial_coverage=run["initial_coverage"],
                                           prefix_tracking_m=run["prefix_tracking_m"],
                                           summaries=run["summaries"], seconds=run["seconds"],
                                           physical_decisions=run["physical_decisions"])
            result["physical_decisions"] += run["physical_decisions"]
            save()
        result["completed"] = True
        save()
    except BaseException as error:
        result["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
