"""Run a finite sequential pilot with explicit GPU scheduling.

Stop if moving-body physics or the static r1 baseline fails its gate. These
are feasibility diagnostics, not a powered evaluation or a training launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from probe_arm_motion import ENDPOINTS, ROOT, idle_gpu, save_json


def motion_smoke_gate(result, archive, expected_steps=40):
    """Body-drive validity is distinct from the stationary gripper's task failure."""
    with np.load(archive, allow_pickle=False) as data:
        human = data["human_vertices"]
    finite = bool(np.isfinite(human).all())
    excursion = float(np.linalg.norm(human - human[0], axis=-1).max()) if finite else None
    failure = result.get("failure")
    tracking = result["body_tracking_max_m"]
    passed = (result.get("body_motion_valid", False) and result["steps"] == expected_steps
              and len(human) == expected_steps + 1 and finite and excursion >= .001
              and np.isfinite(tracking) and tracking <= result["body_tracking_tolerance_m"]
              and (failure is None or failure["kind"] == "invalid_grasp"))
    return dict(passed=bool(passed), excursion_m=excursion, completed_steps=result["steps"],
                grasp_failure=result.get("grasp_failure"),
                scope="Full body-drive diagnostic only; a failed grasp remains invalid task supervision")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motions", type=Path, nargs="+", required=True)
    p.add_argument("--smoke-motions", type=Path, nargs="*", default=[],
                   help="Additional clips for body-drive validation only; policy comparisons still use --motions.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=750)
    p.add_argument("--onsets", type=float, nargs="+", default=[1., 8.])
    p.add_argument("--endpoint", choices=ENDPOINTS, default="interior_armhole")
    p.add_argument("--allow-shared-gpu", action="store_true",
                   help="Run alongside collection when GPU sharing is authorized.")
    p.add_argument("--after-status", type=Path,
                   help="Wait for this existing evaluation's complete/error/interrupted status before requesting GPU work.")
    p.add_argument("--wait-for-gpu", type=float, default=7200.)
    p.add_argument("--wall-budget", type=float, default=7200., help="Seconds after the initial GPU wait, including subsequent waits")
    args = p.parse_args()
    if args.steps < 20 or min(args.wait_for_gpu, args.wall_budget) <= 0 or not np.isfinite([args.wait_for_gpu, args.wall_budget, *args.onsets]).all() or min(args.onsets) < 0:
        raise ValueError("Invalid decision budget, time budget, or motion onset")
    motions = [path.resolve(strict=True) for path in args.motions]
    smoke_motions = list(dict.fromkeys([*motions, *(path.resolve(strict=True) for path in args.smoke_motions)]))
    if len(set(motions)) != len(motions) or len(set(args.onsets)) != len(args.onsets):
        raise ValueError("Motion paths and onsets must be unique")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = dict(state="waiting_for_gpu", arguments=vars(args), cases=[],
                  inference="No effect or generalization claim from a single body/seed pilot")

    def write_status():
        # Readers should see either the previous complete record or the new one.
        save_json(out / "status.tmp.json", status)
        (out / "status.tmp.json").replace(out / "status.json")

    write_status()
    try:
        if args.after_status is not None:
            predecessor = args.after_status.resolve(strict=True)
            status.update(state="waiting_for_predecessor", predecessor_status=str(predecessor))
            write_status()
            waiting_deadline = time.monotonic() + args.wait_for_gpu
            while True:
                previous = json.loads(predecessor.read_text())
                if previous["state"] in ("complete", "error", "interrupted"):
                    status["predecessor_final_state"] = previous["state"]
                    break
                if time.monotonic() >= waiting_deadline:
                    raise TimeoutError("Predecessor wait expired; its process and collection are untouched")
                time.sleep(min(20., max(0., waiting_deadline - time.monotonic())))
        status["gpu_scheduling"] = "shared" if args.allow_shared_gpu else "exclusive_idle_wait"
        write_status()
        if not args.allow_shared_gpu:
            idle_gpu(args.wait_for_gpu)
        deadline = time.monotonic() + args.wall_budget

        def case(name, motion, steps, onset, methods, *, motion_smoke=False):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pilot wall-time budget exhausted")
            command = [sys.executable, "-u", str(Path(__file__).with_name("probe_arm_motion.py")),
                       "--motion", str(motion), "--out", str(out / name), "--steps", str(steps),
                       "--onset", str(onset), "--methods", *methods,
                       "--endpoint", args.endpoint,
                       "--wait-for-gpu", str(min(remaining, args.wait_for_gpu))]
            if args.allow_shared_gpu:
                command.append("--allow-shared-gpu")
            if motion_smoke:
                command.append("--motion-smoke")
            record = dict(name=name, command=command, state="running")
            status["state"] = "running"
            status["cases"].append(record)
            write_status()
            print(f"[pilot] {name}", flush=True)
            started = time.monotonic()
            with (out / f"{name}.log").open("x") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            record.update(returncode=result.returncode, elapsed_s=time.monotonic() - started)
            if result.returncode:
                record["state"] = "failed"
                write_status()
                raise RuntimeError(f"Case {name} failed; inspect {out / (name + '.log')}")
            metrics = json.loads((out / name / "metrics.json").read_text())
            record.update(state="complete", metrics=metrics)
            write_status()
            return metrics

        # Each clip must actually move its simulated body before policy testing.
        for i, motion in enumerate(smoke_motions):
            name = f"smoke_{i}"
            result = case(name, motion, 40, 0., ["hold"], motion_smoke=True)[0]
            gate = motion_smoke_gate(result, out / name / "hold.npz")
            status["cases"][-1]["motion_gate"] = gate
            write_status()
            if not gate["passed"]:
                status.update(state="stopped_at_motion_gate", reason=f"{name}: full body motion must track within tolerance and actual excursion >= 1 mm")
                write_status()
                return
        baseline = case("static_r1", motions[0], args.steps, 1e6, ["r1"])[0]
        if not baseline["success"]:
            status.update(state="stopped_at_static_gate", reason="Static r1 did not maintain physical sleeve success; repair baseline before attributing failure to motion prediction")
            write_status()
            return
        for i, motion in enumerate(motions):
            for j, onset in enumerate(args.onsets):
                metrics = case(f"motion_{i}_onset_{j}", motion, args.steps, onset,
                               ["r1", "gicp", "oracle_pause", "causal_pause", "yoked_pause"])
                if any(row["failure"] and row["failure"]["kind"] == "invalid_physics" for row in metrics):
                    status.update(state="stopped_at_motion_gate", reason="Invalid moving-body physics in a policy case; repair before further comparisons")
                    write_status()
                    return
        status["state"] = "complete"
        write_status()
    except Exception as exc:
        status.update(state="error", error=repr(exc))
        if status["cases"] and status["cases"][-1]["state"] == "running":
            status["cases"][-1]["state"] = "failed"
        write_status()
        raise


if __name__ == "__main__":
    main()
