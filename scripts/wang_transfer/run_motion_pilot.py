"""Run a finite sequential pilot after the shared GPU becomes available.

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

from probe_arm_motion import ROOT, idle_gpu, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motions", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=750)
    p.add_argument("--onsets", type=float, nargs="+", default=[1., 8.])
    p.add_argument("--wait-for-gpu", type=float, default=7200.)
    p.add_argument("--wall-budget", type=float, default=7200., help="Seconds after the initial GPU wait, including subsequent waits")
    args = p.parse_args()
    if args.steps < 20 or min(args.wait_for_gpu, args.wall_budget) <= 0 or not np.isfinite([args.wait_for_gpu, args.wall_budget, *args.onsets]).all() or min(args.onsets) < 0:
        raise ValueError("Invalid decision budget, time budget, or motion onset")
    motions = [path.resolve(strict=True) for path in args.motions]
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
        idle_gpu(args.wait_for_gpu)
        deadline = time.monotonic() + args.wall_budget

        def case(name, motion, steps, onset, methods):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pilot wall-time budget exhausted")
            command = [sys.executable, "-u", str(Path(__file__).with_name("probe_arm_motion.py")),
                       "--motion", str(motion), "--out", str(out / name), "--steps", str(steps),
                       "--onset", str(onset), "--methods", *methods,
                       "--wait-for-gpu", str(min(remaining, args.wait_for_gpu))]
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
        for i, motion in enumerate(motions):
            name = f"smoke_{i}"
            result = case(name, motion, 40, 0., ["hold"])[0]
            with np.load(out / name / "hold.npz", allow_pickle=False) as data:
                excursion = float(np.linalg.norm(data["human_vertices"] - data["human_vertices"][0], axis=-1).max())
            if result["failure"] is not None or result["steps"] != 40 or excursion < .001:
                status.update(state="stopped_at_motion_gate", reason=f"{name}: physics must be valid and actual excursion >= 1 mm")
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
