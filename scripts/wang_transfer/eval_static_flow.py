"""Finite paired static FMVP/flow diagnostics on held-out demonstration starts.

Select one validation recording per garment before simulation. Both methods
share a world with independently copied cells and the collector's completion
hold. This tests reproduction of known-feasible starts, not an unbiased task
success rate, unseen-garment transfer, or autonomous stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from probe_arm_motion import ROOT, idle_gpu, save_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_cases(data, checkpoint, out, source_root, garments, steps, noise):
    manifest = json.loads((data / "manifest.json").read_text())
    validation = [r for r in manifest["episodes"] if r["split"] == "validation"]
    training_bodies = {r["body"] for r in manifest["episodes"] if r["split"] == "train"}
    cases = []
    for garment in garments:
        row = next(r for r in validation if r["garment"] == garment)
        if row["body"] in training_bodies:
            raise ValueError("Evaluation body appears in student training data")
        source = Path(row["source_path"])
        run = json.loads((source.parent.parent / "run.json").read_text())
        package = Path(run["package_root"])
        if digest(package / "uipc_manip/dressing_env.py") != run["environment_sha256"]:
            raise ValueError("Source environment changed; pin it before interpreting reproduction")
        hang = Path(run["hang"])
        hang = hang if hang.is_absolute() else source_root / hang
        r1 = Path(run["checkpoint"])
        r1 = r1 if r1.is_absolute() else source_root / r1
        if digest(hang) != run["hang_sha256"] or digest(r1) != row["checkpoint_sha256"]:
            raise ValueError("Source hang or FMVP checkpoint changed")
        destination = out / garment
        command = [sys.executable, "-u", str(ROOT / "scripts/wang_transfer/collect_garment.py"),
                   "--package-root", str(package), "--checkpoint", str(r1),
                   "--flow-checkpoint", str(checkpoint), "--flow-noise", noise, "--policy-device", "cpu",
                   "--garment", garment, "--hang", str(hang), "--hang-key", run["hang_key"],
                   "--out", str(destination), "--variants", "baseline", "flow", "--bodies", str(row["body"]),
                   "--seed", str(row["seed"]), "--steps", str(steps)]
        for key in ("hold", "success", "success_geometry", "stop_proximal_upper", "body_fit_filter",
                    "fit_sleeve_ratio", "yaw", "rotation", "rotation_gain", "collision_geometry",
                    "abort_gripper_force", "cloth_density", "cloth_strain_rate"):
            value = run.get(key)
            if value is not None:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        command.extend(["--placement-offset-mm", *map(str, run["placement_offset_mm"])])
        if run.get("armhole_endpoint"):
            command.append("--armhole-endpoint")
        if run.get("profiles_json") is not None:
            raise ValueError("This diagnostic requires the default source controller profile")
        cases.append(dict(garment=garment, body=row["body"], seed=row["seed"], source_path=str(source),
                          source_training_arrays_sha256=row["training_arrays_sha256"], command=command,
                          state="pending"))
    return cases


def paired_initial_difference(directory, metrics):
    states = []
    for row in metrics:
        with np.load(directory / row["path"], allow_pickle=False) as data:
            states.append({key: data[key][0].copy() for key in ("positions", "tcp", "obs")})
    if len(states) != 2:
        raise ValueError("Expected exactly two matched controller slots")
    a, b = states
    return dict(cloth_max_m=float(np.linalg.norm(a["positions"] - b["positions"], axis=-1).max()),
                tcp_m=float(np.linalg.norm(a["tcp"] - b["tcp"])),
                observation_max_abs=float(np.abs(a["obs"] - b["obs"]).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("/home/ge47gax/kun/libuipc"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--garments", nargs="+", default=["tshirt_26", "tshirt_4", "tshirt_68", "tshirt_392", "hospital_gown"])
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--flow-noise", choices=("random", "zero"), default="random")
    parser.add_argument("--wait-for-gpu", type=float, default=3600.)
    parser.add_argument("--wall-budget", type=float, default=7200.)
    args = parser.parse_args()
    if min(args.steps, args.wall_budget) <= 0 or args.wait_for_gpu < 0 or len(set(args.garments)) != len(args.garments):
        raise ValueError("Invalid budget or duplicate garment")
    out, checkpoint = args.out.resolve(), args.flow_checkpoint.resolve(strict=True)
    cases = make_cases(args.data.resolve(), checkpoint, out, args.source_root, args.garments, args.steps, args.flow_noise)
    out.mkdir(parents=True, exist_ok=False)
    status = dict(state="waiting_for_gpu", cases=cases, checkpoint_sha256=digest(checkpoint),
                  data_manifest_sha256=digest(args.data / "manifest.json"),
                  hold_control="Same external completion hold for both methods",
                  scope="One validation demonstration start per garment; conditional reproduction diagnostic only")

    def write_status():
        save_json(out / "status.tmp.json", status)
        (out / "status.tmp.json").replace(out / "status.json")

    write_status()
    try:
        idle_gpu(args.wait_for_gpu)
        deadline = time.monotonic() + args.wall_budget
        for case in cases:
            idle_gpu(max(0., min(args.wait_for_gpu, deadline - time.monotonic())))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Paired evaluation wall budget exhausted")
            case["state"], status["state"] = "running", "running"
            write_status()
            print(f"[evaluate] {case['garment']} body {case['body']}", flush=True)
            started = time.monotonic()
            with (out / f"{case['garment']}.log").open("x") as log:
                result = subprocess.run(case["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
            case.update(returncode=result.returncode, elapsed_s=time.monotonic() - started)
            if result.returncode:
                raise RuntimeError(f"Collector failed for {case['garment']}; see its log")
            directory = out / case["garment"]
            metrics = json.loads((directory / "metrics.json").read_text())
            case.update(state="complete", metrics=metrics, initial_difference=paired_initial_difference(directory, metrics))
            write_status()
            print(json.dumps(dict(garment=case["garment"], results=[{k: r[k] for k in
                   ("variant", "accepted", "valid_grasp", "success_state", "transitions", "sim_error")} for r in metrics],
                   initial_difference=case["initial_difference"])), flush=True)
        status["state"] = "complete"
        write_status()
    except Exception as exc:
        status.update(state="error", error=repr(exc))
        for case in cases:
            if case["state"] == "running":
                case["state"] = "error"
        write_status()
        raise


if __name__ == "__main__":
    main()
