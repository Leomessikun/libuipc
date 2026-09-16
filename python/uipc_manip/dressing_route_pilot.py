"""Prepare or execute a bounded, sequential dressing route experiment.

Preparation is CPU-only. Execution delegates to expert_baseline in a fresh process
per candidate, records failures and wall time, and never launches training. This
is a first fixed route ablation, not an autonomous strategy synthesis system.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def candidates() -> list[dict]:
    """Equal episode caps; baseline repeats measure fresh-world reproducibility.

    B and C share the height search. Only C changes the outside route. All
    proposals are declared before running; there is no evaluation-driven tuning.
    """
    return [
        {"name": f"{arm}_{i}", "arm": arm,
         "params": ({"z_offset": 0.12} if arm == "baseline" else
                    {"z_offset": height, **({"outward_offset": 0.04} if arm == "route" else {})})}
        for arm in ("baseline", "parameters", "route")
        for i, height in enumerate((0.12, 0.09, 0.15))
    ]


def build_plan(out: Path, *, region: int = 13, garment: str = "tshirt_26",
               poses=(0, 1, 2), horizon: int = 300, seed: int = 0,
               timeout_s: float = 1800.0) -> dict:
    from .pretrain_wang import REGION_COUNT, TRAIN_POSES, region_configs

    poses = list(poses)
    if not 0 <= region < REGION_COUNT or not poses or len(set(poses)) != len(poses):
        raise ValueError("choose a valid region and distinct, nonempty training poses")
    if not set(poses) <= set(TRAIN_POSES):
        raise ValueError("pilot search uses training poses 0-44 only")
    if horizon <= 0 or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("horizon and timeout must be positive and finite")
    trials = []
    for candidate in candidates():
        name = candidate["name"]
        params_path = out / "parameters" / f"{name}.json"
        run_dir = out / name
        command = [
            sys.executable, "-m", "uipc_manip.expert_baseline",
            "--region", str(region), "--poses", "train", "--pose-ids", *map(str, poses),
            "--garments", garment, "--num-envs", "1", "--seed", str(seed),
            "--horizon", str(horizon), "--expert-params", str(params_path),
            "--save-observations", "--obs-mode", "wang_static_arm", "--no-obs-augment",
            "--out-dir", str(run_dir), "--work-dir", str(out / "workspace"),
        ]
        trials.append({**candidate, "params_path": str(params_path), "out_dir": str(run_dir),
                       "command": command, "timeout_s": timeout_s})
    return {
        "schema": 1, "status": "prepared_not_run", "region": region, "garment": garment,
        "poses": poses, "seed": seed, "horizon": horizon,
        "expected_cells": [list(c) for c in region_configs([region], [garment], poses)],
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("dressing_route_pilot.py", "dressing_heuristic.py", "expert_baseline.py")
        },
        "max_decisions_per_arm": 3 * len(poses) * horizon,
        "max_wall_s_per_arm": 3 * timeout_s,
        "criterion": "Existing final success; report early_turn/paper_filter separately. No metric changes.",
        "claim": "Fixed candidate screening only; requires native validation and frozen-program generalization.",
        "trials": trials,
    }


def summarize_trial(run_dir: Path, expected_cells: list[list]) -> dict:
    """Refuse partial grids and simulator errors instead of rewarding dropped cells."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = json.loads((run_dir / "records.json").read_text())
    expected = sorted((str(g), int(b)) for g, b in expected_cells)
    actual = sorted((str(r["garment"]), int(r["human"])) for r in records)
    ratios = [float(r["final_upperarm_ratio"]) for r in records]
    valid = (actual == expected and not manifest["dropped"]
             and all(not r["sim_error"] for r in records)
             and all(math.isfinite(r) for r in ratios))
    return {
        "valid": valid, "cell_count": len(records),
        "successes": sum(bool(r["success"]) for r in records),
        "paper_filter_passes": sum(bool(r["paper_filter"]) for r in records),
        "mean_final_upperarm_ratio": sum(ratios) / len(ratios) if ratios else None,
        "decisions": sum(int(r["length"]) for r in records),
        "dropped": manifest["dropped"],
        "env": manifest["env"],
    }


def write_plan(plan: dict, out: Path) -> None:
    # A fresh directory prevents an interrupted run from being mistaken for a new result.
    out.mkdir(parents=True, exist_ok=False)
    (out / "parameters").mkdir()
    for trial in plan["trials"]:
        Path(trial["params_path"]).write_text(json.dumps(trial["params"], indent=2) + "\n")
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


def run_trial(command: list[str], log, timeout_s: float) -> int:
    """Bound the whole trial, including any drape-baking subprocesses (Linux)."""
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as proc:
        try:
            return proc.wait(timeout=timeout_s)
        except BaseException:
            # Only this trial's process group; existing training jobs are unrelated.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise


def execute(plan: dict, out: Path) -> list[dict]:
    results = []
    for trial in plan["trials"]:
        start = time.monotonic()
        result = {"name": trial["name"], "arm": trial["arm"], "status": "running"}
        print(f"[route-pilot] {trial['name']}", flush=True)
        try:
            with (out / f"{trial['name']}.log").open("w") as log:
                returncode = run_trial(trial["command"], log, trial["timeout_s"])
            result["returncode"] = returncode
            if returncode != 0:
                result["status"] = "process_failed"
            else:
                summary = summarize_trial(Path(trial["out_dir"]), plan["expected_cells"])
                result.update(summary)
                result["status"] = "complete" if summary["valid"] else "invalid"
        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(status="error", error=str(exc))
        result["wall_s"] = time.monotonic() - start
        results.append(result)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"[route-pilot] {result['status']} after {result['wall_s']:.1f}s", flush=True)
        # Do not spend the rest of the budget after a crash, timeout, or invalid world.
        if result["status"] != "complete":
            break
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwritten.")
    parser.add_argument("--region", type=int, default=13)
    parser.add_argument("--garment", default="tshirt_26")
    parser.add_argument("--pose-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--run", action="store_true", help="Execute sequential native trials; default only writes a plan.")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    plan = build_plan(out, region=args.region, garment=args.garment, poses=args.pose_ids,
                      horizon=args.horizon, seed=args.seed, timeout_s=args.timeout_s)
    write_plan(plan, out)
    print(f"[route-pilot] wrote {out / 'plan.json'} ({len(plan['trials'])} trials)", flush=True)
    if args.run:
        results = execute(plan, out)
        plan["status"] = ("complete" if len(results) == len(plan["trials"])
                          and all(r["status"] == "complete" for r in results) else "incomplete")
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        if plan["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
