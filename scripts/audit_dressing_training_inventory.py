"""Recompute the 2026-09-20 inventory review from saved artifacts, without simulation.

Run from the repository root with PYTHONPATH=python and NumPy/SciPy installed.
Prints JSON; redirect to a new output file to retain the evidence and input hashes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from uipc_manip.dressing_reward import WangRewardConfig, wang_progress


def main():
    root = Path("output/uipc_manip")
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw.decode()

    result = {"scope": "Saved-artifact audit; no native simulation or policy training"}
    runs = {}
    for name in ("state_ub_clean_s1", "wang_teacher_r13_s1"):
        config = json.loads(read(root / name / "config.json"))
        env, sac = config["env"], config["sac_config"]
        period = env["dt"] * env["action_repeat"]
        evaluations = list(csv.DictReader(read(root / name / "eval_log.csv").splitlines()))
        training = list(csv.DictReader(read(root / name / "train_log.csv").splitlines()))
        keys = ("step", "success_rate", "valid_grasp_success_rate",
                "mean_max_upperarm_ratio", "mean_max_forearm_ratio",
                "mean_final_forearm_ratio", "mean_max_threaded", "mean_final_threaded")
        runs[name] = {
            "decision_seconds": period,
            "episode_seconds": period * env["horizon"],
            "translation_per_axis_mm": env["max_translation"] * 1000,
            "rotation_per_axis_deg": math.degrees(env["max_rotation"]),
            "rotation_per_axis_deg_per_second": math.degrees(env["max_rotation"]) / period,
            "discount": sac["discount"],
            "discount_efold_seconds": -period / math.log(sac["discount"]),
            "reward_scale": config["reward_scale"],
            "friction": env["friction"],
            "training_cells": len(env["cells"]),
            "evaluations": len(evaluations),
            "best_coverage_success_rate": max(float(r["success_rate"]) for r in evaluations),
            "first_evaluation": {k: evaluations[0].get(k) for k in keys},
            "last_evaluation": {k: evaluations[-1].get(k) for k in keys},
            "last_training": {k: training[-1].get(k) for k in
                              ("step", "transitions", "elapsed_s", "env_s", "eval_s")},
        }
    result["runs"] = runs

    rows = json.loads(read(root / "expert_r13_heldout_s0/records.json"))
    peak = np.array([r["max_upperarm_ratio"] >= .7 for r in rows])
    final = np.array([r["final_upperarm_ratio"] >= .7 for r in rows])
    grasp = np.array([r["max_tracking_error"] <= .02 for r in rows])
    result["teacher"] = {
        "total": len(rows), "peak_coverage": int(peak.sum()),
        "final_coverage": int(final.sum()), "final_coverage_and_grasp": int((final & grasp).sum()),
        "peak_coverage_and_grasp": int((peak & grasp).sum()),
        "covered_then_lost": int((peak & ~final).sum()),
        "final_coverage_but_invalid_grasp": int((final & ~grasp).sum()),
        "never_covered": int((~peak).sum()),
    }

    calibration = json.loads(read(root / "ral_calibration_20260920/t26_14046/result.json"))
    if not calibration["completed"]:
        raise ValueError("Calibration artifact is incomplete")
    groups = defaultdict(list)
    for row in calibration["records"]:
        groups[row["state"], row["hold"], row["sigma"]].append(row["task_reward_sum"])
    states = sorted({r["state"] for r in calibration["records"]})
    summaries = []
    for hold in (1, 4):
        for sigma in (.25, .5, 1.):
            differences, ratios = [], []
            for state in states:
                a, b = (np.array(groups[state, hold, s]) for s in (sigma, 0.))
                if len(a) != 5 or len(b) != 5:
                    raise ValueError(f"Expected five repeats: {state}, {hold}, {sigma}")
                delta = float(a.mean() - b.mean())
                sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))
                differences.append(delta)
                ratios.append(delta / sd if sd else float("nan"))
            summaries.append({
                "hold": hold, "sigma": sigma, "states": len(states),
                "median_signed_gain": float(np.median(differences)),
                "median_absolute_gain_over_sd": float(np.nanmedian(np.abs(ratios))),
                "positive_gain_states": int(np.sum(np.array(differences) > 0)),
                "gain_above_three_sd": int(np.sum(np.array(ratios) > 3)),
                "loss_below_minus_three_sd": int(np.sum(np.array(ratios) < -3)),
            })
    result["calibration"] = {
        "summaries": summaries,
        "declared_sigmas": calibration["sigmas"],
        "observed_sigmas": sorted({r["sigma"] for r in calibration["records"]}),
        "declared_snapshot_steps": calibration["snapshot_steps"],
        "observed_snapshot_steps": sorted({r["step"] for r in calibration["records"]}),
        "decisions_per_second": calibration["physical_decisions"] / calibration["seconds"],
        "seconds_per_480_decisions_at_observed_rate":
            480 * calibration["seconds"] / calibration["physical_decisions"],
    }

    indices = np.arange(6)
    triangles = np.array([[0, i, i + 1] for i in range(1, 5)])
    angles = np.arange(6) * np.pi / 3
    sweep = []
    for x in (-.002, -.001, .001, .002, .998, .999, 1.001, 1.002):
        cloth = np.stack([np.full(6, x), .1 * np.cos(angles), .1 * np.sin(angles)], axis=1)
        progress = wang_progress(
            cloth, polygon_idx=indices, triangle_idx=triangles, cuff_idx=indices,
            finger=np.array([0., 0, 0]), elbow=np.array([1., 0, 0]),
            shoulder=np.array([2., 0, 0]), human_points=np.array([[10., 10, 10]]),
            cfg=WangRewardConfig(),
        )
        sweep.append({"opening_x": x, "task_reward": progress.task_reward,
                      "total_reward": progress.reward})
    result["straight_arm_reward_counterexample"] = sweep
    read(Path("python/uipc_manip/dressing_reward.py"))
    result["input_sha256"] = hashes
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
