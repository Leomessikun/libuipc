"""Reuse completed RAL branches to compare duration and held-out selection.

No simulator calls. The source did not record whole-branch grasp validity, so
coverage here is never reported as verified dressing success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from uipc_manip.action_selector_audit import heldout_choices


def grasp_timing(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw)
    if not data["completed"]:
        raise ValueError("Require a completed full-episode collection")
    rows = []
    for episode in data["episodes"]:
        if episode["arm"] != "control":
            continue
        trace = episode["trace"]
        violations = [i + 1 for i, t in enumerate(trace) if t["tracking_error"] > 0.02]
        upper = [i + 1 for i, t in enumerate(trace) if t["upperarm_ratio"] > 0]
        rows.append(dict(seed=episode["seed"], first_grasp_violation=violations[0] if violations else None,
                         invalid_decisions=len(violations), first_upperarm=upper[0] if upper else None,
                         final_tracking_error=trace[-1]["tracking_error"]))
    return dict(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(), controls=rows,
                decision_seconds=data["env"]["dt"] * data["env"]["action_repeat"])


def analyse(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw)
    if not data["completed"]:
        raise ValueError("Require a completed collection")
    states = [s["state"] for s in data["states"]]
    candidates = [(h, s) for h in data["holds"] for s in data["sigmas"]]
    lookup = {(r["state"], r["hold"], r["sigma"], r["repeat"]): r for r in data["records"]}
    n, repeats = len(states), data["repeats"]
    base_index = candidates.index((1, 0.0))
    values = {}
    for metric in ("sustained_coverage", "final_upperarm", "axis_gain", "task_reward_sum"):
        values[metric] = np.array([[[lookup[state, h, s, r][metric] for r in range(repeats)]
                                   for h, s in candidates] for state in states])
    output = dict(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                  states=n, branches=len(lookup), original_physical_decisions=data["physical_decisions"],
                  original_seconds=data["seconds"], new_physical_decisions=0,
                  caveats=["No grasp record: coverage is not verified success.",
                           "Selection is at known states using other repeats; not a learned deployable selector.",
                           "Overlapping held-out folds are not independent samples.",
                           "All states already have nonzero upper-arm coverage.",
                           "Duration changes the behavior as well as signal size."], by_selection_metric={})
    for objective in ("sustained_coverage", "task_reward_sum"):
        score = values[objective]
        single = np.array([i for i, (h, _) in enumerate(candidates) if h == 1])
        four = np.array([i for i, (h, _) in enumerate(candidates) if h == 4])
        chosen = {
            "policy_one_step": np.full((n, repeats), base_index),
            "hold_base_four": np.full((n, repeats), candidates.index((4, 0.0))),
            "hold_four_sigma_one": np.full((n, repeats), candidates.index((4, 1.0))),
            "select_single_heldout": single[heldout_choices(score[:, single])],
            "select_four_heldout": four[heldout_choices(score[:, four])],
            "select_joint_heldout": heldout_choices(score),
        }
        table = {}
        for name, indices in chosen.items():
            table[name] = {}
            for metric, outcomes in values.items():
                selected = outcomes[np.arange(n)[:, None], indices, np.arange(repeats)[None]]
                gains = selected - outcomes[:, base_index]
                table[name][metric] = dict(mean=float(selected.mean()), gain=float(gains.mean()),
                                          positive_state_means=int((gains.mean(axis=1) > 0).sum()),
                                          states_gain_over_001_all_repeats=int((gains.min(axis=1) > 0.01).sum()),
                                          per_state_mean_gain=gains.mean(axis=1).tolist())
            table[name]["chosen_holds"] = {str(h): int(sum(candidates[c][0] == h for c in indices.flat))
                                            for h in data["holds"]}
        output["by_selection_metric"][objective] = table
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/uipc_manip/ral_calibration_20260920/t26_14046/result.json"))
    parser.add_argument("--episode-sources", type=Path, nargs="*", default=[
        Path("output/uipc_manip/recovery_intervention_20260919/t26_lift/result.json"),
        Path("output/uipc_manip/recovery_intervention_20260919/t392_forward/result.json")])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.source)
    result["full_episode_grasp_timing"] = [grasp_timing(p) for p in args.episode_sources]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for objective, table in result["by_selection_metric"].items():
        print(objective)
        for name, scores in table.items():
            print(name, {m: round(scores[m]["gain"], 6) for m in ("sustained_coverage", "task_reward_sum")}, scores["chosen_holds"])
