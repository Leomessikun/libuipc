"""Read-only integrity and outcome report for the finite physical repair pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_sim2real_audit import write_json


def analyze(out):
    protocol = json.loads((out / "protocol.json").read_text())
    batches = [json.loads(p.read_text()) for p in sorted((out / "completed").glob("*.json"))]
    static_keys = ("faces", "human_vertices", "human_faces", "arm_vertices", "arm_faces",
                   "grasp_indices", "initial_grasp_offsets", "opening_idx", "finger", "elbow", "shoulder")
    checks, outcomes = [], []
    for batch in batches:
        rows = batch["rows"]
        base = next(r for r in rows if r["feedback_repair"]["controller"]["kind"] == "base")
        with np.load(Path(base["run_dir"]) / base["path"], allow_pickle=False) as z:
            reference = {k: z[k] for k in (*static_keys, "positions", "tcp")}
        state = min(protocol["settings"]["start"], len(reference["positions"]) - 1)
        for row in rows:
            with np.load(Path(row["run_dir"]) / row["path"], allow_pickle=False) as z:
                mismatches = [k for k in static_keys if not np.array_equal(z[k], reference[k])]
                if not np.array_equal(z["tcp"][0], reference["tcp"][0]):
                    mismatches.append("initial_tcp")
                initial = float(np.linalg.norm(z["positions"][0] - reference["positions"][0], axis=1).max())
                common_state = min(state, len(z["positions"]) - 1)
                before = float(np.linalg.norm(z["positions"][common_state] - reference["positions"][common_state], axis=1).max())
                zero_force = not bool(np.any(z["policy_force"]))
                first = row["success_state"]
                # Recompute the pilot's physical-sleeve acceptance from saved
                # states, independently of the collector's accepted flag.
                accepted = False
                if first is not None and len(z["actions"]) >= first + 20:
                    proximal = z["sleeve_proximal_upper_fraction"][first:]
                    accepted = (len(proximal) >= 21 and bool(np.all(proximal >= .7))
                                and bool(np.all(z["sleeve_wrapped"][first:]))
                                and bool(np.all(z["grasp_valid"])) and row["sim_error"] is None)
                changed = np.flatnonzero(z["controller_id"] == 6)
                in_window = not len(changed) or (changed.min() >= protocol["settings"]["start"]
                             and changed.max() < sum(protocol["settings"][k] for k in ("start", "prefix_steps", "suffix_steps")))
                checks.append(dict(job_id=row["job_id"], controller=row["feedback_repair"]["controller"]["name"],
                                   static_mismatches=mismatches, initial_max_vertex_gap_m=initial,
                                   preintervention_state=common_state, preintervention_max_vertex_gap_m=before,
                                   zero_force_policy_input=zero_force, bounded_intervention=bool(in_window),
                                   success_rule_matches=bool(accepted == row["accepted"])))
            outcomes.append(dict(job_id=row["job_id"], phase=row["phase"], condition=row["condition"],
                                 body=row["body"], repeat=row["repeat"], controller=row["feedback_repair"]["controller"]["name"],
                                 accepted=row["accepted"], transitions=row["transitions"], failure=row["sim_error"],
                                 base_accepted=base["accepted"], delta=int(row["accepted"]) - int(base["accepted"]),
                                 response_available=row["feedback_repair"]["response_feature"] is not None))
    result = dict(completed_batches=len(batches), completed_attempts=len(outcomes), outcomes=outcomes, reset_checks=checks,
                  integrity_pass=bool(checks) and all(not c["static_mismatches"] and c["zero_force_policy_input"]
                                                     and c["bounded_intervention"] and c["success_rule_matches"] for c in checks),
                  max_initial_vertex_gap_m=max((c["initial_max_vertex_gap_m"] for c in checks), default=None),
                  max_preintervention_vertex_gap_m=max((c["preintervention_max_vertex_gap_m"] for c in checks), default=None),
                  interpretation="Matched-reset outcomes, not identical-state counterfactuals. Two repeated development cases cannot establish novelty, statistical significance, or hardware transfer.")
    if (out / "source_fit.json").exists():
        result["source_gate"] = json.loads((out / "source_fit.json").read_text())["source_gate"]
    write_json(out / "analysis.json", result)
    return {k: v for k, v in result.items() if k not in ("outcomes", "reset_checks")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("output/uipc_manip/feedback_repairs_20261005"))
    a = p.parse_args()
    print(json.dumps(analyze(a.out), indent=2))
