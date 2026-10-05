"""Audit completed trials and report a disclosed, secondary matched-reset analysis.

The frozen exact-state analysis remains unchanged. No physical rollouts are run.
Matched resets compare the same spawn/configuration followed by independent
settling; they do not assert identical post-settling hidden states.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from run_sim2real_audit import DEFAULT_OUT, POLICIES, exact_mcnemar, mean_ci, write_json

STATIC_ARRAYS = ("tcp", "human_vertices", "human_faces", "grasp_indices",
                 "initial_grasp_offsets", "faces", "arm_vertices", "opening_idx",
                 "finger", "shoulder", "elbow")
MATERIALS = {
    "friction_x2": {"friction": .6}, "youngs_x2": {"cloth_youngs": 12000.},
    "bending_x2": {"cloth_bending_stiffness": .2},
    "density_x1p5": {"cloth_density": 1125.},
    "thickness_x1p5": {"cloth_thickness": .000225},
}


def array_hash(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode())
        h.update(str((value.dtype.str, value.shape)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def analyze(out=DEFAULT_OUT):
    started = time.monotonic()
    protocol = json.loads((out / "protocol.json").read_text())
    strict = json.loads((out / "summary.json").read_text())
    trials, configs, positions = {}, {}, {}
    for file in sorted((out / "completed").glob("*.json")):
        job = json.loads(file.read_text())
        for row in job["rows"]:
            if row.get("no_legal_start"):
                continue
            key = row["condition"], row["policy"], row["body"]
            path = Path(row["log"]).with_suffix("") / row["path"]
            cfg = json.loads(path.with_name("config.json").read_text())
            configs[key] = cfg
            with np.load(path, allow_pickle=False) as f:
                values = {k: f[k][0].copy() if k == "tcp" else f[k].copy()
                          for k in STATIC_ARRAYS}
                positions[key] = f["positions"][0].copy()
            run = json.loads(path.parent.parent.joinpath("run.json").read_text())
            with np.load(path.parent.parent / "hang.npz") as f:
                values["hang"] = f[run["hang_key"]].copy()
            trials[key] = dict(row, static_arrays_sha256=array_hash(values))
    errors = []
    for key, row in trials.items():
        c, p, b = key
        ref = ("nominal", "fmvp_sim", b)
        base, cfg = configs[ref], configs[key]
        expected = dict(base["config"], **MATERIALS.get(c, {}))
        if cfg["config"] != expected:
            errors.append(dict(key=key, reason="configuration differs beyond declared intervention"))
        if cfg["placement"] != base["placement"]:
            errors.append(dict(key=key, reason="placement changed"))
        if row["static_arrays_sha256"] != trials[ref]["static_arrays_sha256"]:
            errors.append(dict(key=key, reason="static geometry/grasp/hang changed"))
        if row["seed"] != trials[ref]["seed"]:
            errors.append(dict(key=key, reason="environment seed changed"))
    rejected = {tuple(e["key"]) for e in errors}
    conditions = {}
    for c in strict["conditions"]:
        bodies = [b for b in protocol["measured_bodies"]
                  if all((c, p, b) in trials and (c, p, b) not in rejected for p in POLICIES)]
        entry = dict(matched_reset_blocks=len(bodies), comparisons={},
                     successes={p: sum(trials[c, p, b]["accepted"] for b in bodies) for p in POLICIES})
        max_gap, rms_gap = 0., 0.
        for b in bodies:
            for p, q in itertools.combinations(POLICIES, 2):
                distance = np.linalg.norm(positions[c, p, b] - positions[c, q, b], axis=1)
                max_gap = max(max_gap, float(distance.max()))
                rms_gap = max(rms_gap, float(np.sqrt(np.mean(distance ** 2))))
        entry["post_settle_max_vertex_gap_mm"] = max_gap * 1000
        entry["post_settle_max_rms_gap_mm"] = rms_gap * 1000
        for p in ("r1", "flow"):
            diff = [int(trials[c, p, b]["accepted"]) - int(trials[c, "fmvp_sim", b]["accepted"]) for b in bodies]
            paired = [b for b in bodies if all(("nominal", q, b) in trials and
                      ("nominal", q, b) not in rejected for q in POLICIES)]
            interaction = [(int(trials[c, p, b]["accepted"]) - int(trials[c, "fmvp_sim", b]["accepted"]))
                           - (int(trials["nominal", p, b]["accepted"]) - int(trials["nominal", "fmvp_sim", b]["accepted"])) for b in paired]
            wins, losses = diff.count(1), diff.count(-1)
            entry["comparisons"][p] = dict(wins=wins, losses=losses,
                exact_mcnemar_p=exact_mcnemar(wins, losses), gain=mean_ci(diff),
                gain_change_from_nominal=mean_ci(interaction), interaction_blocks=len(paired))
        conditions[c] = entry
    delta = []
    for key in positions:
        c, p, b = key
        distance = np.linalg.norm(positions[key] - positions["nominal", p, b], axis=1)
        delta.append(dict(condition=c, policy=p, body=b,
                          max_vertex_gap_mm=float(distance.max()) * 1000,
                          rms_gap_mm=float(np.sqrt(np.mean(distance ** 2))) * 1000))
    result = dict(
        analysis="Secondary analysis specified after noticing all exact-state hashes mismatch; not the frozen primary endpoint.",
        strict_exact_state_blocks={c: x["paired_units"] for c, x in strict["conditions"].items()},
        reset_design="Exact static arrays/hang/grasp/TCP/placement/configuration check, then independently settled trials. No outcome-dependent geometric tolerance.",
        validation_errors=errors, conditions=conditions, post_settle_change_from_nominal=delta,
        outcomes=[{k: row[k] for k in ("condition", "policy", "body", "accepted", "transitions", "initial_geometry_sha256", "static_arrays_sha256")} for row in trials.values()],
        repeatability=strict["repeatability"], cost=strict["cost"],
        nominal_gain_present={p: conditions["nominal"]["comparisons"][p]["gain"]["mean"] for p in ("r1", "flow")},
        limits=["Exact hidden-state counterfactual pairing failed; these are matched-reset block comparisons.",
                "Physics interventions include material-dependent settling before the policy starts.",
                "Nominal r1 gain is absent on the selected seven bodies; cannot estimate retention of the historical 201-unit gain.",
                "One attempt per case/condition/policy; stochastic sensitivity and parameter sensitivity are not isolated.",
                "One garment, seven bodies, no hardware. No significance, equivalence, or causal bottleneck claim."],
        cpu_wall_seconds=time.monotonic() - started)
    write_json(out / "matched_reset_summary.json", result)
    print(json.dumps({"validation_errors": errors, "conditions": conditions,
                      "cost": result["cost"], "cpu_wall_seconds": result["cpu_wall_seconds"]}, indent=2))
    return result


if __name__ == "__main__":
    analyze()
