"""Extract FMVP-driven trajectories from recorded Genesis+IPC episodes.

Each output contains obs[0:k+1] and actions[0:k], where k is the first
verified milestone state. Upper-arm prefixes take priority; otherwise retain
the forearm-threading prefix. Subsequent scripted-expert and hold actions never
enter the extracted dataset. ``actions`` are executed commands and can include
recorded deterministic speed scaling; ``policy_actions`` are the raw FMVP
proposals. Simulated load is a screen, not physical safety certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


STATE_KEYS = {"obs", "positions", "tcp", "gripper_force", "arm_force",
              "upperarm_ratio", "forearm_ratio"}
ACTION_KEYS = {"actions", "policy_actions", "rewards", "executed_translation",
               "tracking_error", "early_turn", "grasp_valid", "speed_scale",
               "controller_id"}
STATIC_KEYS = {"faces", "arm_vertices", "arm_faces", "opening_idx", "finger",
               "shoulder", "elbow"}
OPTIONAL_STATE_KEYS = {"body_force"}
OPTIONAL_STATIC_KEYS = {"human_vertices", "human_faces"}


def candidate(data, progress_key: str, threshold: float, hold: int,
              max_peak_N: float | None,
              max_edge_p99_ratio: float | None = None) -> tuple[dict | None, str]:
    progress = data[progress_key]
    crossings = np.flatnonzero(progress >= threshold)
    if not crossings.size:
        return None, "threshold_not_reached"
    if crossings[0] < 1:
        return None, "threshold_at_initial_state"
    complete = crossings[crossings + hold < len(progress)]
    if not complete.size:
        return None, "insufficient_validation_states"
    stable = [int(k) for k in complete if np.min(progress[k:k + hold + 1]) >= threshold]
    if not stable:
        return None, "threshold_not_held"
    k = stable[0]
    if not np.all(data["controller_id"][:k] == 0):
        return None, "non_checkpoint_action_before_milestone"
    if not np.all(data["grasp_valid"][:k]):
        return None, "invalid_grasp_before_milestone"
    force = np.linalg.norm(data["gripper_force"][1:k + 1], axis=1)
    peak = float(force.max())
    if max_peak_N is not None and peak > max_peak_N:
        return None, "simulated_load_cutoff"
    strain = {}
    if max_edge_p99_ratio is not None:
        faces = np.asarray(data["faces"])
        edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                                   faces[:, [2, 0]])), axis=1), axis=0)
        positions = np.asarray(data["positions"][:k + 1], dtype=np.float64)
        initial = np.linalg.norm(positions[0, edges[:, 0]] - positions[0, edges[:, 1]], axis=1)
        if np.any(initial <= 1e-9):
            return None, "degenerate_initial_cloth_edge"
        lengths = np.linalg.norm(positions[:, edges[:, 0]] - positions[:, edges[:, 1]], axis=-1)
        ratios = lengths / initial
        peak_p99 = float(np.percentile(ratios, 99, axis=1).max())
        strain = {"edge_ratio_p99_peak": peak_p99,
                  "edge_ratio_p99_at_milestone": float(np.percentile(ratios[-1], 99)),
                  "edge_ratio_max_peak": float(ratios.max()),
                  "edge_ratio_reference": "first recorded cloth state, not rest mesh"}
        if peak_p99 > max_edge_p99_ratio:
            return None, "cloth_edge_ratio_p99_cutoff"
    nonarm = {}
    if "body_force" in data.files:
        # Both arrays are resultant forces. This is a useful torso-contact
        # diagnostic, although forces at distinct vertices can cancel.
        torso = np.linalg.norm(data["body_force"][1:k + 1] - data["arm_force"][1:k + 1], axis=1)
        nonarm = {"nonarm_resultant_peak_N": float(torso.max()),
                  "nonarm_resultant_p90_N": float(np.percentile(torso, 90))}
    return {"milestone_state": k, "milestone_value": float(progress[k]),
            "validation_min": float(np.min(progress[k:k + hold + 1])),
            "validation_controller_ids": np.unique(data["controller_id"][k:k + hold]).astype(int).tolist(),
            "gripper_peak_N": peak, "gripper_p90_N": float(np.percentile(force, 90)),
            "action_scaling_applied": bool(np.any(np.abs(data["speed_scale"][:k] - 1.) > 1e-6)),
            "min_speed_scale": float(np.min(data["speed_scale"][:k])),
            **nonarm, **strain}, "accepted"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, nargs="+", required=True,
                   help="Collector run directories, each containing run.json and metrics.json.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--hold", type=int, default=5,
                   help="Recorded decisions that must retain the milestone after crossing.")
    p.add_argument("--max-gripper-peak-N", type=float, default=None,
                   help="Optional simulation-only gross-load screen; no real safety meaning.")
    p.add_argument("--max-edge-p99-ratio", type=float, default=None,
                   help="Reject a prefix if any frame's 99th-percentile cloth-edge length exceeds this multiple of its first recorded length.")
    p.add_argument("--topology-audit", type=Path, default=None,
                   help="Ring audit from audit_sleeve_topology.py; required for full-body extraction.")
    p.add_argument("--allow-unverified-topology", action="store_true",
                   help="Explicitly permit historical full-body extraction without a ring audit; outputs stay unverified.")
    args = p.parse_args()
    if args.hold < 1:
        p.error("--hold must be positive")
    if args.max_gripper_peak_N is not None and args.max_gripper_peak_N <= 0:
        p.error("--max-gripper-peak-N must be positive")
    if args.max_edge_p99_ratio is not None and args.max_edge_p99_ratio < 1:
        p.error("--max-edge-p99-ratio must be at least 1")
    if args.out.exists():
        p.error(f"output already exists: {args.out}")
    if args.topology_audit is not None and args.allow_unverified_topology:
        p.error("Choose a topology audit or explicitly allow unverified topology, not both")
    audit_rows = {}
    audit_hash = None
    if args.topology_audit is not None:
        audit = json.loads(args.topology_audit.read_text())
        if audit["hold"] != args.hold:
            p.error("Topology audit hold window differs from extractor hold window")
        audit_hash = hashlib.sha256(args.topology_audit.read_bytes()).hexdigest()
        for row in audit["episodes"]:
            source = Path(row["source"]).resolve()
            if source in audit_rows:
                p.error(f"Duplicate topology audit source: {source}")
            audit_rows[source] = row

    manifest = []
    seen = set()
    source_checkpoints = set()
    collision_geometries = set()
    cloth_density_overrides = set()
    cloth_strain_rate_overrides = set()
    rotation_modes = set()
    source_runs = []
    for root in args.sources:
        root = root.resolve()
        run = json.loads((root / "run.json").read_text())
        checkpoint = run.get("checkpoint_sha256")
        if not checkpoint:
            raise ValueError(f"No checkpoint hash in {root / 'run.json'}")
        source_checkpoints.add(checkpoint)
        collision_geometries.add(run.get("collision_geometry", "arm"))
        cloth_density_overrides.add(run.get("cloth_density"))
        cloth_strain_rate_overrides.add(run.get("cloth_strain_rate"))
        rotation_modes.add(run.get("rotation", "off"))
        source_runs.append((root, run))
    if (len(source_checkpoints) != 1 or len(collision_geometries) != 1
            or len(cloth_density_overrides) != 1 or len(cloth_strain_rate_overrides) != 1
            or len(rotation_modes) != 1):
        raise ValueError("Do not mix checkpoint weights, collision geometry, cloth material, or rotation mode in one prefix dataset")
    if "full_body" in collision_geometries and not audit_rows and not args.allow_unverified_topology:
        p.error("Full-body extraction requires --topology-audit; use --allow-unverified-topology only for legacy diagnostics")
    for root, run in source_runs:
        checkpoint = run["checkpoint_sha256"]
        for rec in json.loads((root / "metrics.json").read_text()):
            source = (root / rec["path"]).resolve()
            if source in seen:
                raise ValueError(f"Duplicate source episode: {source}")
            seen.add(source)
            if rec["variant"] == "expert":
                continue
            with np.load(source, allow_pickle=False) as data:
                if not STATE_KEYS | ACTION_KEYS | STATIC_KEYS <= set(data.files):
                    raise ValueError(f"Missing required fields: {source}")
                t = len(data["actions"])
                if any(len(data[key]) != t + 1 for key in STATE_KEYS):
                    raise ValueError(f"State/action alignment error: {source}")
                if any(len(data[key]) != t for key in ACTION_KEYS):
                    raise ValueError(f"Action alignment error: {source}")
                if any(len(data[key]) != t + 1 for key in OPTIONAL_STATE_KEYS & set(data.files)):
                    raise ValueError(f"Optional state/action alignment error: {source}")
                if ("human_vertices" in data.files) != ("human_faces" in data.files):
                    raise ValueError(f"Incomplete full-body mesh: {source}")
                upper, upper_reason = candidate(data, "upperarm_ratio", .7, args.hold,
                                                args.max_gripper_peak_N, args.max_edge_p99_ratio)
                forearm, forearm_reason = candidate(data, "forearm_ratio", .5, args.hold,
                                                    args.max_gripper_peak_N, args.max_edge_p99_ratio)
                audit_row = audit_rows.get(source)
                if audit_rows and audit_row is None:
                    raise ValueError(f"Missing topology audit for {source}")
                if audit_row is not None:
                    if audit_row["source_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
                        raise ValueError(f"Topology audit is stale for {source}")
                    if upper is not None:
                        inspected = audit_row["milestones"]["upperarm_ratio_held"]
                        if (inspected is None or inspected["state"] != upper["milestone_state"]
                                or not audit_row["retained_through_upper_hold"]):
                            upper, upper_reason = None, "sleeve_ring_retention_failed"
                    if forearm is not None:
                        inspected = audit_row["milestones"]["forearm_ratio_held"]
                        if (inspected is None or inspected["state"] != forearm["milestone_state"]
                                or not inspected["first_two_engaged"] or inspected["s"][0] < .2):
                            forearm, forearm_reason = None, "forearm_ring_engagement_failed"
                if upper is not None:
                    stage, chosen = "upperarm", upper
                elif forearm is not None:
                    stage, chosen = "forearm", forearm
                else:
                    manifest.append({"source": str(source), "selected": False,
                                     "upperarm_reason": upper_reason,
                                     "forearm_reason": forearm_reason})
                    continue

                k = chosen["milestone_state"]
                quality_class = ("ring_retained_upperarm" if audit_row is not None and stage == "upperarm"
                                 else "partial_cuff_progress" if audit_row is not None
                                 else "unverified_topology")
                target = args.out / root.name / rec["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                original = json.loads(str(data["metadata_json"]))
                meta = {"body": rec["body"], "seed": rec["seed"],
                        "variant": rec["variant"], **chosen, "stage": stage,
                        "success_state": k if stage == "upperarm" else None,
                        "transitions": k, "source_episode": str(source),
                        "source_episode_metrics": original,
                        "source_checkpoint_sha256": checkpoint,
                        "collision_geometry": run.get("collision_geometry", "arm"),
                        "cloth_density_override": run.get("cloth_density"),
                        "cloth_strain_rate_override": run.get("cloth_strain_rate"),
                        "rotation_mode": run.get("rotation", "off"),
                        "quality_class": quality_class,
                        "topology_audit_sha256": audit_hash,
                        "hang_key": run.get("hang_key", "k300"),
                        "placement_offset_mm": run.get("placement_offset_mm", [0., 0., 0.]),
                        "validation_decisions": args.hold,
                        "max_edge_p99_ratio": args.max_edge_p99_ratio,
                        "validation_source": "recorded states after milestone; controller IDs recorded separately; excluded from extracted actions",
                        "force_note": "simulator gripper-load ranking only; not a real-world safety limit"}
                arrays = {key: data[key][:k + 1] for key in STATE_KEYS}
                arrays.update({key: data[key][:k + 1] for key in OPTIONAL_STATE_KEYS & set(data.files)})
                arrays.update({key: data[key][:k] for key in ACTION_KEYS})
                arrays.update({key: data[key] for key in STATIC_KEYS})
                arrays.update({key: data[key] for key in OPTIONAL_STATIC_KEYS & set(data.files)})
                arrays["metadata_json"] = json.dumps(meta)
                np.savez_compressed(target, **arrays)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                manifest.append({"source": str(source), "path": str(target.resolve()),
                                 "sha256": digest, "selected": True, "body": rec["body"],
                                 "seed": rec["seed"], "variant": rec["variant"],
                                 "checkpoint_sha256": checkpoint,
                                 "collision_geometry": run.get("collision_geometry", "arm"),
                                 "cloth_density_override": run.get("cloth_density"),
                                 "cloth_strain_rate_override": run.get("cloth_strain_rate"),
                                 "rotation_mode": run.get("rotation", "off"),
                                 "quality_class": quality_class,
                                 "topology_audit_sha256": audit_hash,
                                 "hang_key": run.get("hang_key", "k300"),
                                 "placement_offset_mm": run.get("placement_offset_mm", [0., 0., 0.]),
                                 "stage": stage, "transitions": k, **chosen,
                                 "upperarm_reason": upper_reason,
                                 "forearm_reason": forearm_reason})
                print(f"[prefix] {stage} body={rec['body']} seed={rec['seed']} "
                      f"variant={rec['variant']} transitions={k} -> {target}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    selected = [item for item in manifest if item["selected"]]
    result = {"hold": args.hold, "max_gripper_peak_N": args.max_gripper_peak_N,
              "max_edge_p99_ratio": args.max_edge_p99_ratio,
              "checkpoint_sha256": next(iter(source_checkpoints), None),
              "collision_geometry": next(iter(collision_geometries), None),
              "cloth_density_override": next(iter(cloth_density_overrides), None),
              "cloth_strain_rate_override": next(iter(cloth_strain_rate_overrides), None),
              "rotation_mode": next(iter(rotation_modes), None),
              "topology_audit_sha256": audit_hash,
              "total_source_episodes": len(manifest),
              "selected_episodes": len(selected),
              "selected_upperarm": sum(item["stage"] == "upperarm" for item in selected),
              "selected_forearm": sum(item["stage"] == "forearm" for item in selected),
              "selected_transitions": sum(item["transitions"] for item in selected),
              "episodes": manifest}
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[complete] {result['selected_episodes']}/{result['total_source_episodes']} "
          f"FMVP-driven prefixes ({result['selected_upperarm']} upper arm, "
          f"{result['selected_forearm']} forearm), {result['selected_transitions']} transitions",
          flush=True)


if __name__ == "__main__":
    main()
