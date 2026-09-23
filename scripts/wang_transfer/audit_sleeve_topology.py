"""Audit recorded IPC sleeve trajectories with Newton's semantic ring checks.

This reads saved cloth states; it does not rerun the simulator or use plots.
The retention gate is a geometric diagnostic, not a real-world safety test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


NEWTON_ROOT = Path(__file__).resolve().parents[3] / "newton"


def load_topology_module(path: Path):
    spec = importlib.util.spec_from_file_location("newton_dressing_topology", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load topology module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_held(progress: np.ndarray, threshold: float, hold: int) -> int | None:
    for k in np.flatnonzero(progress >= threshold):
        if k + hold < len(progress) and np.all(progress[k:k + hold + 1] >= threshold):
            return int(k)
    return None


def tensor_list(value: torch.Tensor) -> list[float]:
    return [round(float(x), 5) for x in value.detach().cpu().tolist()]


def topology_at(module, positions: torch.Tensor, rings: list[np.ndarray],
                wrist: torch.Tensor, elbow: torch.Tensor, shoulder: torch.Tensor) -> dict:
    topology = module.multi_section_topology(
        [positions[ring] for ring in rings], wrist, elbow, shoulder)
    retention = module.evaluate_sleeve_retention(topology)
    engaged = (topology["w"].abs() >= .5) & (topology["coverage"] >= .55)
    engaged_s = topology["s"][engaged]
    engaged_span = float((engaged_s.max() - engaged_s.min()).item()) if len(engaged_s) > 1 else 0.
    return {
        "retained": bool(retention["retained"].item()),
        "contiguous_wrapped_rings": int(retention["contiguous_wrapped_rings"].item()),
        "progress_span": round(float(retention["progress_span"].item()), 5),
        "engaged_ring_count": int(engaged.count_nonzero().item()),
        "engaged_span": round(engaged_span, 5),
        "first_two_engaged": bool(engaged[:2].all().item()),
        "s": tensor_list(topology["s"]),
        "w": tensor_list(topology["w"]),
        "coverage": tensor_list(topology["coverage"]),
        "radial_mean": tensor_list(topology["radial_mean"]),
    }


@torch.no_grad()
def audit_episode(path: Path, semantics_path: Path, module, rings: list[np.ndarray],
                  cuff: np.ndarray, hold: int) -> dict:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        if metadata.get("collision_geometry") != "full_body" or "human_vertices" not in data:
            raise ValueError(f"Full-body recording required: {path}")
        positions = data["positions"]
        if max(int(r.max()) for r in rings) >= positions.shape[1]:
            raise ValueError(f"Sleeve semantics exceed cloth vertex count: {path}")
        if not np.array_equal(data["opening_idx"], cuff):
            raise ValueError(f"Cuff semantic indices do not match recording: {path}")
        wrist = torch.as_tensor(data["finger"], dtype=torch.float32)
        elbow = torch.as_tensor(data["elbow"], dtype=torch.float32)
        shoulder = torch.as_tensor(data["shoulder"], dtype=torch.float32)
        upper = first_held(data["upperarm_ratio"], .7, hold)
        forearm = first_held(data["forearm_ratio"], .5, hold)
        milestones = {"initial": 0, "forearm_ratio_held": forearm,
                      "upperarm_ratio_held": upper, "final": len(positions) - 1}
        examined = {}
        retained_states = []
        max_engaged = 0
        max_engaged_span = 0.
        for k in range(len(positions)):
            result = topology_at(module, torch.as_tensor(positions[k]), rings, wrist, elbow, shoulder)
            if result["retained"]:
                retained_states.append(k)
            max_engaged = max(max_engaged, result["engaged_ring_count"])
            max_engaged_span = max(max_engaged_span, result["engaged_span"])
            if k in milestones.values():
                examined[k] = result
        return {
            "source": str(path.resolve()),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "body": metadata["body"],
            "rotation_mode": metadata.get("rotation_mode", "off"),
            "cloth_strain_rate_override": metadata.get("cloth_strain_rate_override"),
            "states": len(positions),
            "retained_state_count": len(retained_states),
            "first_retained_state": retained_states[0] if retained_states else None,
            "retained_through_upper_hold": (all(k in retained_states for k in range(upper, upper + hold + 1))
                                            if upper is not None else False),
            "max_engaged_ring_count": max_engaged,
            "max_engaged_span": round(max_engaged_span, 5),
            "milestones": {name: ({"state": k, **examined[k]} if k is not None else None)
                           for name, k in milestones.items()},
            "semantics": str(semantics_path.resolve()),
        }


@torch.no_grad()
def audit_newton_raw(path: Path, module, rings: list[np.ndarray]) -> dict:
    # This is a trusted local experiment artifact produced by collect_demos.py.
    raw = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for episode in raw["episodes"]:
        landmarks = torch.as_tensor(raw["landmarks"][episode["env_slot"]], dtype=torch.float32)
        cloth = episode["cloth"]
        if max(int(r.max()) for r in rings) >= cloth.shape[1]:
            raise ValueError(f"Sleeve semantics exceed Newton cloth vertex count: {path}")
        initial = topology_at(module, torch.as_tensor(cloth[0].astype(np.float32)),
                              rings, landmarks[1], landmarks[2], landmarks[3])
        final = topology_at(module, torch.as_tensor(cloth[-1].astype(np.float32)),
                            rings, landmarks[1], landmarks[2], landmarks[3])
        rows.append({"episode": int(episode["episode"]), "outcome": episode["outcome"],
                     "human_id": int(episode["human_id"]), "states": len(cloth),
                     "initial": initial, "final": final})
    return {"source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "saved_episodes": len(rows),
            "strict_retained_final": sum(row["final"]["retained"] for row in rows),
            "episodes": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--newton-raw", type=Path, default=None,
                        help="Also audit final cloth states in Newton collect_demos raw_states.pt.")
    parser.add_argument("--newton-root", type=Path, default=NEWTON_ROOT)
    parser.add_argument("--semantics", type=Path, default=None)
    parser.add_argument("--hold", type=int, default=5)
    args = parser.parse_args()
    if not args.episodes and args.newton_raw is None:
        parser.error("Provide IPC episodes, --newton-raw, or both")
    if args.hold < 1:
        parser.error("--hold must be positive")
    semantics_path = (args.semantics or args.newton_root / "exts/newton_isaaclab_tasks/"
                      "newton_isaaclab_tasks/dressing/data/garment_semantics/"
                      "tshirt_26__s4.0000_auto_semantics.npz")
    topology_path = (args.newton_root / "exts/newton_isaaclab_tasks/"
                     "newton_isaaclab_tasks/dressing/mdp/topology.py")
    module = load_topology_module(topology_path)
    with np.load(semantics_path, allow_pickle=False) as source:
        rings = [source[f"right_sleeve_ring_{i:02d}"].astype(np.int64)
                 for i in range(int(source["right_ring_count"][0]))]
        cuff = source["right_cuff_loop_ordered"].astype(np.int64)
    torch.set_num_threads(1)
    episodes = []
    for path in args.episodes:
        row = audit_episode(path, semantics_path, module, rings, cuff, args.hold)
        episodes.append(row)
        upper = row["milestones"]["upperarm_ratio_held"]
        print(f"[audit] body={row['body']} upper_ratio_held={upper is not None} "
              f"retained_at_upper={upper['retained'] if upper else None} "
              f"retained_states={row['retained_state_count']}/{row['states']}", flush=True)
    report = {"topology_module": str(topology_path.resolve()),
              "topology_module_sha256": hashlib.sha256(topology_path.read_bytes()).hexdigest(),
              "semantics_sha256": hashlib.sha256(semantics_path.read_bytes()).hexdigest(),
              "hold": args.hold, "episodes": episodes}
    if args.newton_raw is not None:
        report["newton_raw"] = audit_newton_raw(args.newton_raw, module, rings)
        print(f"[audit] Newton saved={report['newton_raw']['saved_episodes']} "
              f"strict_retained_final={report['newton_raw']['strict_retained_final']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[complete] {len(episodes)} episodes -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
