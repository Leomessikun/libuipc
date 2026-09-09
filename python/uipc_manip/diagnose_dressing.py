"""Record reproducible dressing reachability probes and compare saved trajectories.

``audit`` reads the bake cache on CPU; ``rollout`` requires the real GPU
simulator. NPZ traces contain an initial frame followed by one frame per
action. Replaying a trace reuses its normalized actions, and verifies the
cell and action scaling before starting the simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path
import time

import numpy as np

from .dressing_assets import DressingCache, DressingCacheConfig
from .dressing_reward import WangRewardConfig, wang_progress


ACTION_CONFIG_KEYS = ("dt", "action_repeat", "max_ee_speed_m_s", "max_rotation", "clip_rotation_to_yz")


def cell_fingerprint(cell) -> str:
    digest = hashlib.sha256()
    for name in ("cloth", "faces", "opening_idx", "grasp_idx", "picker_idx", "alignment_idx",
                 "picker_pos", "arm_points", "arm_faces", "human_points", "finger", "elbow", "shoulder"):
        array = np.ascontiguousarray(getattr(cell, name))
        digest.update(name.encode())
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def progress(cell, cloth, cfg):
    return wang_progress(
        cloth, polygon_idx=cell.opening_idx, triangle_idx=cell.polygon_triangles(),
        cuff_idx=cell.cuff_idx, finger=cell.finger, elbow=cell.elbow,
        shoulder=cell.shoulder, human_points=cell.human_points, cfg=cfg,
    )


def validate_replay(metadata: dict, cell, config: dict) -> None:
    if metadata["cell_sha256"] != cell_fingerprint(cell):
        raise ValueError("Replay cell geometry or semantic indices differ from the source")
    for key in ACTION_CONFIG_KEYS:
        if metadata["config"][key] != config[key]:
            raise ValueError(f"Replay action scaling mismatch: {key}")


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _config(path, garment, human):
    from .dressing_env import DressingConfig
    from .dressing_obs import DressingObsConfig

    data = json.loads(Path(path).read_text()) if path else {}
    data = data.get("config", data)
    data.pop("max_translation", None)  # Derived field in saved descriptions.
    allowed = {f.name for f in fields(DressingConfig)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown dressing configuration fields: {sorted(unknown)}")
    if "obs" in data:
        data["obs"] = DressingObsConfig(**data["obs"])
    if "reward" in data:
        data["reward"] = WangRewardConfig(**data["reward"])
    if "cache" in data:
        data["cache"] = DressingCacheConfig(**{k: Path(v) for k, v in data["cache"].items()})
    data.update(garments=(garment,), human=human)
    return DressingConfig(**data)


def audit(args):
    cfg = _config(args.config, args.garment, args.human)
    cell = DressingCache(cfg.cache).load(args.garment, args.human)
    pr = progress(cell, cell.cloth, cfg.reward)
    anchors = cell.anchor_indices(cfg.anchor_count)
    return {
        "mode": "cached_state_cpu_audit", "cell": cell.name,
        "cell_sha256": cell_fingerprint(cell), "config": cfg.to_dict(),
        "anchor_indices": anchors.tolist(), "opening_indices": cell.opening_idx.tolist(),
        "initial_forearm_ratio": pr.forearm_ratio, "initial_upperarm_ratio": pr.upperarm_ratio,
        "initial_reward": pr.reward,
        "anchor_to_arm_vertex_distance_m": float(np.linalg.norm(cell.arm_points - cell.picker_pos, axis=1).min()),
        "opening_radius_mean_m": cell.opening_radius_mean,
        "note": "Cached geometry audit only; no simulation or reachability conclusion.",
    }


def rollout(args):
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_heuristic import HeuristicDressingPolicy

    cfg = _config(args.config, args.garment, args.human)
    cell = DressingCache(cfg.cache).load(args.garment, args.human)
    replay = None
    if args.actions_from:
        source = Path(args.actions_from)
        validate_replay(json.loads(source.with_suffix(".json").read_text()), cell, cfg.to_dict())
        with np.load(source, allow_pickle=False) as data:
            replay = data["actions"].copy()
        if replay.ndim != 2 or replay.shape[1] != 6 or not np.isfinite(replay).all() or np.abs(replay).max() > 1:
            raise ValueError("Replay actions must be finite normalized actions shaped (steps, 6)")
        if len(replay) < args.steps:
            raise ValueError("Source trace has fewer actions than requested steps")
    requested_horizon = cfg.horizon
    # Keep the last recorded frame: env.step() otherwise auto-resets at horizon.
    cfg.horizon = max(cfg.horizon, args.steps + 1)
    cfg.workspace = str(Path(args.out).resolve() / "world")
    cfg.augment_obs = False
    env = GenesisIPCDressingEnv(cfg, num_envs=1)
    arrays = {key: [] for key in ("actions", "positions", "tcp", "targets", "stages")}
    rows = []
    metadata = {
        "schema_version": 1, "simulator": "libuipc", "cell": cell.name,
        "cell_sha256": cell_fingerprint(cell), "config": cfg.to_dict(),
        "requested_horizon": requested_horizon,
        "policy": "action_replay" if replay is not None else args.policy,
        "replay_source": str(args.actions_from) if args.actions_from else None,
        "source_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("dressing_env.py", "dressing_heuristic.py", "dressing_reward.py", "dressing_assets.py")
        },
    }

    def capture():
        arrays["positions"].append(env.positions()[0])
        arrays["tcp"].append(env._anchor[0].copy())
        arrays["targets"].append(env._pickers[0]["targets"].copy())

    try:
        env.reset([cfg.seed])
        expert = HeuristicDressingPolicy(env)
        metadata["anchor_indices"] = env._pickers[0]["anchor_idx"].tolist()
        metadata["snapshot_tracking_error_m"] = env.snapshot_tracking_error
        capture()
        for step in range(args.steps):
            stage = expert.stage_names()[0] if args.policy == "heuristic" and replay is None else metadata["policy"]
            action = replay[step:step + 1] if replay is not None else (
                expert.actions() if args.policy == "heuristic" else np.zeros((1, 6), dtype=np.float32)
            )
            before = env._anchor[0].copy()
            start = time.monotonic()
            _, rewards, dones, infos = env.step(action)
            info = infos[0]
            if info.get("sim_error"):
                metadata["error"] = info["error"]
                break
            if dones[0]:
                raise RuntimeError("Unexpected reset while capturing diagnostic state")
            capture()
            arrays["actions"].append(action[0].copy())
            arrays["stages"].append(stage)
            command = action[0, :3] * cfg.max_translation
            executed = env._anchor[0] - before
            row = {key: info[key] for key in (
                "upperarm_ratio", "forearm_ratio", "tracking_error", "early_turn", "collision", "task_reward", "success"
            )}
            row.update(step=step + 1, stage=stage, reward=float(rewards[0]),
                       command_translation_m=float(np.linalg.norm(command)),
                       executed_translation_m=float(np.linalg.norm(executed)),
                       blocked_translation_m=float(np.linalg.norm(command - executed)),
                       wall_seconds=time.monotonic() - start)
            rows.append(row)
            if (step + 1) % args.report_every == 0:
                print(json.dumps(row), flush=True)
    finally:
        env.close()
        out = Path(args.out)
        np.savez_compressed(out / "trace.npz", **{key: np.asarray(value) for key, value in arrays.items()})
        metadata["steps_recorded"] = len(rows)
        metadata["completed"] = len(rows) == args.steps
        metadata["metrics"] = rows
        if rows:
            metadata["summary"] = {
                "final_forearm_ratio": rows[-1]["forearm_ratio"],
                "final_upperarm_ratio": rows[-1]["upperarm_ratio"],
                "max_upperarm_ratio": max(r["upperarm_ratio"] for r in rows),
                "max_tracking_error_m": max(r["tracking_error"] for r in rows),
                "blocked_decisions": sum(r["blocked_translation_m"] > 1e-7 for r in rows),
            }
        _write_json(out / "trace.json", metadata)
    return {key: metadata[key] for key in ("cell", "steps_recorded", "completed", "summary") if key in metadata}


def compare_traces(left: Path, right: Path) -> dict:
    lm = json.loads(left.with_suffix(".json").read_text())
    rm = json.loads(right.with_suffix(".json").read_text())
    with np.load(left, allow_pickle=False) as l, np.load(right, allow_pickle=False) as r:
        count = min(len(l["actions"]), len(r["actions"]))
        if not count:
            raise ValueError("Cannot compare traces without recorded actions")
        same_shape = l["positions"].shape[1:] == r["positions"].shape[1:]
        return {
            "common_decisions": count,
            "same_cell": lm["cell_sha256"] == rm["cell_sha256"],
            "same_action_scaling": all(lm["config"][key] == rm["config"][key] for key in ACTION_CONFIG_KEYS),
            "same_actions": bool(np.array_equal(l["actions"][:count], r["actions"][:count])),
            "initial_cloth_max_difference_m": float(np.linalg.norm(l["positions"][0] - r["positions"][0], axis=-1).max()) if same_shape else None,
            "final_cloth_max_difference_m": float(np.linalg.norm(l["positions"][count] - r["positions"][count], axis=-1).max()) if same_shape else None,
            "tcp_max_difference_m": float(np.linalg.norm(l["tcp"][:count + 1] - r["tcp"][:count + 1], axis=-1).max()),
            "left_summary": lm.get("summary"), "right_summary": rm.get("summary"),
            "note": "Equal normalized actions alone do not establish equal initial state, contact, material, or solver settings.",
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "rollout", "compare"))
    parser.add_argument("--garment", default="tshirt_26")
    parser.add_argument("--human", type=int, default=0)
    parser.add_argument("--config", help="DressingConfig JSON or description containing config")
    parser.add_argument("--out", default="output/uipc_manip/diagnostic")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--policy", choices=("heuristic", "hold"), default="heuristic")
    parser.add_argument("--actions-from", help="Previously saved trace.npz; requires sibling trace.json")
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.report_every < 1:
        parser.error("--steps and --report-every must be positive")
    if args.mode == "compare":
        if args.left is None or args.right is None:
            parser.error("compare requires --left and --right")
        result = compare_traces(args.left, args.right)
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        if (Path(args.out) / "trace.npz").exists() and args.mode == "rollout":
            parser.error("Output already contains a trace; choose a new --out")
        result = audit(args) if args.mode == "audit" else rollout(args)
        _write_json(Path(args.out) / "summary.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
