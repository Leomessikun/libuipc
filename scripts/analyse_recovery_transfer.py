"""Audit saved recovery continuations and frozen-actor command errors, without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from uipc_manip.physics_gradient_actor import load_agent
from uipc_manip.recovery_teacher import cap_commands, route_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.root / "summary.json").read_text())
    if not summary["completed"]:
        raise ValueError("Wait for the fixed comparison to complete")
    worlds = {name: json.loads((args.root / name / "result.json").read_text())
              for name in summary["prefixes"]}
    cfg = worlds["original_bc"]["env"]
    caps = (.008 / cfg["max_translation"], .05 / cfg["max_rotation"])
    torch.set_num_threads(1)
    routes = worlds["original_bc"]["routes"]
    actors = {r["name"]: load_agent(Path(r["checkpoint"]), cfg["point_budget"], 6, "cpu")
              for r in routes if r["kind"] == "checkpoint"}
    results = dict(matrix={}, episodes=[], command_fit={}, max_actor_replay_error=0.,
                   source_hashes={})
    traces = []

    def read_tape(path):
        results["source_hashes"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as tape:
            return {k: tape[k].copy() for k in ("obs", "actions", "step_metrics")}

    def predict(agent, obs):
        return np.concatenate([agent.act(obs[i:i + 32], deterministic=True)
                               for i in range(0, len(obs), 32)])

    teacher_sets = {}
    for prefix, world in worlds.items():
        assert world["completed"] and len(world["records"]) == 3 * summary["slots"] * summary["repeats"]
        assert world["physical_decisions"] == summary["slots"] * (
            summary["approach"] + 3 * summary["repeats"] * (cfg["horizon"] - summary["approach"]))
        assert max(world["restore_errors"]) <= 1e-5
        results["matrix"][prefix] = {}
        for route in routes:
            records = [r for r in world["records"] if r["route"] == route["name"]]
            assert {(r["slot"], r["repeat"]) for r in records} == {
                (s, r) for s in range(summary["slots"]) for r in range(summary["repeats"])}
            results["matrix"][prefix][route["name"]] = route_summary(records)
        teacher_sets[prefix] = []
        for record in world["records"]:
            tape = read_tape(args.root / prefix / record["path"])
            metrics = [json.loads(v) for v in tape["step_metrics"]]
            traces.append((prefix, record["route"], metrics))
            assert len(metrics) == cfg["horizon"] - summary["approach"]
            # The normal environment path omits this optional failure flag.
            assert not record["sim_error"] and all(not m.get("sim_error", False) for m in metrics)
            assert metrics[-1]["time_limit"] and not any(m["time_limit"] for m in metrics[:-1])
            assert np.isclose(min(m["upperarm_ratio"] for m in metrics[-12:]), record["sustained_coverage"])
            max_tracking = max(world["prefix_tracking_m"][record["slot"]],
                               max(m["tracking_error"] for m in metrics))
            assert np.isclose(max_tracking, record["max_tracking_error"])
            assert record["whole_episode_grasp_valid"] == (max_tracking <= .02)
            actions = tape["actions"]
            assert np.linalg.norm(actions[:, :3], axis=1).max() * cfg["max_translation"] <= .00800001
            assert np.linalg.norm(actions[:, 3:], axis=1).max() * cfg["max_rotation"] <= .05000001
            assert np.all(actions[:, 3] == 0)
            invalid = [summary["approach"] + i + 1 for i, m in enumerate(metrics) if m["tracking_error"] > .02]
            results["episodes"].append(dict(prefix=prefix, route=record["route"], slot=record["slot"],
                repeat=record["repeat"], sustained_coverage=record["sustained_coverage"],
                peak_coverage=max(m["upperarm_ratio"] for m in metrics),
                max_tracking_m=record["max_tracking_error"], first_branch_grasp_violation=min(invalid, default=None)))
            if record["route"] in actors:
                indices = [0, 59, 119, len(metrics) - 1]
                reconstructed = cap_commands(predict(actors[record["route"]], tape["obs"][indices]), *caps)
                error = float(np.abs(reconstructed - actions[indices]).max())
                results["max_actor_replay_error"] = max(results["max_actor_replay_error"], error)
                assert error < 1e-4, (prefix, record["path"], error)
            else:
                teacher_sets[prefix].append(tape)

    source = next(Path(p).parent for p in summary["source_hashes"] if p.endswith("verification/result.json"))
    old = json.loads((source / "result.json").read_text())
    teacher_sets["original_training_rows"] = [read_tape(source / r["path"]) for r in old["records"]
                                                if r["route"] == routes[-1]["name"]]
    for name, tapes in teacher_sets.items():
        obs = np.concatenate([t["obs"] for t in tapes])
        labels = np.concatenate([t["actions"] for t in tapes])
        moving = np.linalg.norm(labels, axis=1) > 1e-6
        results["command_fit"][name] = dict(rows=len(obs), moving_rows=int(moving.sum()), actors={})
        for actor_name, agent in actors.items():
            raw = predict(agent, obs)
            executed = cap_commands(raw, *caps)
            stats = {}
            for group, mask in (("all", np.ones(len(obs), bool)), ("moving_teacher", moving), ("stopped_teacher", ~moving)):
                if not mask.any():
                    continue
                delta = executed[mask] - labels[mask]
                stats[group] = dict(rows=int(mask.sum()),
                    raw_active_mse=float(np.square((raw - labels)[mask][:, [0, 1, 2, 4, 5]]).mean()),
                    capped_active_mse=float(np.square(delta[:, [0, 1, 2, 4, 5]]).mean()),
                    translation_rms_mm=float(np.sqrt(np.square(delta[:, :3]).sum(axis=1).mean()) * cfg["max_translation"] * 1000),
                    rotation_rms_deg=float(np.rad2deg(np.sqrt(np.square(delta[:, 3:]).sum(axis=1).mean()) * cfg["max_rotation"])))
            results["command_fit"][name]["actors"][actor_name] = stats
    (args.root / "analysis.json").write_text(json.dumps(results, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    colors = {"original_bc": "#636363", "recovery_bc": "#d95f02", "expert_outward": "#1b9e77"}
    names = {"original_bc": "Original policy", "recovery_bc": "Recovery-trained policy", "expert_outward": "Existing teacher"}
    steps = np.arange(summary["approach"] + 1, cfg["horizon"] + 1)
    for col, prefix in enumerate(worlds):
        for route in colors:
            group = [m for p, r, m in traces if p == prefix and r == route]
            for row, key, scale in ((0, "upperarm_ratio", 1), (1, "tracking_error", 1000)):
                values = np.array([[m[key] for m in trace] for trace in group]) * scale
                axes[row, col].plot(steps, values.mean(axis=0), color=colors[route], label=names[route])
                axes[row, col].fill_between(steps, values.min(axis=0), values.max(axis=0), color=colors[route], alpha=.13)
        axes[0, col].set_title(f"Prefix: {names[prefix]}")
        axes[0, col].axhline(.7, color="black", linestyle=":", linewidth=1)
        axes[0, col].set_ylim(0, 1.05)
        axes[1, col].axhline(20, color="black", linestyle=":", linewidth=1)
        axes[1, col].set_xlabel("Episode decision")
    axes[0, 0].set_ylabel("Upper-arm coverage")
    axes[1, 0].set_ylabel("Grasp tracking error (mm)")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Frozen continuations from matched states: tshirt_68 / body 14046\nLines: means; bands: min–max across two slots × two repeats (not confidence intervals)")
    fig.tight_layout()
    fig.savefig(args.root / "continuations.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in results.items() if k not in ("source_hashes", "episodes")}, indent=2))


if __name__ == "__main__":
    main()
