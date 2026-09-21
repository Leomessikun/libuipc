"""How large an edit would an EXPO-style policy need on this task?

Real-Time EXPO-FT keeps a base policy frozen, learns a small *edit* of its proposed
action, and picks among candidates with a learned Q. That architecture is only
sample-efficient if the base is already nearly right: the edit policy has a bounded
output, and whatever lies outside that bound is unreachable no matter how good the
critic is.

This project has the one measurement that decides it. The recovery-transfer audit
holds eight continuations of a controller that *succeeds* (4/4 with valid grasp,
sustained coverage .979/.988) together with the observations it saw, and the frozen
deployable actors that *fail* from the same states (0/4). So for every decision of a
successful trajectory we can ask what the deployable base would have commanded, and
how far the edit would have had to move it.

The answer is reported as a fraction of the action box, because that is the unit an
edit bound is expressed in: an action coordinate lives in [-1, 1], so an edit of 2.0
in one coordinate can reach anything, and an edit of .1 can reach almost nothing.

Read-only: no simulation, no training, no GPU.

Usage:
    python scripts/measure_expo_edit_budget.py \
        --root output/uipc_manip/recovery_transfer_audit_20260920 \
        --out output/uipc_manip/expo_edit_budget_20260921/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from uipc_manip.physics_gradient_actor import load_agent
from uipc_manip.recovery_teacher import cap_commands

ACTIVE = [0, 1, 2, 4, 5]
"""The five commanded coordinates; X rotation is disabled by the continuation caps."""


def predict(agent, obs: np.ndarray) -> np.ndarray:
    return np.concatenate([agent.act(obs[i:i + 32], deterministic=True) for i in range(0, len(obs), 32)])


def describe(norms: np.ndarray, budgets) -> dict:
    return {
        "decisions": int(norms.size),
        "mean": float(norms.mean()),
        "median": float(np.median(norms)),
        "p90": float(np.percentile(norms, 90)),
        "max": float(norms.max()),
        "fraction_within": {f"{b:g}": float((norms <= b).mean()) for b in budgets},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/uipc_manip/recovery_transfer_audit_20260920"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    args = parser.parse_args()

    summary = json.loads((args.root / "summary.json").read_text())
    report = {"root": str(args.root), "cell": summary["cell"], "prefixes": {}}

    for prefix_dir in sorted(p for p in args.root.iterdir() if (p / "result.json").exists()):
        world = json.loads((prefix_dir / "result.json").read_text())
        env = world["env"]
        # The continuation caps the audit executed under: 8 mm and .05 rad, in normalised units.
        caps = (0.008 / env["max_translation"], 0.05 / env["max_rotation"])
        routes = {r["name"]: r for r in world["routes"]}
        expert = next(name for name, r in routes.items() if r["kind"] == "expert")
        actors = {name: load_agent(Path(r["checkpoint"]), world["env"]["point_budget"], 6, "cpu")
                  for name, r in routes.items() if r["kind"] == "checkpoint"}

        tapes = []
        for record in world["records"]:
            if record["route"] != expert:
                continue
            with np.load(prefix_dir / record["path"], allow_pickle=False) as tape:
                tapes.append((tape["obs"].copy(), tape["actions"].copy()))
        obs = np.concatenate([t[0] for t in tapes])
        teacher = np.concatenate([t[1] for t in tapes])
        moving = np.linalg.norm(teacher, axis=1) > 1e-6

        entry = {"successful_teacher_decisions": int(len(obs)), "moving": int(moving.sum()), "actors": {}}
        for name, agent in actors.items():
            # The base is scored as it would actually execute, under the same caps as the teacher.
            base = cap_commands(predict(agent, obs), *caps)
            edit = (teacher - base)[:, ACTIVE]
            norms = np.linalg.norm(edit, axis=1)
            entry["actors"][name] = {
                "all": describe(norms, args.budgets),
                "moving_teacher": describe(norms[moving], args.budgets),
                "per_coordinate_p90": [float(np.percentile(np.abs(edit[:, i]), 90)) for i in range(len(ACTIVE))],
                # The reachable set of a bounded edit is a box around the base action, so the
                # largest single coordinate is what the bound has to cover, not the norm.
                "max_abs_coordinate": float(np.abs(edit).max()),
                "base_action_norm_mean": float(np.linalg.norm(base[:, ACTIVE], axis=1).mean()),
                "teacher_action_norm_mean": float(np.linalg.norm(teacher[:, ACTIVE], axis=1).mean()),
            }
        report["prefixes"][prefix_dir.name] = entry

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    box = float(np.sqrt(len(ACTIVE)))
    print(f"action box: each coordinate in [-1, 1], so the largest possible edit norm is {box:.3f}\n")
    for prefix, entry in report["prefixes"].items():
        print(f"== {prefix}: {entry['successful_teacher_decisions']} decisions of a succeeding teacher")
        for name, stats in entry["actors"].items():
            s = stats["moving_teacher"]
            within = " ".join(f"<={b}:{f:.2f}" for b, f in s["fraction_within"].items())
            print(f"   base {name:<13} edit norm median {s['median']:.3f} p90 {s['p90']:.3f} max {s['max']:.3f} "
                  f"| max single coordinate {stats['max_abs_coordinate']:.3f}")
            print(f"   {'':<18} fraction of decisions within an edit bound of {within}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
