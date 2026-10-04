"""CPU-only paired action sensitivity on preserved observations; not a rollout success test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".claude/worktrees/residual-rl/python"))
from uipc_manip.obs import ObsSpec
from uipc_manip.wang_bridge import ReferencePolicy
from uipc_manip.flow_policy import FlowPolicy
from sim2real_perturbations import perturb_observation


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "output/uipc_manip/sim2real_audit_20261005")
    a = p.parse_args()
    protocol = json.loads((a.out / "protocol.json").read_text())
    old = ROOT / "output/uipc_manip/policy_eval_test_20260927/results.jsonl"
    rows = [json.loads(l) for l in old.read_text().splitlines()]
    samples = []
    for b in protocol["measured_bodies"]:
        row = next(r for r in rows if r.get("policy") == "r1" and r.get("garment") == "tshirt_26" and r["body"] == b)
        path = Path(row["log"]).with_suffix("") / row["path"]
        with np.load(path) as d:
            for t in (0, 20, 40, 80, 120, 160, 200):
                if t < len(d["obs"]):
                    samples.append((b, t, d["obs"][t].copy(), d["gripper_force"][t].copy(), row["max_translation_m"]))
    spec = ObsSpec((samples[0][2].size - 7) // 7)
    conditions = {"noise_3mm": dict(noise_m=.003), "dropout_30pct": dict(dropout=.3),
                  "voxel_origin_half_cell": dict(voxel_shift=[.03125] * 3)}
    results, started = [], time.monotonic()
    for name, checkpoint in protocol["policies"].items():
        cls = FlowPolicy if name == "flow" else ReferencePolicy
        policy = cls(checkpoint, device="cpu", yaw_deg=267.)
        for b, t, obs, force, step_m in samples:
            seed = b * 10000 + t
            f = force if name == "flow" else np.zeros(3)

            def action(o, voxel):
                torch.manual_seed(seed)
                if name == "flow":
                    policy.generator.manual_seed(seed)
                    policy.encoder_policy.voxel = voxel
                else:
                    policy.voxel = voxel
                pos, flags, valid, _ = spec.unpack_numpy(o)
                return policy.act(pos[valid], flags[valid], f)[:3]

            nominal = action(obs, .0625)
            for condition, kw in conditions.items():
                perturbed = perturb_observation(obs, spec, seed=20261005, body=b, frame=t, **kw)
                v = action(perturbed, 0. if "voxel_shift" in kw else .0625)
                results.append(dict(policy=name, body=b, frame=t, condition=condition,
                                    translation_change_mm=float(np.linalg.norm(v - nominal) * step_m * 1000),
                                    reversed_direction=bool(np.dot(v, nominal) < 0),
                                    nominal_translation_mm=float(np.linalg.norm(nominal) * step_m * 1000)))
        print(f"[sensitivity] {name}: {len(samples)} preserved states", flush=True)
    summary = {}
    for name in protocol["policies"]:
        summary[name] = {}
        for c in conditions:
            group = [r for r in results if r["policy"] == name and r["condition"] == c]
            v = np.array([r["translation_change_mm"] for r in group])
            summary[name][c] = dict(states=len(group), median_change_mm=float(np.median(v)),
                                    p90_change_mm=float(np.quantile(v, .9)),
                                    direction_reversals=sum(r["reversed_direction"] for r in group))
    record = dict(summary=summary, rows=results, wall_seconds=time.monotonic() - started,
                  checkpoint_sha256=protocol["checkpoint_sha256"], states_per_policy=len(samples),
                  limits="Open-loop on preserved r1 states. Nominal/perturbed flow share noise per state. Actions are raw translation proposals, not executed control or success. States within body are correlated.")
    (a.out / "observation_sensitivity.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
