"""Matched dressing episodes with controller-rejection and grasp diagnostics.

The reference checkpoint supplies the environment contract. Policies see their
normal observations; no physics gradients or privileged inputs are added.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from uipc_manip import train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.dressing_live import LiveCellFactory
from uipc_manip.physics_gradient_actor import load_agent
from uipc_manip.sac import SACAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", nargs="+", required=True, help="NAME=CHECKPOINT entries")
    parser.add_argument("--cells", nargs="+", default=["tshirt_26:14046", "tshirt_26:14049"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    policies = [entry.split("=", 1) for entry in args.policy]
    if any(len(entry) != 2 or not Path(entry[1]).is_file() for entry in policies):
        parser.error("Every policy must name an existing checkpoint")
    cells = [(g, int(b)) for g, b in (entry.rsplit(":", 1) for entry in args.cells)]
    payload = SACAgent.read_checkpoint(args.reference)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    cfg = replace(train_sac.dressing_config(targs), cells=tuple(cells), decision_watchdog=False,
                  workspace=str(args.out.parent / (args.out.stem + "_assets")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=len(cells), cell_factory=LiveCellFactory(cfg.live))
    result = {"reference": str(args.reference), "cells": cells, "rounds": args.rounds,
              "build_seconds": time.perf_counter() - started, "env": cfg.to_dict(), "evaluations": []}
    original_step = env.step
    traces = []

    def step(actions):
        before = time.perf_counter()
        output = original_step(actions)
        elapsed = time.perf_counter() - before
        for slot, info in enumerate(output[3]):
            row = {k: info[k] for k in ("episode_step", "upperarm_ratio", "forearm_ratio", "early_turn",
                    "tracking_error", "grasp_valid", "valid_grasp_success", "collision_rejected_substeps",
                    "tether_rejected_substeps", "commanded_translation_m", "accepted_anchor_translation_m",
                    "sim_error") if k in info}
            traces.append({"slot": slot, "decision_seconds": elapsed, "action": actions[slot].tolist(), **row})
        return output

    env.step = step
    try:
        for rep in range(args.rounds):
            # Reverse order on alternating rounds to reduce order confounding.
            for name, checkpoint in policies[::1 if rep % 2 == 0 else -1]:
                agent = load_agent(Path(checkpoint), env.spec.point_budget, env.action_dim, "cuda")
                if agent.cfg.history_length != 1:
                    raise ValueError("This diagnostic currently supports single-frame actors only")
                traces.clear()
                targs._eval_round = rep + 1
                before = time.perf_counter()
                evaluation = train_sac.evaluate(env, agent.act, env.spec, targs, len(cells),
                                                slot_cells=cells, heldout_slots=range(len(cells)))
                row = {"name": name, "checkpoint": checkpoint, "round": rep,
                       "seed_base": targs.seed * 1000 + 97 * (rep + 1),
                       "seconds": time.perf_counter() - before, "evaluation": evaluation,
                       "trace": list(traces)}
                result["evaluations"].append(row)
                result["total_seconds"] = time.perf_counter() - started
                args.out.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({"name": name, "round": rep, "seconds": row["seconds"],
                                  "coverage": evaluation["mean_final_upperarm_ratio"],
                                  "success": evaluation["success_rate"],
                                  "valid_grasp_success": evaluation["valid_grasp_success_rate"],
                                  "sim_errors": evaluation["sim_errors"]}), flush=True)
                del agent
    finally:
        env.close()


if __name__ == "__main__":
    main()
