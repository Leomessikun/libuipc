"""Label the states a policy visits with the action that was measured to help there.

The prior-support audit found that the behavior flow is extrapolating wherever a
recovery matters, because its data is the scripted teacher's trajectories and the
states that need a recovery belong to the policy's. This collects the missing pairs
directly: the trained policy drives, and every decision is recorded with the action
the counterfactual branch study measured to be the best fixed recovery for that
garment, in the arm's own frame.

The executed action is the policy's, so the states are the policy's own; the recorded
action is the label. That is behavior cloning in the sense of DAgger, and it is what
the dataset is for. **The successors and rewards belong to the executed action, not to
the label, so this dataset can fit a behavior model and must not be used for value
learning**; the manifest says so in its own field.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip import decision_branches as db  # noqa: E402
from uipc_manip import train_sac  # noqa: E402
from uipc_manip.dressing_env import GenesisIPCDressingEnv  # noqa: E402
from uipc_manip.dressing_live import LiveCellFactory  # noqa: E402
from uipc_manip.expert_baseline import EpisodeTape  # noqa: E402
from uipc_manip.sac import SACAgent  # noqa: E402

# The branch study's winner per garment; everything else falls back to the arm's axis.
BEST_MACRO = {"tshirt_26": "lift", "tshirt_392": "forward", "tshirt_68": "forward",
              "hospital_gown": "forward", "tshirt_4": "forward"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bodies", default="14045,14046,14047,14048,14049")
    p.add_argument("--garments", default="hospital_gown,tshirt_26,tshirt_68,tshirt_4,tshirt_392")
    p.add_argument("--seed", type=int, default=11000)
    p.add_argument("--decisions", type=int, default=300)
    p.add_argument("--slots", type=int, default=25)
    p.add_argument("--step-m", type=float, default=0.008)
    p.add_argument("--shoulder-extension-m", type=float, default=0.05)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    bodies = [int(b) for b in args.bodies.split(",") if b.strip()]
    garments = [g for g in args.garments.split(",") if g.strip()]
    cells = tuple((g, b) for g in garments for b in bodies)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    base = train_sac.dressing_config(targs)
    started = time.perf_counter()
    args.out.mkdir(parents=True)
    (args.out / "episodes").mkdir()
    result = dict(checkpoint=str(args.checkpoint), policy="sac_states_labelled_with_measured_macro",
                  labels="branch-study best macro per garment", best_macro=BEST_MACRO,
                  executed="policy", label_matches_successor=False,
                  usable_for="behavior model only; successors and rewards belong to the executed action",
                  transition_schema="explicit_successors_v1", obs_dim=None, action_dim=6,
                  point_budget=base.point_budget, decisions=args.decisions, records=[], completed=False)

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "episode_metrics.json").write_text(json.dumps(result["records"], indent=2) + "\n")
        (args.out / "manifest.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2) + "\n")

    from uipc_manip.physics_gradient_actor import load_agent

    for start in range(0, len(cells), args.slots):
        chunk = cells[start:start + args.slots]
        cfg = replace(base, cells=chunk, decision_watchdog=False, contact_force_readout=False,
                      workspace=str(args.out / "assets"),
                      reward=replace(base.reward, upperarm_extension_m=args.shoulder_extension_m))
        env = GenesisIPCDressingEnv(cfg, num_envs=len(chunk), cell_factory=LiveCellFactory(cfg.live))
        result["obs_dim"] = env.obs_dim
        result["env"] = cfg.to_dict()
        try:
            agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
            obs = env.reset([args.seed + start + i for i in range(len(chunk))])
            tapes = [EpisodeTape(env.metric_keys, save_observations=True) for _ in chunk]
            macros = [next(m for m in db.MACROS if m["name"] == BEST_MACRO.get(g, "forward")) for g, _ in chunk]
            for _ in range(args.decisions):
                privileged = env.privileged()
                executed = agent.act(obs, deterministic=True)
                labels = []
                for i, cell in enumerate(env.cells):
                    directions = db.slot_directions(env._anchor[i], cell.finger, cell.elbow, cell.shoulder)
                    labels.append(db.macro_action(macros[i], 0, 1, directions, executed[i],
                                                  args.step_m, cfg.max_translation))
                nxt, reward, done, info = env.step(executed)
                for i, tape in enumerate(tapes):
                    tape.step(privileged[i], labels[i], obs[i], BEST_MACRO.get(chunk[i][0], "forward"),
                              float(reward[i]), info[i], next_obs=info[i].get("terminal_obs", nxt[i]))
                obs = nxt
                if any(row.get("sim_error") for row in info) or np.any(done):
                    break
            for i, tape in enumerate(tapes):
                index = len(result["records"])
                record = tape.record(index, info[i], chunk[i])
                record.update(index=index, label_macro=BEST_MACRO.get(chunk[i][0], "forward"), kept=True)
                path = Path("episodes") / f"episode_{index:05d}.npz"
                tape.save(args.out / path, record)
                record["path"] = str(path)
                result["records"].append(record)
                print(json.dumps({k: record[k] for k in ("index", "garment", "human", "label_macro",
                                                         "final_upperarm_ratio", "max_tracking_error")}), flush=True)
            save()
        finally:
            env.close()
            gc.collect()
    result["completed"] = True
    save()
    print(json.dumps(dict(episodes=len(result["records"]), decisions=args.decisions,
                          seconds=round(result["seconds"], 1))), flush=True)


if __name__ == "__main__":
    main()
