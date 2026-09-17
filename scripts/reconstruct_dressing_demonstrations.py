"""Replay existing IPC expert actions to record missing policy observations.

No new expert actions are generated. Replayed outcomes are measured afresh and
never inherit the old success label. The physical-quality admission rule is
explicitly separate from the historical early-turn/paper filter.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from uipc_manip import train_sac
from uipc_manip.dressing_env import GenesisIPCDressingEnv
from uipc_manip.dressing_live import LiveCellFactory
from uipc_manip.expert_baseline import EpisodeTape
from uipc_manip.sac import SACAgent


def admissible(record, coverage=.7, tracking=.02):
    return (not record.get("sim_error", False)
            and float(record.get("final_upperarm_ratio", -np.inf)) >= coverage
            and float(record.get("max_tracking_error", np.inf)) <= tracking)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=4100)
    p.add_argument("--include-failures", action="store_true",
                   help="Reconstruct every simulator-valid source episode for RL; BC admission remains separate.")
    p.add_argument("--upperarm-extension-m", type=float, default=0.0,
                   help="Explicit reward-version change; extend the shoulder ray by this many metres.")
    args = p.parse_args()
    if min(args.batch_size, args.repeats) < 1:
        p.error("Positive batch size and repeats required")
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads((args.source / "manifest.json").read_text())
    records = json.loads((args.source / "records.json").read_text())
    chosen = [r for r in records if not r.get("sim_error", False)] if args.include_failures else [r for r in records if admissible(r)]
    if not chosen:
        raise ValueError("No historical physically valid successful episodes to reconstruct")
    payload = SACAgent.read_checkpoint(args.reference)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    reference_cfg = train_sac.dressing_config(targs)
    targs._resume_env = source["env"]
    cfg = train_sac.restore_env_config(reference_cfg, targs)
    # Refuse a silent sim-to-sim experiment. These differences don't change physics.
    excluded = {"cells", "human", "garments", "workspace", "seed", "show_viewer", "logging_level",
                "decision_watchdog", "contact_force_readout"}
    a, b = cfg.to_dict(), reference_cfg.to_dict()
    mismatch = [k for k in a if k not in excluded and json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True)]
    if mismatch:
        raise ValueError(f"Source/reference environment mismatch: {mismatch}")
    args.out.mkdir(parents=True)
    (args.out / "episodes").mkdir()
    cfg = replace(cfg, workspace=str(args.out / "assets"), decision_watchdog=False, contact_force_readout=False,
                  reward=replace(cfg.reward, upperarm_extension_m=args.upperarm_extension_m))
    factory = LiveCellFactory(cfg.live)
    started = time.perf_counter()
    result = dict(source=str(args.source), reference=str(args.reference), env=cfg.to_dict(),
                  policy="replayed_existing_expert_actions", admission_rule="geometric_and_grasp_v1",
                  min_final_coverage=.7, max_tracking_m=.02, early_turn_is_separate=True,
                  source_episodes=len(chosen), repeats=args.repeats, obs_dim=reference_cfg.point_budget * 7 + 7,
                  point_budget=cfg.point_budget, action_dim=6, records=[], completed=False,
                  includes_failures=args.include_failures, transition_schema="explicit_successors_v1")

    def save():
        result["seconds"] = time.perf_counter() - started
        result["kept_episodes"] = sum(r["kept"] for r in result["records"])
        (args.out / "episode_metrics.json").write_text(json.dumps(result["records"], indent=2) + "\n")
        (args.out / "manifest.json").write_text(json.dumps({k:v for k,v in result.items() if k != "records"}, indent=2) + "\n")

    for start in range(0, len(chosen), args.batch_size):
        group = chosen[start:start + args.batch_size]
        cells = tuple((r["garment"], int(r["human"])) for r in group)
        arrays = [np.load(args.source / r["path"]) for r in group]
        if any(a["actions"].shape != (cfg.horizon, 6) for a in arrays):
            raise ValueError("Every saved action sequence must match the full source horizon")
        env = GenesisIPCDressingEnv(replace(cfg, cells=cells), num_envs=len(group), cell_factory=factory)
        result["obs_dim"] = env.obs_dim
        try:
            for rep in range(args.repeats):
                obs = env.reset([args.seed + start + i + rep * 100 for i in range(len(group))])
                tapes = [EpisodeTape(env.metric_keys, save_observations=True) for _ in group]
                for t in range(cfg.horizon):
                    priv = env.privileged()
                    actions = np.stack([a["actions"][t] for a in arrays])
                    nxt, reward, done, info = env.step(actions)
                    for i, tape in enumerate(tapes):
                        tape.step(priv[i], actions[i], obs[i], "saved_expert", float(reward[i]), info[i],
                                  next_obs=info[i].get("terminal_obs", nxt[i]))
                    obs = nxt
                    if any(row.get("sim_error") for row in info) or np.any(done):
                        break
                for i, tape in enumerate(tapes):
                    idx = len(result["records"])
                    record = tape.record(idx, info[i], cells[i])
                    record.update(index=idx, source_path=str(args.source / group[i]["path"]),
                                  source_sha256=hashlib.sha256((args.source / group[i]["path"]).read_bytes()).hexdigest(),
                                  source_record=group[i], repeat=rep,
                                  kept=bool(t == cfg.horizon - 1 and admissible(record)))
                    path = Path("episodes") / f"episode_{idx:05d}.npz"
                    tape.save(args.out / path, record)
                    record["path"] = str(path)
                    result["records"].append(record)
                    print(json.dumps({k:record[k] for k in ("index", "garment", "human", "kept", "final_upperarm_ratio", "max_tracking_error")}), flush=True)
                save()
        finally:
            env.close()
            for a in arrays:
                a.close()
            gc.collect()
    result["completed"] = True
    save()


if __name__ == "__main__":
    main()
