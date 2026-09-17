"""Test closed-loop recovery teachers at states visited by a dressing policy.

Reuse geometric route primitives, balanced native IPC branches, and EpisodeTape.
Selection uses full remaining episodes. A fresh world verifies the chosen route
against policy and scaled-policy controls before any observations become labels.
This is an experimental corrective-imitation teacher, not a new SAC update.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .parallel_trajopt import candidate_schedule


ROUTES = (
    dict(name="policy", kind="policy"),
    dict(name="policy_scaled", kind="scaled"),
    dict(name="expert_middle", kind="expert", stage=2),
    dict(name="expert_align", kind="expert", stage=3),
    dict(name="expert_hook", kind="expert", stage=5),
    dict(name="expert_outward", kind="expert", stage=2, outward_offset=.04),
)


def cap_commands(actions, translation_cap, rotation_cap):
    out = np.clip(np.asarray(actions, dtype=np.float32), -1., 1.).copy()
    out[..., 3] = 0
    for part, cap in ((slice(0, 3), translation_cap), (slice(3, 6), rotation_cap)):
        norm = np.linalg.norm(out[..., part], axis=-1, keepdims=True)
        out[..., part] *= np.minimum(1., cap / np.maximum(norm, 1e-12))
    return out


def route_summary(records):
    valid = all(r["whole_episode_grasp_valid"] and not r["sim_error"] for r in records)
    values = np.asarray([r["sustained_coverage"] for r in records])
    valid = bool(valid and np.isfinite(values).all())
    return dict(episodes=len(records), valid=valid,
                score=float(values.min()) if valid else None,
                final_coverage=[r["final_upperarm_ratio"] for r in records],
                sustained_coverage=values.tolist(),
                success=sum(r["whole_episode_grasp_valid"] and r["sustained_coverage"] >= .7
                            and not r["sim_error"] for r in records))


def verified_improvement(candidate, policy, scaled, *, margin=.05, threshold=.7):
    """Require sustained success and improvement over both controls on every slot."""
    groups = [candidate, policy, scaled]
    if not candidate or any(len(g) != len(candidate) for g in groups):
        raise ValueError("Verification requires nonempty matched samples")
    if any([r["slot"] for r in g] != [r["slot"] for r in candidate] for g in groups):
        raise ValueError("Verification slot identities differ")
    if any(r["sim_error"] for g in groups for r in g):
        return False
    for c, p, s in zip(*groups, strict=True):
        values = np.asarray([r["sustained_coverage"] for r in (c, p, s)])
        if (not np.isfinite(values).all() or not c["whole_episode_grasp_valid"]
                or values[0] < threshold or values[0] - max(values[1:]) < margin):
            return False
    return True


def branch_world(cfg, checkpoint, out, routes, approach, slots, seed):
    from . import physics_gradient_probe as probe
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_heuristic import HeuristicDressingPolicy
    from .dressing_live import LiveCellFactory
    from .expert_baseline import EpisodeTape
    from .physics_gradient_actor import load_agent

    out.mkdir(parents=True)
    (out / "episodes").mkdir()
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(replace(cfg, workspace=str(out / "assets")), num_envs=slots,
                               cell_factory=LiveCellFactory(cfg.live))
    agent = load_agent(checkpoint, env.spec.point_budget, env.action_dim, "cuda")
    if agent.cfg.history_length != 1:
        raise ValueError("Recovery pilot requires a single-frame actor")
    result = dict(completed=False, env=cfg.to_dict(), routes=routes, approach=approach,
                  seed=seed, records=[], physical_decisions=0, restore_errors=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        obs = env.reset([seed + i for i in range(slots)])
        prefix_tracking = np.zeros(slots)
        prefix_early = np.zeros(slots, dtype=bool)
        prefix = []
        for _ in range(approach):
            action = agent.act(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            result["physical_decisions"] += slots
            if np.any(done) or any(r.get("sim_error") for r in info):
                raise RuntimeError("Policy approach failed")
            prefix_tracking = np.maximum(prefix_tracking, [r["tracking_error"] for r in info])
            prefix_early |= [r["early_turn"] for r in info]
            prefix.append(action.copy())
        np.savez_compressed(out / "prefix.npz", actions=np.stack(prefix), obs=obs)
        snapshot = probe.take_snapshot(env, "recovery_teacher", approach)
        result["initial_coverage"] = [r["upperarm_ratio"] for r in info]
        result["prefix_tracking_m"] = prefix_tracking.tolist()
        print(json.dumps(dict(phase=out.name, initial_coverage=result["initial_coverage"])), flush=True)
        # Equal physical command limits for all route families, with a scaled
        # policy control to expose improvements explained only by larger motions.
        translation_cap = min(1., .008 / cfg.max_translation)
        rotation_cap = min(1., .05 / cfg.max_rotation)
        for _, assignment in candidate_schedule(len(routes), slots, 1):
            error = probe.restore(env, snapshot)
            result["restore_errors"].append(error)
            if error > 1e-5:
                raise RuntimeError(f"Restore error {error} m")
            obs = env.observation()
            teachers = []
            for route in routes:
                if route["kind"] == "expert":
                    teacher = HeuristicDressingPolicy(env, outward_offset=route.get("outward_offset", 0.))
                    teacher.stage[:] = route["stage"]
                else:
                    teacher = None
                teachers.append(teacher)
            tapes = [EpisodeTape(env.metric_keys, save_observations=True) for _ in range(slots)]
            rejections = np.zeros((slots, 2), dtype=int)
            command_distance = np.zeros(slots)
            accepted_distance = np.zeros(slots)
            for t in range(approach, cfg.horizon):
                policy_action = agent.act(obs, deterministic=True)
                actions = policy_action.copy()
                positions = env.positions()
                for idx in np.unique(assignment):
                    route = routes[idx]
                    mask = assignment == idx
                    if route["kind"] == "expert":
                        actions[mask] = teachers[idx].actions(positions)[mask]
                    elif route["kind"] == "scaled":
                        norm = np.linalg.norm(actions[mask, :3], axis=-1, keepdims=True)
                        actions[mask, :3] *= translation_cap / np.maximum(norm, 1e-12)
                actions = cap_commands(actions, translation_cap, rotation_cap)
                privileged = env.privileged()
                nxt, reward, done, info = env.step(actions, reset_on_done=False)
                result["physical_decisions"] += slots
                if any(r.get("sim_error") for r in info):
                    raise RuntimeError("Simulator failure invalidated branch comparison")
                if bool(np.any(done)) != (t == cfg.horizon - 1):
                    raise RuntimeError("Unexpected branch termination")
                for i, tape in enumerate(tapes):
                    tape.step(privileged[i], actions[i], obs[i], routes[assignment[i]]["name"], float(reward[i]), info[i])
                    rejections[i] += [info[i]["collision_rejected_substeps"], info[i]["tether_rejected_substeps"]]
                    command_distance[i] += info[i]["commanded_translation_m"]
                    accepted_distance[i] += info[i]["accepted_anchor_translation_m"]
                obs = nxt
            # Retaining the true terminal state prevents an auto-reset from
            # destroying the branch snapshot or recording next-episode labels.
            assert env._episode_step == cfg.horizon
            assert np.array_equal(obs, np.stack([r["terminal_obs"] for r in info]))
            for slot, tape in enumerate(tapes):
                record = tape.record(slot, info[slot], cfg.cells[slot])
                record["max_tracking_error"] = max(record["max_tracking_error"], float(prefix_tracking[slot]))
                record["early_turn"] |= bool(prefix_early[slot])
                record["paper_filter"] = record["success"] and not record["early_turn"]
                record.update(route=routes[assignment[slot]]["name"], kept=False,
                    branch_step=approach, whole_episode_grasp_valid=record["max_tracking_error"] <= .02,
                    sustained_coverage=min(r["upperarm_ratio"] for r in tape.step_metrics[-12:]),
                    collision_rejections=int(rejections[slot, 0]), tether_rejections=int(rejections[slot, 1]),
                    commanded_translation_m=float(command_distance[slot]), accepted_translation_m=float(accepted_distance[slot]))
                path = Path("episodes") / f"episode_{len(result['records']):05d}.npz"
                record["path"] = str(path)
                tape.save(out / path, record)
                result["records"].append(record)
                print(json.dumps({k:record[k] for k in ("route", "slot", "final_upperarm_ratio", "sustained_coverage", "whole_episode_grasp_valid")}), flush=True)
            save()
        result["summaries"] = {r["name"]: route_summary([e for e in result["records"] if e["route"] == r["name"]]) for r in routes}
        result["completed"] = True
        save()
        return result
    finally:
        env.close()
        del agent, env
        gc.collect()


def main():
    from . import train_sac
    from .sac import SACAgent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cell", default="tshirt_68:14046")
    parser.add_argument("--approach", type=int, default=120)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2197)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False)
    if args.slots < 1 or not 1 <= args.approach <= cfg.horizon - 12:
        parser.error("Positive slots and an approach leaving at least 12 decisions required")
    if cfg.augment_obs or not cfg.clip_rotation_to_yz:
        raise ValueError("Unaugmented observations and five-axis controller required")
    args.out.mkdir(parents=True)
    search = branch_world(cfg, args.checkpoint, args.out / "search", ROUTES, args.approach, args.slots, args.seed)
    eligible = [r for r in ROUTES[2:] if search["summaries"][r["name"]]["valid"]]
    selected = max(eligible, key=lambda r: search["summaries"][r["name"]]["score"]) if eligible else None
    result = dict(checkpoint=str(args.checkpoint), selected=selected, admitted=False,
                  search_seconds=search["seconds"], physical_decisions=search["physical_decisions"])
    if selected is not None and search["summaries"][selected["name"]]["score"] >= .7:
        verify = branch_world(cfg, args.checkpoint, args.out / "verification", (selected, ROUTES[1], ROUTES[0]),
                              args.approach, args.slots, args.seed + 1000)
        records = {r["name"]: sorted([e for e in verify["records"] if e["route"] == r["name"]], key=lambda e:e["slot"])
                   for r in (selected, ROUTES[1], ROUTES[0])}
        result["admitted"] = verified_improvement(records[selected["name"]], records["policy"], records["policy_scaled"])
        result["verification_seconds"] = verify["seconds"]
        result["physical_decisions"] += verify["physical_decisions"]
        for r in verify["records"]:
            r["kept"] = bool(result["admitted"] and r["route"] == selected["name"])
            if r["kept"]:
                path = args.out / "verification" / r["path"]
                with np.load(path) as data:
                    arrays = {k: data[k] for k in data.files}
                arrays["record"] = np.asarray(json.dumps(r))
                np.savez_compressed(path, **arrays)
        (args.out / "verification" / "episode_metrics.json").write_text(json.dumps(verify["records"], indent=2) + "\n")
        manifest = dict(env=cfg.to_dict(), obs_dim=cfg.point_budget * 7 + 7, point_budget=cfg.point_budget,
                        action_dim=6, admission_rule="verified_full_continuation_v1", completed=True,
                        source_checkpoint=str(args.checkpoint), branch_step=args.approach,
                        kept_episodes=sum(r["kept"] for r in verify["records"]))
        (args.out / "verification" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result["completed"] = True
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
