"""Bounded actor-action audit on restored states of an existing dressing policy.

No policy optimization. Compare six-frame IPC surrogates with finite differences
through actual env.step, then test finite corrections with closed-loop continuation.
The finite differences are diagnostic oracles, not a proposed cheap training method.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from uipc_manip import physics_gradient_actor as actor
from uipc_manip import physics_gradient_adjoint as adjoint
from uipc_manip import physics_gradient_probe as probe
from uipc_manip import physics_gradient_trajopt as trajopt
from uipc_manip.physics_gradient_finetune import bounded_proposal


def task_improvement(references, candidates, coverage_margin=0.01):
    """Both repeated comparisons must improve coverage beyond baseline variability.

    Axis and return are reported separately, never substituted for dressing coverage.
    Grasp must stay valid throughout both continuations. These are repeated simulator
    executions from the same state, not independent task samples.
    """
    if len(references) != 2 or len(candidates) != 2:
        raise ValueError("Two references and two candidate executions are required")
    margin = max(coverage_margin, 2 * abs(references[0]["coverage"] - references[1]["coverage"]))
    deltas = [c["coverage"] - r["coverage"] for r, c in zip(references, candidates, strict=True)]
    accepted = all(np.isfinite(d) and d > margin and c["all_grasps_valid"] and r["all_grasps_valid"]
                   for d, r, c in zip(deltas, references, candidates, strict=True))
    return dict(accepted=bool(accepted), coverage_margin=float(margin), coverage_deltas=deltas)


def finite_action_differences(action, epsilon, evaluate, clip_x=True):
    """Finite-scale slopes of all scalar evaluator outputs, respecting the action box.

    Denominators use the actual clipped interval; boundary results are secants rather
    than an incorrectly normalized central derivative. Each action uses a full restore.
    """
    gradients, outcomes = {}, []
    for k in range(6):
        if clip_x and k == 3:
            continue
        plus, minus = np.clip(action, -1, 1), np.clip(action, -1, 1)
        plus[k], minus[k] = min(1, plus[k] + epsilon), max(-1, minus[k] - epsilon)
        span = plus[k] - minus[k]
        p, m = evaluate(plus), evaluate(minus)
        for name in p:
            gradients.setdefault(name, np.zeros(6))[k] = (p[name] - m[name]) / span
        outcomes.append(dict(axis=k, span=float(span), plus=p, minus=m))
    return gradients, outcomes


def main(argv=None):
    from uipc_manip import train_sac
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.sac import SACAgent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", default=["tshirt_26:14049", "tshirt_26:14046"])
    parser.add_argument("--steps", nargs="+", type=int, default=[180, 240])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument("--fd-repeats", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1097)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.horizon < 1 or args.fd_repeats < 1 or min(args.steps) < 1 or min(args.epsilons) <= 0 or not 0 < args.radius <= 1:
        parser.error("Positive steps, horizon, epsilons and radius in (0, 1] required")
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    cfg = train_sac.dressing_config(targs)
    if cfg.augment_obs:
        raise ValueError("This audit requires the saved policy's unaugmented evaluation contract")
    if max(args.steps) + args.horizon >= cfg.horizon:
        raise ValueError("Snapshots plus continuation must precede automatic episode reset")
    args.out.mkdir(parents=True)
    report = dict(checkpoint=str(args.checkpoint), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  scope="development action audit, no training; two repeated executions per state",
                  limitations=["six-frame chain uses projected last-iterate Hessians and inertia coupling",
                               "cloth-arm friction history and changing discrete observation/controller branches are not analytically differentiated",
                               "V=min Q(o_next,mu(o_next)) is a continuation surrogate, not the full SAC objective"],
                  cells=[], states=[], transitions=0)
    started = time.perf_counter()

    def save():
        report["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(report, indent=2, default=lambda x: x.tolist()) + "\n")

    rng = np.random.default_rng(args.seed)
    for cell_index, entry in enumerate(args.cells):
        garment, body = entry.rsplit(":", 1)
        cell_cfg = replace(cfg, cells=((garment, int(body)),), decision_watchdog=False,
                           workspace=str(args.out / f"assets_{cell_index}"))
        env = GenesisIPCDressingEnv(cell_cfg, num_envs=1, cell_factory=LiveCellFactory(cell_cfg.live))
        try:
            agent = actor.load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
            if agent.cfg.history_length != 1:
                raise ValueError("Only single-frame policies are supported by this audit")
            feature, layout = adjoint.adjoint_feature(env), adjoint.cloth_layout(env)
            if feature is None:
                raise RuntimeError("Native adjoint feature missing")
            capture = actor.ObservationCapture(env)
            obs = env.reset([args.seed])
            snaps, trace = [], []
            # Replay the saved policy to the selected decisions; no new teacher corpus.
            for step in range(1, max(args.steps) + 1):
                action = agent.act(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)
                report["transitions"] += 1
                if infos[0].get("sim_error") or done[0]:
                    raise RuntimeError(f"Approach terminated: {infos[0]}")
                trace.append(dict(step=step, action=action[0].tolist(), coverage=infos[0]["upperarm_ratio"],
                                  forearm=infos[0]["forearm_ratio"], grasp_valid=infos[0]["grasp_valid"]))
                if step in args.steps:
                    snap = probe.take_snapshot(env, f"policy_{step}", step)
                    snap["policy_observation"] = obs[0].copy()
                    snap["grasp_valid"] = infos[0]["grasp_valid"]
                    snaps.append(snap)
            report["cells"].append(dict(cell=entry, config=cell_cfg.to_dict(), approach=trace))
            save()
            for snap in snaps:
                obs = snap["policy_observation"]
                action = agent.act(obs[None], deterministic=True)[0].astype(np.float64)
                state_started = time.perf_counter()
                pg = actor.physics_gradient(env, snap, action, feature, layout, agent, capture, task_direction=True)
                report["transitions"] += 1
                record = dict(cell=entry, step=snap["episode_step"], initial=snap["measure"], initial_grasp_valid=snap["grasp_valid"],
                              action=action, gradients={k: pg[k] for k in ("chain", "chain_cloth_only", "direct_tool", "last_frame", "task_chain")},
                              device_residual=pg["device_residual"], finite_differences=[], candidates={})
                report["states"].append(record)

                def measure_action(a):
                    roll = trajopt.rollout(env, snap, a[None, :])
                    report["transitions"] += 1
                    if roll["restore_error_m"] > 1e-5:
                        raise RuntimeError("Snapshot restore error exceeds 10 micrometres")
                    obj = trajopt.objective(env, roll["positions"], layout)
                    return dict(value=actor.value(agent, env.observation([roll["positions"]])[0]),
                                frozen_value=actor.frozen_value(agent, capture, pg["cap"], roll["positions"], env._anchor[0]),
                                fixed_tool_value=actor.frozen_value(agent, capture, pg["cap"], roll["positions"]),
                                task=obj["value"], coverage=roll["measure"]["upperarm_ratio"],
                                axis_m=obj["axis_m"], grasp_valid=float(roll["decisions"][0]["grasp_valid"]))

                for eps in args.epsilons:
                    draws, outcomes = [], []
                    for _ in range(args.fd_repeats):
                        draw, sides = finite_action_differences(action, eps, measure_action, cfg.clip_rotation_to_yz)
                        draws.append(draw)
                        outcomes.append(sides)
                    gradients = {k: np.mean([d[k] for d in draws], axis=0) for k in draws[0]}
                    record["finite_differences"].append(dict(epsilon=eps, gradients=gradients, outcomes=outcomes, draws=draws,
                        repeat_cosine={k: probe.cosine(draws[0][k], draws[1][k]) for k in draws[0]} if len(draws) > 1 else {},
                        chain_frozen_cosine=probe.cosine(pg["chain"], gradients["frozen_value"]),
                        old_last_frozen_cosine=probe.cosine(pg["last_frame"], gradients["frozen_value"]),
                        chain_full_cosine=probe.cosine(pg["chain"], gradients["value"]),
                        task_cosine=probe.cosine(pg["task_chain"], gradients["task"])))
                    save()
                directions = dict(ipc_value=pg["chain"], ipc_task=pg["task_chain"],
                                  fd_value=gradients["value"], fd_task=gradients["task"],
                                  sac_gradient=actor.sac_action_gradient(agent, obs, action)["dQ_da"], random=rng.normal(size=6))
                actions = {name: bounded_proposal(action, g, args.radius, args.radius, cfg.clip_rotation_to_yz)
                           for name, g in directions.items()}
                actions["sac"] = action
                runs = {name: [] for name in actions}

                def continuation(a):
                    restore_error = probe.restore(env, snap)
                    if restore_error > 1e-5:
                        raise RuntimeError("Snapshot restore error exceeds 10 micrometres")
                    trace = []
                    obs_next = obs[None]
                    for t in range(args.horizon):
                        command = a if t == 0 else agent.act(obs_next, deterministic=True)[0]
                        obs_next, reward, done, infos = env.step(command[None])
                        report["transitions"] += 1
                        info = infos[0]
                        if info.get("sim_error") or done[0]:
                            raise RuntimeError(f"Continuation terminated: {info}")
                        trace.append(dict(coverage=info["upperarm_ratio"], grasp_valid=info["grasp_valid"],
                                          reward=float(reward[0]), tracking_error=info["tracking_error"],
                                          accepted_m=info["accepted_anchor_translation_m"],
                                          collision_rejections=info["collision_rejected_substeps"],
                                          tether_rejections=info["tether_rejected_substeps"]))
                    return dict(coverage=trace[-1]["coverage"], axis_m=probe.upperarm_axis(env, env.positions()),
                                all_grasps_valid=all(r["grasp_valid"] for r in trace),
                                return_value=sum(agent.cfg.discount**i * r["reward"] for i, r in enumerate(trace)),
                                restore_error_m=restore_error, trace=trace)

                order = list(actions)
                rng.shuffle(order)
                for rep in range(2):
                    for name in order[::1 if rep == 0 else -1]:
                        runs[name].append(continuation(actions[name]))
                record["references"] = runs["sac"]
                for name in directions:
                    record["candidates"][name] = dict(action=actions[name], runs=runs[name],
                                                       **task_improvement(runs["sac"], runs[name]))
                record["seconds"] = time.perf_counter() - state_started
                save()
                print(json.dumps(dict(cell=entry, step=snap["episode_step"],
                                      cosine=[r["chain_full_cosine"] for r in record["finite_differences"]],
                                      accepted={k: v["accepted"] for k, v in record["candidates"].items()},
                                      seconds=record["seconds"])), flush=True)
            del agent
        finally:
            env.close()
    report["completed"] = True
    save()


if __name__ == "__main__":
    main()
