"""Compare action extraction from one frozen Q with independent IPC outcomes.

This is a finite-horizon intervention diagnostic, not a new trained policy or an
unbiased estimate of SAC's infinite-horizon soft Q. All selectors have the same
normalized action trust region. Physics is never differentiated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def project(action, base, radius, active):
    delta = (action - base) * active
    delta *= np.minimum(1.0, radius / np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1e-12))
    return np.clip(base + delta, -1, 1).astype(np.float32)


def make_bank(base, score, gradient, rng, *, radius=0.5, samples=8, iterations=8, clip_x=True):
    """Return base, best projected-gradient iterate, and random actions.

    score/gradient are batched callbacks on [states, action_dim]. The gradient
    arm retains its best iterate including the base; the sampled selector also
    includes the base. Disabled X rotation stays at the actor's original output
    so Q is queried with the same ignored command coordinate in every arm.
    """
    base = np.asarray(base, dtype=np.float32)
    active = np.ones(base.shape[-1], dtype=np.float32)
    if clip_x:
        active[3] = 0
    current, best = base.copy(), base.copy()
    best_q = np.asarray(score(base))
    for _ in range(iterations):
        g = np.asarray(gradient(current)) * active
        g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-12)
        current = project(current + radius / 4 * g, base, radius, active)
        q = np.asarray(score(current))
        improve = q > best_q
        best[improve], best_q[improve] = current[improve], q[improve]
    bank = [base, best]
    # Uniform directions on the active sphere, projected into the action box.
    for _ in range(samples):
        direction = rng.standard_normal(base.shape) * active
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)
        bank.append(project(base + radius * direction, base, radius, active))
    bank = np.stack(bank, axis=1)
    q = np.stack([score(bank[:, k]) for k in range(bank.shape[1])], axis=1)
    eligible = np.array([0, *range(2, bank.shape[1])])
    sampled = eligible[q[:, eligible].argmax(axis=1)]
    return bank, q, sampled


def heldout_choices(scores):
    """[states, candidates, repeats] -> indices using only other repeats."""
    scores = np.asarray(scores)
    if scores.ndim != 3 or scores.shape[-1] < 2 or not np.isfinite(scores).all():
        raise ValueError("Finite outcomes and at least two repeats are required")
    return np.stack([np.delete(scores, r, axis=2).mean(axis=2).argmax(axis=1)
                     for r in range(scores.shape[2])], axis=1)


def summarize(result):
    """Physical outcomes are evaluated separately from learned-Q increases."""
    states, records = result["states"], result["records"]
    n, k, repeats = len(states), len(states[0]["actions"]), result["arguments"]["repeats"]
    lookup = {(r["state"], r["candidate"], r["repeat"]): r for r in records}
    metrics = ["sustained_coverage", "valid_sustained_coverage", "discounted_env_return",
               "final_coverage", "prefix_and_branch_grasp_valid", "window_success"]
    output = dict(states=n, branches=len(records), seconds=result["seconds"],
                  physical_decisions=result["physical_decisions"], metrics={}, per_state=[])
    q = np.asarray([s["q"] for s in states])
    idx = np.arange(n)[:, None]
    rep = np.arange(repeats)[None, :]
    selections = {
        "policy": np.zeros((n, repeats), dtype=int),
        "q_gradient": np.ones((n, repeats), dtype=int),
    }
    if k > 2:
        selections["q_sampled"] = np.repeat(np.asarray([s["sampled_index"] for s in states])[:, None], repeats, axis=1)
        selections["fixed_random"] = np.full((n, repeats), 2, dtype=int)
    coverage = np.array([[[lookup[s["id"], c, r]["valid_sustained_coverage"]
                           for r in range(repeats)] for c in range(k)] for s in states])
    selections["forward_selected"] = heldout_choices(coverage)
    output["mean_predicted_q_gain"] = {
        name: float((q[np.arange(n), indices[:, 0]] - q[:, 0]).mean())
        for name, indices in selections.items() if name != "forward_selected"}
    for metric in metrics:
        values = np.array([[[lookup[s["id"], c, r][metric] for r in range(repeats)]
                            for c in range(k)] for s in states], dtype=float)
        baseline = values[:, 0]
        table = {}
        for name, choice in selections.items():
            selected = values[idx, choice, rep]
            gains = selected - baseline
            table[name] = dict(mean=float(selected.mean()), gain=float(gains.mean()),
                               positive_state_means=int((gains.mean(axis=1) > 0).sum()),
                               states_gain_over_001_all_repeats=int((gains.min(axis=1) > 0.01).sum()))
        output["metrics"][metric] = table
    for i, state in enumerate(states):
        row = dict(id=state["id"], initial=state["initial"], q_gain={}, valid_coverage={}, gains={})
        for name, choice in selections.items():
            selected = coverage[i, choice[i], np.arange(repeats)]
            row["valid_coverage"][name] = selected.tolist()
            row["gains"][name] = (selected - coverage[i, 0]).tolist()
            row["q_gain"][name] = (q[i, choice[i]] - q[i, 0]).tolist()
        output["per_state"].append(row)
    return output


def main():
    from . import physics_gradient_probe as probe, train_sac
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_live import LiveCellFactory
    from .physics_gradient_actor import load_agent
    from .sac import SACAgent
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", default=["tshirt_26:14046", "tshirt_392:14046"])
    parser.add_argument("--slots-per-cell", type=int, default=2)
    parser.add_argument("--steps", type=int, nargs="+", default=[60, 140])
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8, help="Zero validates only the policy and Q-gradient arms")
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9201)
    parser.add_argument("--max-seconds", type=float, default=3000)
    args = parser.parse_args()
    if min(args.slots_per_cell, args.iterations, *args.steps) < 1 or args.samples < 0 or args.repeats < 2 or args.window < 12:
        parser.error("Positive dimensions, at least two repeats and a 12-decision tail required")
    if not 0 < args.radius <= 1 or args.max_seconds <= 0:
        parser.error("radius must be in (0, 1] and max-seconds positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    cells = []
    for entry in args.cells:
        garment, body = entry.rsplit(":", 1)
        cells.extend([(garment, int(body))] * args.slots_per_cell)
    cfg = replace(train_sac.dressing_config(targs), cells=tuple(cells),
                  contact_force_readout=False, decision_watchdog=False, workspace=str(args.out / "assets"))
    if max(args.steps) + args.window > cfg.horizon or cfg.augment_obs:
        raise ValueError("Unaugmented observations and branches no later than the episode horizon required")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result = dict(completed=False, arguments=arguments, env=cfg.to_dict(), states=[], records=[],
                  physical_decisions=0, restore_errors_m=[],
                  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  branches_reach_episode_end=all(s + args.window == cfg.horizon for s in args.steps),
                  protocol="One modified action, then unchanged deterministic SAC; 12-decision minimum coverage; "
                           "prefix plus branch grasp validity. Forward selection excludes its evaluation repeat. "
                           "See branches_reach_episode_end for endpoint scope. Not independent task samples "
                           "or infinite-horizon soft-Q calibration.")
    arrays = {}

    def save():
        result["seconds"] = time.perf_counter() - started
        tmp = args.out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
        tmp.replace(args.out / "result.json")
        np.savez_compressed(args.out / "observations.npz", **arrays)

    save()
    env = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        env = GenesisIPCDressingEnv(cfg, num_envs=len(cells), cell_factory=LiveCellFactory(cfg.live))
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        if agent.cfg.history_length != 1 or agent.cfg.critic_input != "points" or agent.cfg.algo != "sac":
            raise ValueError("This audit requires the scalar single-frame point critic")
        for parameter in agent.critic.parameters():
            parameter.requires_grad_(False)
        rng = np.random.default_rng(args.seed)
        seeds = [args.seed + i for i in range(len(cells))]
        obs = env.reset(seeds)
        step = 0
        prefix_valid = np.ones(len(cells), dtype=bool)

        def advance(action, *, allow_terminal=False):
            if time.perf_counter() - started > args.max_seconds:
                raise TimeoutError("Audit wall-clock budget exhausted")
            next_obs, reward, done, rows = env.step(action, reset_on_done=False)
            result["physical_decisions"] += len(cells)
            if (np.any(done) and not allow_terminal) or any(r.get("sim_error") for r in rows):
                raise RuntimeError("Unexpected termination inside the bounded audit")
            if allow_terminal and not np.all(done):
                raise RuntimeError("Expected the common time-limit endpoint")
            return next_obs, reward, rows

        for target in sorted(set(args.steps)):
            while step < target:
                obs, _, rows = advance(agent.act(obs, deterministic=True))
                prefix_valid &= np.asarray([r["grasp_valid"] for r in rows])
                step += 1
            snapshot = probe.take_snapshot(env, f"selector_{target}", target)
            base = agent.act(obs, deterministic=True)
            frozen_obs = agent._unpack(torch.as_tensor(obs, device=agent.device))

            def q_scores(action):
                with torch.no_grad():
                    q1, q2 = agent.critic(frozen_obs, torch.as_tensor(action, device=agent.device))
                    return torch.minimum(q1, q2).cpu().numpy().reshape(-1)

            def q_gradient(action):
                action = torch.as_tensor(action, device=agent.device).requires_grad_(True)
                q1, q2 = agent.critic(frozen_obs, action)
                return torch.autograd.grad(torch.minimum(q1, q2).sum(), action)[0].cpu().numpy()

            query_started = time.perf_counter()
            bank, q, sampled = make_bank(base, q_scores, q_gradient, rng, radius=args.radius,
                                        samples=args.samples, iterations=args.iterations, clip_x=cfg.clip_rotation_to_yz)
            ids = [f"{g}__{b}__seed{seed}__step{target}" for (g, b), seed in zip(cells, seeds, strict=True)]
            for i, sid in enumerate(ids):
                result["states"].append(dict(id=sid, slot=i, step=target, seed=seeds[i],
                    initial={k: rows[i][k] for k in ("upperarm_ratio", "forearm_ratio", "threaded", "tracking_error")},
                    prefix_grasp_valid=bool(prefix_valid[i]), actions=bank[i].tolist(), q=q[i].tolist(),
                    sampled_index=int(sampled[i]), query_batch_seconds=time.perf_counter() - query_started))
                arrays[f"{sid}__observation"] = obs[i].copy()
            save()
            for repeat in range(args.repeats):
                for candidate in rng.permutation(bank.shape[1]):
                    error = probe.restore(env, snapshot)
                    result["restore_errors_m"].append(error)
                    if error > 1e-5:
                        raise RuntimeError(f"Position restore error {error} m")
                    traces = [[] for _ in cells]
                    commands = []
                    for t in range(args.window):
                        action = bank[:, candidate] if t == 0 else agent.act(branch, deterministic=True)
                        commands.append(action.copy())
                        branch, reward, info = advance(action, allow_terminal=(target + t + 1 == cfg.horizon))
                        for i, row in enumerate(info):
                            trace = {k: row[k] for k in ("upperarm_ratio", "forearm_ratio", "threaded", "grasp_valid",
                                     "tracking_error", "collision_rejected_substeps", "tether_rejected_substeps")}
                            trace["reward"] = float(reward[i])
                            traces[i].append(trace)
                    for i, sid in enumerate(ids):
                        trace = traces[i]
                        valid = bool(prefix_valid[i] and all(t["grasp_valid"] for t in trace))
                        coverage = min(t["upperarm_ratio"] for t in trace[-12:])
                        result["records"].append(dict(state=sid, candidate=int(candidate), repeat=repeat,
                            sustained_coverage=coverage, valid_sustained_coverage=coverage if valid else 0.0,
                            final_coverage=trace[-1]["upperarm_ratio"], prefix_and_branch_grasp_valid=valid,
                            window_success=bool(valid and coverage >= cfg.reward.success_upperarm_ratio),
                            discounted_env_return=float(sum(agent.cfg.discount**t * r["reward"] for t, r in enumerate(trace))),
                            trace=trace))
                        arrays[f"{sid}__c{candidate}__r{repeat}__terminal_observation"] = branch[i].copy()
                        arrays[f"{sid}__c{candidate}__r{repeat}__commands"] = np.stack(commands)[:, i]
                    print(json.dumps(dict(step=target, candidate=int(candidate), repeat=repeat,
                                          decisions=result["physical_decisions"], seconds=round(time.perf_counter() - started, 1))), flush=True)
                    save()
            error = probe.restore(env, snapshot)
            if error > 1e-5:
                raise RuntimeError(f"Approach restore error {error} m")
            obs = env.observation()
        result["completed"] = True
        save()
        (args.out / "summary.json").write_text(json.dumps(summarize(result), indent=2) + "\n")
    except BaseException as error:
        result["error"] = repr(error)
        save()
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
