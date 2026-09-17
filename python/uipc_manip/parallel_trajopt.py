"""Parallel shooting through IPC with a CUDA cross-entropy trajectory optimizer.

Contact sequences emerge from the native forward solves. This is derivative-free
optimal control, not a new SAC update or a complete contact adjoint. Candidates
share a collision-isolated IPC world; time steps remain sequential. CUDA samples,
interpolates, projects and ranks plans. The existing environment's controller,
reward, observations and snapshot transfers still involve the CPU.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def project_actions(actions: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Action box and per-decision translation/rotation norm caps of the reference.

    Rotation about x is inactive in the supported dressing controller. Comparing
    routes cannot silently give one candidate a larger command magnitude.
    """
    out = actions.clamp(-1, 1).clone()
    out[..., 3] = 0
    ref = reference.clamp(-1, 1).clone()
    ref[..., 3] = 0
    for part in (slice(0, 3), slice(3, 6)):
        cap = torch.linalg.vector_norm(ref[..., part], dim=-1, keepdim=True)
        norm = torch.linalg.vector_norm(out[..., part], dim=-1, keepdim=True)
        out[..., part] *= torch.clamp(cap / norm.clamp_min(1e-12), max=1)
    return out


def expand_knots(knots: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    offsets = F.interpolate(knots.transpose(1, 2), size=len(reference), mode="linear", align_corners=True)
    return project_actions(reference[None] + offsets.transpose(1, 2), reference)


def candidate_schedule(count: int, slots: int, repeats: int):
    """Each candidate visits each slot exactly once per repeat, without padding."""
    if min(count, slots, repeats) < 1:
        raise ValueError("Positive candidate, slot and repeat counts required")
    for rep in range(repeats):
        for rotation in range(count):
            yield rep, (np.arange(slots) + rotation) % count


def robust_scores(coverage: torch.Tensor, axis: torch.Tensor, valid: torch.Tensor,
                  risk: float = 1.0) -> torch.Tensor:
    """Worst last-four-step coverage + 0.1 normalized final opening-axis progress.

    Inputs are [candidate, replicate, decision]. A single invalid/nonfinite
    trajectory rejects that candidate. Risk penalizes between-replicate spread.
    This search surrogate is separate from full-episode success evaluation.
    """
    window = min(4, coverage.shape[-1])
    value = coverage[..., -window:].amin(dim=-1) + 0.1 * axis[..., -1]
    finite = torch.isfinite(coverage).all(dim=-1) & torch.isfinite(axis).all(dim=-1)
    safe = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
    score = safe.mean(dim=-1) - risk * safe.std(dim=-1, unbiased=False)
    return score.masked_fill(~(valid.all(dim=-1) & finite).all(dim=-1), -torch.inf)


class TrajectoryCEM:
    def __init__(self, reference: torch.Tensor, population: int, knots: int,
                 seed: int, sigma: float = 0.35):
        if reference.ndim != 2 or reference.shape[-1] != 6 or not torch.isfinite(reference).all():
            raise ValueError("A finite [horizon,6] reference is required")
        if population < 4 or not 2 <= knots <= len(reference) or sigma <= 0:
            raise ValueError("population >= 4, 2 <= knots <= horizon and positive sigma required")
        self.reference = project_actions(reference, reference)
        self.population = population
        self.generator = torch.Generator(device=reference.device).manual_seed(seed)
        self.mean = reference.new_zeros((knots, 6))
        self.std = torch.full_like(self.mean, sigma)
        self.best = self.mean.clone()

    def ask(self):
        noise = torch.randn((self.population, *self.mean.shape), device=self.mean.device,
                            dtype=self.mean.dtype, generator=self.generator)
        knots = self.mean + self.std * noise
        knots[..., 3] = 0
        knots[0].zero_()  # Exact original plan remains a control in every generation.
        knots[1] = self.best
        return knots, expand_knots(knots, self.reference)

    def tell(self, knots: torch.Tensor, scores: torch.Tensor):
        finite = torch.nonzero(torch.isfinite(scores), as_tuple=False).flatten()
        if finite.numel() == 0:
            return False
        order = finite[scores[finite].argsort(descending=True)]
        elite = knots[order[:max(2, self.population // 4)]]
        self.best = knots[order[0]].clone()
        self.mean = 0.25 * self.mean + 0.75 * elite.mean(dim=0)
        self.std = (0.25 * self.std + 0.75 * elite.std(dim=0, unbiased=False)).clamp_min(0.08)
        return True


class IPCBatchEvaluator:
    """Restore the complete batch and balance candidate assignments across slots."""
    def __init__(self, env, snapshot, device: torch.device):
        self.env, self.snapshot, self.device = env, snapshot, device
        self.decisions = 0
        self.records = []

    def evaluate(self, plans: torch.Tensor, repeats: int = 1, tail_agent=None, tail_steps: int = 0,
                 closed_loop_candidates: tuple[int, ...] = ()):
        from . import physics_gradient_probe as probe

        if plans.ndim != 3 or plans.shape[-1] != 6 or not torch.isfinite(plans).all():
            raise ValueError("Finite [candidate,horizon,6] plans required")
        env, n = self.env, self.env.num_envs
        count, horizon = plans.shape[:2]
        if tail_steps < 0 or self.snapshot["episode_step"] + horizon + tail_steps >= env.cfg.horizon:
            raise ValueError("Rollout must stop before environment auto-reset")
        if (tail_steps or closed_loop_candidates) and tail_agent is None:
            raise ValueError("A policy is required for the closed-loop tail")
        if any(idx < 0 or idx >= count for idx in closed_loop_candidates):
            raise ValueError("Closed-loop candidate index out of range")
        started = time.perf_counter()
        # One host transfer per whole bank, not one Torch synchronization per action.
        bank = plans.detach().cpu().numpy()
        shape = (count, n * repeats, horizon + tail_steps)
        coverage, axis, tracking, reward = [np.full(shape, np.nan, dtype=np.float64) for _ in range(4)]
        valid = np.zeros(shape, dtype=bool)
        rejection = np.zeros((*shape, 2), dtype=np.int32)
        restore_errors, restore_s, step_s = [], 0.0, 0.0
        axes = np.stack([c.shoulder - c.elbow for c in env.cells])
        length2 = np.square(axes).sum(axis=1)
        for rep, indices in candidate_schedule(count, n, repeats):
            before = time.perf_counter()
            error = probe.restore(env, self.snapshot)
            restore_s += time.perf_counter() - before
            restore_errors.append(error)
            if error > 1e-5:
                raise RuntimeError(f"Position restore mismatch: {error} m")
            chosen = indices
            mask = np.isin(indices, closed_loop_candidates)
            obs = env.observation() if mask.any() else None
            for t in range(horizon + tail_steps):
                action = bank[chosen, t] if t < horizon else tail_agent.act(obs, deterministic=True)
                if t < horizon and mask.any():
                    action = action.copy()
                    action[mask] = tail_agent.act(obs[mask], deterministic=True)
                before = time.perf_counter()
                obs, rew, done, info = env.step(action)
                step_s += time.perf_counter() - before
                self.decisions += n
                if np.any(done) or any(row.get("sim_error") for row in info):
                    raise RuntimeError(f"Trajectory terminated unexpectedly: {info}")
                positions = env.positions()
                for slot, idx in enumerate(indices):
                    row, sample, cell = info[slot], rep * n + slot, env.cells[slot]
                    coverage[idx, sample, t] = row["upperarm_ratio"]
                    center = positions[slot][cell.opening_idx].mean(axis=0)
                    axis[idx, sample, t] = np.dot(center - cell.elbow, axes[slot]) / length2[slot]
                    tracking[idx, sample, t] = row["tracking_error"]
                    valid[idx, sample, t] = row["grasp_valid"]
                    reward[idx, sample, t] = rew[slot]
                    rejection[idx, sample, t] = (row["collision_rejected_substeps"], row["tether_rejected_substeps"])
        cov_t = torch.as_tensor(coverage, device=self.device)
        axis_t = torch.as_tensor(axis, device=self.device)
        valid_t = torch.as_tensor(valid, device=self.device)
        scores = robust_scores(cov_t, axis_t, valid_t)
        # Transfer completed summaries once; native calls already synchronize physics.
        record = dict(step_seconds=step_s, restore_seconds=restore_s,
                      candidates=count, slots=n, repeats_per_slot=repeats,
                      closed_loop_candidates=list(closed_loop_candidates),
                      physical_decisions=n * sum(1 for _ in candidate_schedule(count, n, repeats)) * (horizon + tail_steps),
                      restore_error_max_m=max(restore_errors), scores=[s if np.isfinite(s) else None for s in scores.cpu().tolist()],
                      coverage=coverage.tolist(), axis=axis.tolist(), tracking_m=tracking.tolist(),
                      valid=valid.tolist(), rewards=reward.tolist(), rejections=rejection.tolist())
        record["seconds"] = time.perf_counter() - started
        self.records.append(record)
        return scores, record


def main():
    from . import physics_gradient_probe as probe, train_sac
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_live import LiveCellFactory
    from .physics_gradient_actor import load_agent
    from .sac import SACAgent

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--population", type=int, default=8)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--knots", type=int, default=4)
    p.add_argument("--approach", type=int, default=60)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--tail-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=1097)
    banks = p.add_mutually_exclusive_group()
    banks.add_argument("--benchmark-bank", type=Path, help="Only replay a saved bank/prefix; no optimization or validation")
    banks.add_argument("--validate-bank", type=Path, help="Validate saved reference/CEM/scaled/random plans plus closed-loop SAC")
    args = p.parse_args()
    if min(args.batch_size, args.iterations, args.approach, args.repeats) < 1:
        p.error("Positive batch size, iterations, approach and repeats required")
    if not torch.cuda.is_available():
        raise RuntimeError("This native runner requires CUDA")
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.batch_size,
                  workspace=str(args.out / "assets"), contact_force_readout=False, decision_watchdog=False)
    if cfg.augment_obs or not cfg.clip_rotation_to_yz:
        raise ValueError("This experiment requires unaugmented observations and the five-axis controller")
    if args.approach + args.horizon + args.tail_steps >= cfg.horizon or not 2 <= args.knots <= args.horizon:
        p.error("Choose valid knots and a total horizon before automatic reset")
    args.out.mkdir(parents=True)
    start = time.perf_counter()
    result = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  config=cfg.to_dict(), completed=False, evaluations=[])

    def save():
        result["total_seconds"] = time.perf_counter() - start
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    env = GenesisIPCDressingEnv(cfg, num_envs=args.batch_size, cell_factory=LiveCellFactory(cfg.live))
    try:
        device = torch.device("cuda")
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        if agent.cfg.history_length != 1:
            raise ValueError("Native runner currently supports single-frame source policies")
        obs = env.reset([args.seed] * args.batch_size)
        bank_path = args.benchmark_bank or args.validate_bank
        loaded = np.load(bank_path) if bank_path else None
        if loaded is not None and (loaded["prefix"].shape != (args.approach, 6)
                                   or loaded["plans"].shape[1:] != (args.horizon, 6)):
            raise ValueError("Saved bank prefix/horizon does not match runner arguments")
        prefix = []
        for t in range(args.approach):
            a = loaded["prefix"][t] if loaded is not None else agent.act(obs[:1], deterministic=True)[0]
            prefix.append(a.copy())
            obs, _, done, info = env.step(np.repeat(a[None], args.batch_size, axis=0))
            if np.any(done) or any(r.get("sim_error") for r in info):
                raise RuntimeError(f"Approach failed: {info}")
        snap = probe.take_snapshot(env, "parallel_trajopt", args.approach)
        result["initial_coverage"] = [r.upperarm_ratio for r in env.progress()]
        result["initial_position_component_spread_m"] = float(np.ptp(np.stack(snap["positions"]), axis=0).max())
        print(json.dumps(dict(approach_finished=True, initial_coverage=result["initial_coverage"])), flush=True)
        evaluator = IPCBatchEvaluator(env, snap, device)
        result["evaluations"] = evaluator.records
        if loaded is not None:
            plans = torch.as_tensor(loaded["plans"], device=device)
            if args.validate_bank:
                if len(plans) not in (4, 5):
                    raise ValueError("Expected saved reference/CEM/scaled/random controls, optionally with SAC")
                controls = torch.cat([plans[:4], plans[:1]])
            else:
                _, record = evaluator.evaluate(plans, args.repeats)
                result["benchmark_transitions_per_s"] = record["physical_decisions"] / record["seconds"]
        else:
            reference = []
            for _ in range(args.horizon):
                a = agent.act(obs[:1], deterministic=True)[0]
                reference.append(a.copy())
                obs, _, done, info = env.step(np.repeat(a[None], args.batch_size, axis=0))
                if np.any(done) or any(r.get("sim_error") for r in info):
                    raise RuntimeError(f"Reference failed: {info}")
            ref = torch.as_tensor(np.stack(reference), device=device)
            optimizer = TrajectoryCEM(ref, args.population, args.knots, args.seed)
            random_control = None
            best = optimizer.reference.clone()
            for iteration in range(args.iterations):
                knots, plans = optimizer.ask()
                if iteration == 0:
                    np.savez_compressed(args.out / "benchmark_bank.npz", prefix=np.stack(prefix), plans=plans.cpu().numpy())
                    random_control = plans[2].clone()
                scores, record = evaluator.evaluate(plans, args.repeats)
                if optimizer.tell(knots, scores):
                    best = plans[scores.argmax()].clone()
                print(json.dumps(dict(iteration=iteration, seconds=record["seconds"], scores=record["scores"])), flush=True)
                save()
            # Scale-only is capped by the same reference norms, so it cannot gain by
            # issuing larger commands. Random route is selected before any scoring.
            controls = torch.stack([optimizer.reference, best,
                                    project_actions(1.5 * optimizer.reference, optimizer.reference), random_control,
                                    optimizer.reference])
            np.savez_compressed(args.out / "plans.npz", prefix=np.stack(prefix), plans=controls.cpu().numpy())
        if loaded is None or args.validate_bank:
            _, validation = evaluator.evaluate(controls, args.repeats, agent, args.tail_steps, closed_loop_candidates=(4,))
            result["validation_names"] = ["reference", "cem", "scaled_capped", "fixed_random", "closed_loop_sac"]
            result["validation_index"] = len(evaluator.records) - 1
            result["validation_final_coverage"] = np.asarray(validation["coverage"])[..., -1].tolist()
            valid = np.asarray(validation["valid"]).all(axis=-1)
            sustained = np.asarray(validation["coverage"])[..., -min(12, args.horizon + args.tail_steps):].min(axis=-1)
            result["validation_sustained_valid_success"] = (valid & (sustained >= cfg.reward.success_upperarm_ratio)).tolist()
        result["physical_decisions"] = evaluator.decisions + args.batch_size * (args.approach + (0 if loaded is not None else args.horizon))
        result["completed"] = True
        save()
        print(json.dumps({k: v for k, v in result.items() if k in ("completed", "physical_decisions", "total_seconds", "validation_final_coverage", "benchmark_transitions_per_s")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
