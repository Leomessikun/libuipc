"""Does deciding the move and the measurement together beat deciding the move alone?

The joint operator's one behaviour that no ordinary reweighting has is a dead zone: an
action whose estimated gain does not clear a threshold on the scale of its own standard
error keeps its probability. On this task that should matter, because the failure every
long run shows - a peak of 3 of 25 at 174,264 transitions and 0 of 25 at the end - is
what chasing differences below the environment's noise looks like.

So this tests the operator where it claims to earn its keep. At states the policy
visits, a small set of four-decision action segments is evaluated cheaply; the operator
plans a policy change and an allocation from that thin evidence; the planned evaluations
are collected; the policy is updated against what was actually bought. Then **fresh,
independent evaluations** score four policies over the same candidates:

* ``stay``      the reference distribution, which is doing nothing;
* ``greedy``    all mass on the largest estimated mean;
* ``exponential`` the ordinary KL-regularised reweighting, with no error penalty;
* ``joint``     this operator.

If the dead zone is worth anything, ``joint`` beats ``greedy`` and ``exponential`` on
the held-out evaluations while the planning data is thin, and the three converge as the
budget grows. So the budget is swept, and swept *on one set of states and one pool of
evaluations*: a pool of independent evaluations is collected once per candidate, each
budget's plan draws the first ``n_i`` of that pool, and a disjoint held-out set scores
them all. The sweep is therefore paired and costs nothing beyond the pool.

A first smoke run showed why the sweep is necessary rather than decorative. At the
scale the calibration reports as resolving - a four-decision segment displaced by a full
normalised unit, 5.29 pooled standard deviations - every candidate clears its threshold,
the dead zone never binds, and the operator is the ordinary exponential update with
extra bookkeeping. Whatever it is worth has to show up where the evidence is thin.

Two things are truncated on purpose and both are limitations, not results. The score is
the task reward summed over a sixteen-decision window rather than a full continuation,
because a full one costs about 175 decisions and a day of this workstation would then
buy a few hundred policy updates; the calibration measured this window's margin at 5.29
pooled standard deviations for a four-decision segment. And the reference distribution
over a sampled candidate set is uniform, which is not the continuous policy's own
measure, so the KL here is over the finite set and not over the policy.
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

from uipc_manip import joint_policy_query as jpq  # noqa: E402
from uipc_manip import physics_gradient_probe as probe  # noqa: E402
from uipc_manip import train_sac  # noqa: E402


def exponential_update(p, mu, eta):
    """KL-regularised reweighting. The level of ``mu`` is a gauge freedom that the
    normalisation removes, so it is centred first: without that, returns of order ten
    against an eta of order a tenth overflow together and the update returns ``p``."""
    mu = np.asarray(mu, dtype=np.float64)
    q = np.asarray(p, dtype=np.float64) * np.exp(np.clip((mu - mu.max()) / eta, -60, 60))
    return q / q.sum()


def greedy(p, mu):
    q = np.zeros_like(np.asarray(p, dtype=np.float64))
    q[int(np.argmax(mu))] = 1.0
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--snapshot-steps", default="100,140,180")
    p.add_argument("--candidates", type=int, default=6, help="Action segments compared at each state")
    p.add_argument("--displacement", type=float, default=1.0,
                   help="Normalized action displacement of a candidate, the scale the calibration resolves")
    p.add_argument("--hold", type=int, default=4, help="Decisions a segment lasts")
    p.add_argument("--window", type=int, default=16, help="Decisions scored per evaluation")
    p.add_argument("--pilot", type=int, default=2, help="Evaluations per candidate the plan is made from")
    p.add_argument("--pool", type=int, default=8, help="Independent evaluations per candidate the plans draw from")
    p.add_argument("--budgets", default="4,8,16,32,64",
                   help="Evaluation budgets per state to sweep, in units of one evaluation")
    p.add_argument("--holdout", type=int, default=6, help="Disjoint evaluations per candidate for scoring")
    p.add_argument("--eta", type=float, default=None,
                   help="Absolute KL weight; by default it is scaled to the spread between candidates")
    p.add_argument("--eta-scale", type=float, default=1.0,
                   help="eta as a multiple of the standard deviation of the estimated means across candidates")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--n-min", type=float, default=1.0)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=8801)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    budgets = sorted(int(b) for b in args.budgets.split(",") if b.strip())
    if args.pilot > args.pool:
        raise ValueError("The pilot is drawn from the pool")
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.physics_gradient_actor import load_agent
    from uipc_manip.sac import SACAgent

    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False, workspace=str(args.out / "assets"))
    if steps[-1] + args.window >= cfg.horizon:
        raise ValueError("The last branch must end before the episode's time limit")
    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(cfg, num_envs=args.slots, cell_factory=LiveCellFactory(cfg.live))
    M = args.candidates
    result = dict(completed=False, checkpoint=str(args.checkpoint), cell=args.cell, env=cfg.to_dict(),
                  arguments={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                  physical_decisions=0, states=[], records=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        agent = load_agent(args.checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        rng = np.random.default_rng(args.seed)
        obs = env.reset([args.seed + i for i in range(args.slots)])
        step = 0

        def evaluate(snapshot, segments, which):
            """Run one evaluation of candidate ``which`` for every slot; return its score."""
            error = probe.restore(env, snapshot)
            if error > 1e-5:
                raise RuntimeError(f"Restore error {error} m")
            branch = env.observation()
            total = np.zeros(args.slots)
            for t in range(args.window):
                action = segments[:, which] if t < args.hold else agent.act(branch, deterministic=True)
                branch, _, done, rows = env.step(np.clip(action, -1, 1).astype(np.float32), reset_on_done=False)
                result["physical_decisions"] += args.slots
                if np.any(done) or any(r.get("sim_error") for r in rows):
                    raise RuntimeError(f"Branch for candidate {which} ended at {t}")
                total += np.asarray([float(r["task_reward"]) for r in rows])
            return total

        for target in steps:
            while step < target:
                obs, _, done, info = env.step(agent.act(obs, deterministic=True))
                result["physical_decisions"] += args.slots
                step += 1
                if np.any(done) or any(r.get("sim_error") for r in info):
                    raise RuntimeError(f"Approach terminated at decision {step}")
            snapshot = probe.take_snapshot(env, f"step{target}", target)
            base = np.asarray(agent.act(obs, deterministic=True), dtype=np.float64)
            # Candidate 0 is the policy's own segment; the rest displace it along fresh
            # random directions at the scale the calibration measured to resolve.
            segments = np.repeat(base[:, None, :], M, axis=1)
            for m in range(1, M):
                d = rng.standard_normal((args.slots, env.action_dim))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                segments[:, m] = np.clip(base + args.displacement * d, -1.0, 1.0)
            ids = [f"{garment}__{body}__seed{args.seed + i}__step{target}" for i in range(args.slots)]
            for i in range(args.slots):
                result["states"].append(dict(state=ids[i], slot=i, step=target,
                                             upperarm_ratio=float(info[i]["upperarm_ratio"]),
                                             segments=[[float(x) for x in segments[i, m]] for m in range(M)]))

            # One pool of independent evaluations per candidate, and a disjoint held-out
            # set. Every budget below plans and updates against prefixes of the pool, so
            # the sweep is paired across budgets, states and candidates.
            pool = np.zeros((args.slots, M, args.pool))
            for m in range(M):
                for r in range(args.pool):
                    pool[:, m, r] = evaluate(snapshot, segments, m)
                save()
            holdout = np.zeros((args.slots, M, args.holdout))
            for m in range(M):
                for r in range(args.holdout):
                    holdout[:, m, r] = evaluate(snapshot, segments, m)
                save()

            cost = np.full(M, float(args.window))
            reference = np.full(M, 1.0 / M)
            for i in range(args.slots):
                truth = holdout[i].mean(axis=1)
                pilot_mu = pool[i][:, :args.pilot].mean(axis=1)
                pilot_sigma = np.maximum(pool[i][:, :args.pilot].std(axis=1, ddof=1), 1e-9)
                eta = args.eta if args.eta else max(args.eta_scale * float(pilot_mu.std()), 1e-6)
                for budget in budgets:
                    # A budget that cannot even pay the floor buys no plan; the operator
                    # says so and the sweep records the point rather than losing the run.
                    if budget <= M * args.n_min:
                        result["records"].append(dict(state=ids[i], slot=i, step=target, budget=budget,
                                                      infeasible="the floor alone exhausts the budget"))
                        continue
                    out = jpq.plan(reference, pilot_mu, pilot_sigma, cost, float(budget * args.window),
                                   eta=eta, beta=args.beta, n_min=args.n_min)
                    # The plan may ask for more than the pool holds; it gets what exists,
                    # and the update is told the truth about what it got.
                    n_actual = np.minimum(np.maximum(np.rint(out.n).astype(int), int(args.n_min)), args.pool)
                    mu_hat = np.asarray([pool[i, m, :n_actual[m]].mean() for m in range(M)])
                    sigma_hat = np.maximum(np.asarray(
                        [pool[i, m, :n_actual[m]].std(ddof=1) if n_actual[m] > 1 else pilot_sigma[m]
                         for m in range(M)]), 1e-9)
                    q_joint = jpq.update(reference, mu_hat, sigma_hat, n_actual.astype(float),
                                         eta=eta, beta=args.beta)
                    policies = dict(stay=reference, greedy=greedy(reference, mu_hat),
                                    exponential=exponential_update(reference, mu_hat, eta),
                                    joint=q_joint)
                    result["records"].append(dict(
                        state=ids[i], slot=i, step=target, budget=budget, eta=float(eta),
                        truncated=bool((np.rint(out.n).astype(int) > args.pool).any()),
                        planned_n=[float(x) for x in out.n], actual_n=[int(x) for x in n_actual],
                        thresholds=[float(x) for x in out.thresholds],
                        pilot_mu=[float(x) for x in pilot_mu],
                        mu_hat=[float(x) for x in mu_hat], sigma_hat=[float(x) for x in sigma_hat],
                        holdout_mu=[float(x) for x in truth],
                        holdout_sem=[float(np.std(holdout[i, m], ddof=1) / np.sqrt(args.holdout))
                                     for m in range(M)],
                        best_by_holdout=int(np.argmax(truth)),
                        moved=int(np.count_nonzero(np.abs(q_joint - reference) > 1e-6)),
                        **{f"q_{k}": [float(x) for x in v] for k, v in policies.items()},
                        **{f"value_{k}": float(np.dot(v, truth)) for k, v in policies.items()}))
                rows = [r for r in result["records"][-len(budgets):] if "infeasible" not in r]
                print(json.dumps(dict(state=ids[i], step=target,
                                      budgets=[r["budget"] for r in rows],
                                      moved=[r["moved"] for r in rows],
                                      joint_minus_exp=[round(r["value_joint"] - r["value_exponential"], 4)
                                                       for r in rows])), flush=True)
            save()
            probe.restore(env, snapshot)
            obs = env.observation()
            step = target
        result["completed"] = True
        save()
        print(json.dumps(dict(states=len(result["states"]), decisions=result["physical_decisions"],
                              minutes=round(result["seconds"] / 60, 1))), flush=True)
    finally:
        env.close()
        gc.collect()


if __name__ == "__main__":
    main()
