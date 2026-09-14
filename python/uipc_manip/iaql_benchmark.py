"""Run the IAQL physics gate, fixed-target regression, and matched online smoke.

PYTHONPATH=build/python/src:python <torch-python> -m uipc_manip.iaql_benchmark --out output/iaql/run
All results are for the named direct-picker diagnostic, not robot dressing.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np
import torch

from .iaql import soft_targets
from .iaql_env import IAQLClothEnv, IAQLEnvConfig
from .obs import ObsSpec
from .sac import SACAgent, SACConfig


def agent_for(state_dim, seed=0, weight=0.0):
    torch.manual_seed(seed)
    return SACAgent(ObsSpec(3), 3, SACConfig(actor_type="state", critic_input="privileged",
        privileged_dim=int(state_dim), hidden_dim=128, batch_size=32, actor_lr=3e-4, critic_lr=3e-4,
        actor_log_std_min=-5, actor_log_std_max=1, adjoint_weight=weight, state_activation="silu"), "cpu")


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def labels(agent, rows, noise=None):
    return soft_targets(agent.actor, agent.critic_target, tensor([r["next_obs"] for r in rows]),
        tensor([[r["reward"]] for r in rows]), torch.ones(len(rows), 1),
        tensor([r["tangent"] for r in rows]), tensor([r["reward_gradient"] for r in rows]),
        agent.alpha.detach(), agent.cfg.discount, agent.cfg.reward_abs_bound/(1-agent.cfg.discount), noise)


def soft_value(agent, obs, reward, noise):
    from .models import gaussian_logprob, squash
    with torch.no_grad():
        state = tensor(obs)[None]
        mu, ls = agent.actor.head(state)
        _, action, lp = squash(mu, mu+ls.exp()*noise, gaussian_logprob(noise, ls))
        q1, q2 = agent.critic_target(state, action)
        value = reward + agent.cfg.discount*(torch.minimum(q1, q2)-agent.alpha*lp)
        return float(value.clamp(-agent.cfg.reward_abs_bound/(1-agent.cfg.discount), agent.cfg.reward_abs_bound/(1-agent.cfg.discount)))


def comparison(pred, ref):
    pred, ref = np.asarray(pred), np.asarray(ref)
    denom = np.linalg.norm(pred)*np.linalg.norm(ref)
    return dict(cosine=float(pred@ref/denom) if denom > 1e-12 else None,
                relative_error=float(np.linalg.norm(pred-ref)/max(np.linalg.norm(ref), 1e-8)))


def guided_action(env, rng, noise=0.3):
    direction = (env.goal-env.positions()[env.marker].mean(0))/env.cfg.max_translation
    return np.clip(direction + rng.normal(0, noise, 3), -0.85, 0.85)


def probe(env, agent, args):
    """The gate. Every snapshot's centre decision is captured once per export mode (the same
    restored state; the modes differ only in the matrix the backend leaves behind) and the
    differences are taken once, in the default mode, against all of them."""
    rng = np.random.default_rng(args.seed)
    noise = tensor([[0.2, -0.7, 0.4]])
    records = []
    modes = list(args.export_modes)
    for i in range(args.snapshots):
        env.reset(100+i)
        # Every snapshot follows at least one advance: a dump straight after a recover is not
        # restorable in every scene (2026-09-14 physics-gradient entry in the handoff).
        for _ in range(args.guided_steps_min + i % 4 * 4):
            env.step(guided_action(env, rng))
        snap = env.snapshot()
        action = rng.uniform(-0.5, 0.5, 3)
        captures = {}
        for mode in modes:
            env.restore(snap)
            env.set_export_mode(mode)
            captures[mode] = env.step(action, capture=True)
        env.set_export_mode(modes[0])
        center = captures[modes[0]]
        row = dict(center, next_obs=center["obs"])
        y, g = labels(agent, [row], noise)
        others = {mode: labels(agent, [dict(c, next_obs=c["obs"])], noise)[1][0] for mode, c in captures.items()}
        checks = []
        for eps in args.eps:
            gradients, rewards, state_derivatives = [], [], []
            for _ in range(args.fd_repeats):
                gv, gr, gs = [], [], []
                for axis in range(3):
                    e = np.eye(3)[axis]*eps
                    pair = []
                    for u in (action+e, action-e):
                        env.restore(snap)
                        out = env.step(u)
                        pair.append((soft_value(agent, out["obs"], out["reward"], noise), out["reward"], out["obs"]))
                    gv.append((pair[0][0]-pair[1][0])/(2*eps))
                    gr.append((pair[0][1]-pair[1][1])/(2*eps))
                    gs.append((pair[0][2]-pair[1][2])/(2*eps))
                gradients.append(gv)
                rewards.append(gr)
                state_derivatives.append(np.array(gs).T)
            fd = np.mean(gradients, 0)
            fd_state = np.mean(state_derivatives, 0)
            state_errors = {}
            for name, sl in [("position", slice(0, 3*env.n)), ("velocity", slice(3*env.n, 6*env.n)), ("tool", slice(6*env.n, 6*env.n+3))]:
                state_errors[name] = [comparison(center["tangent"][sl, axis], fd_state[sl, axis]) for axis in range(3)]
            by_mode = {}
            for mode, c in captures.items():
                by_mode[mode] = dict(bellman=comparison(others[mode].numpy(), fd),
                                     reward=comparison(c["reward_gradient"], np.mean(rewards, 0)),
                                     state_errors={name: [comparison(c["tangent"][sl, axis], fd_state[sl, axis]) for axis in range(3)]
                                                   for name, sl in [("position", slice(0, 3*env.n)), ("velocity", slice(3*env.n, 6*env.n))]},
                                     capture_forward_s=c["capture_forward_s"])
            checks.append(dict(epsilon=eps, bellman=comparison(g[0].numpy(), fd),
                               reward=comparison(center["reward_gradient"], np.mean(rewards, 0)),
                               fd_gradient=fd.tolist(), repeat_std=np.std(gradients, axis=0).tolist(), state_errors=state_errors,
                               by_mode=by_mode))
        records.append(dict(snapshot=i, action=action.tolist(), value=float(y), gradient=g[0].tolist(),
                            modes={mode: others[mode].tolist() for mode in modes},
                            checks=checks, capture_forward_s=center["capture_forward_s"], tangent_s=center["tangent_s"]))
        print(json.dumps({"probe": records[-1]}), flush=True)
    # Both step sizes must agree, so a favorable difference scale cannot hide an unstable local map.
    good = [all(c["bellman"]["cosine"] is not None and c["bellman"]["cosine"] >= .95 and
                c["bellman"]["relative_error"] <= .2 for c in r["checks"]) for r in records]
    return dict(records=records, accepted=good, accepted_fraction=float(np.mean(good)))


def collect(env, count, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(count):
        if i % 16 == 0:
            env.reset(seed*1000+i//16)
        obs = env.observation()
        action = guided_action(env, rng, noise=0.8) if i % 3 else rng.uniform(-.9, .9, 3)
        out = env.step(action, capture=True)
        rows.append(dict(obs=obs, action=action, next_obs=out["obs"], reward=out["reward"],
                         tangent=out["tangent"].astype(np.float32), reward_gradient=out["reward_gradient"], episode=i//16))
        if (i+1) % 16 == 0:
            print(json.dumps({"collected": i+1, "distance": out["distance"]}), flush=True)
    return rows


def episode_split(episodes):
    """Episode-disjoint train/test rows: every fourth episode is held out."""
    episodes = np.asarray(episodes)
    train = np.flatnonzero(episodes % 4 != 3)
    test = np.flatnonzero(episodes % 4 == 3)
    if not len(test) or not len(train):
        raise ValueError("Need at least 64 transitions for episode-disjoint fit evaluation")
    return train, test


def train_critic(agent, obs, action, y, g, train, rng, updates):
    """Fixed-teacher critic regression on paired labels; the actor and temperature never move."""
    for _ in range(updates):
        idx = rng.choice(train, agent.cfg.batch_size)
        agent._update_critic(obs[idx], action[idx], y[idx], obs[idx], torch.zeros(len(idx), 1),
                             state=obs[idx], paired_targets=(y[idx], g[idx], torch.ones(len(idx))))


def head_errors(agent, obs, action, y, g, test):
    """Held-out value and slope errors of both Q heads against the paired labels."""
    a = action[test].clone().requires_grad_(True)
    qs = agent.critic(obs[test], a)
    errors = []
    for q in qs:
        slope = torch.autograd.grad(q.sum(), a, retain_graph=True)[0]
        errors.append(dict(value_mse=float((q.detach()-y[test]).square().mean()),
                           slope_mse=float((slope-g[test]).square().mean()),
                           cosine=float(torch.nn.functional.cosine_similarity(slope, g[test]).mean())))
    return errors


def fit(env, rows, args):
    teacher = agent_for(env.obs_dim, args.seed+10)
    teacher.save(args.out/"fixed_teacher.pt", step=0, metadata={"phase": "fixed_teacher_regression"})
    generator = torch.Generator().manual_seed(args.seed+20)
    noise = torch.randn((len(rows), 3), generator=generator)
    y, g = labels(teacher, rows, noise)
    obs, action = tensor([r["obs"] for r in rows]), tensor([r["action"] for r in rows])
    train, test = episode_split([r["episode"] for r in rows])
    scale = max(float(g[train].square().mean().sqrt()), .01)
    results = []
    for weight in [0.0, args.beta]:
        agent = agent_for(env.obs_dim, args.seed+30, weight)
        agent.cfg.adjoint_gradient_scale = scale
        train_critic(agent, obs, action, y, g, train, np.random.default_rng(args.seed+40), args.fit_updates)
        result = dict(weight=weight, train_rows=len(train), test_rows=len(test), gradient_scale=scale,
                      heads=head_errors(agent, obs, action, y, g, test))
        results.append(result)
        agent.save(args.out/f"fit_beta_{weight}.pt", step=args.fit_updates, metadata={"phase": "fixed_teacher_regression"})
        print(json.dumps({"fit": result}), flush=True)
    np.savez_compressed(args.out/"fixed_dataset.npz", obs=obs.numpy(), actions=action.numpy(), targets=y.numpy(),
                        gradients=g.numpy(), episodes=np.array([r["episode"] for r in rows]),
                        next_obs=np.stack([r["next_obs"] for r in rows]), rewards=np.array([r["reward"] for r in rows]),
                        tangents=np.stack([r["tangent"] for r in rows]), reward_gradients=np.stack([r["reward_gradient"] for r in rows]),
                        policy_noise=noise.numpy())
    return results


def refit(args):
    """No simulator: re-fit critics on a saved fixed-teacher dataset. A weight sweep, the design's
    shuffled-label control (training slopes permuted across rows, held-out slopes intact) and the
    constant train-mean slope as the trivial predictor. Writes ``refit.json`` next to the dataset."""
    data = np.load(args.out/"fixed_dataset.npz")
    obs, action = torch.as_tensor(data["obs"]), torch.as_tensor(data["actions"])
    y, g = torch.as_tensor(data["targets"]), torch.as_tensor(data["gradients"])
    train, test = episode_split(data["episodes"])
    scale = max(float(g[train].square().mean().sqrt()), .01)
    mean_slope = g[train].mean(0, keepdim=True).expand(len(test), -1)
    results = [dict(weight=None, labels="train_mean_slope", train_rows=len(train), test_rows=len(test), gradient_scale=scale,
                    heads=[dict(value_mse=None, slope_mse=float((mean_slope-g[test]).square().mean()),
                                cosine=float(torch.nn.functional.cosine_similarity(mean_slope, g[test]).mean()))])]
    for weight, shuffled in [(w, False) for w in args.refit_weights] + [(w, True) for w in args.refit_weights if w > 0]:
        train_labels = g.clone()
        if shuffled:
            train_labels[train] = g[np.random.default_rng(args.seed+70).permutation(train)]
        agent = agent_for(obs.shape[1], args.seed+30, weight)
        agent.cfg.adjoint_gradient_scale = scale
        train_critic(agent, obs, action, y, train_labels, train, np.random.default_rng(args.seed+40), args.fit_updates)
        result = dict(weight=weight, labels="shuffled" if shuffled else "paired", train_rows=len(train), test_rows=len(test),
                      gradient_scale=scale, heads=head_errors(agent, obs, action, y, g, test))
        results.append(result)
        print(json.dumps({"refit": result}), flush=True)
    (args.out/"refit.json").write_text(json.dumps(dict(seed=args.seed, fit_updates=args.fit_updates, results=results), indent=2))
    return results


def evaluate(env, agent, seeds, horizon):
    episodes = []
    for seed in seeds:
        obs = env.reset(seed)
        total = 0.0
        for _ in range(horizon):
            out = env.step(agent.act(obs[None], deterministic=True)[0])
            obs = out["obs"]
            total += out["reward"]
        episodes.append(dict(seed=seed, return_=total, distance=out["distance"], success=out["success"]))
    return episodes


def sidecar_batch(batch, state_dim, action_dim=3, shuffle=None):
    """Mechanics for the rows the bounded sidecar still holds; the others are valid 0 with zero tangents.
    ``shuffle`` (a Generator) permutes the mechanics among the valid rows: the design's shuffled-label
    control, matched to the paired arm in sidecar lifetime, sampling, weight and scale."""
    has = torch.tensor([("tangent" in r) for r in batch])
    tangent = torch.stack([tensor(r["tangent"]) if "tangent" in r else torch.zeros(state_dim, action_dim) for r in batch])
    reward_gradient = torch.stack([tensor(r["reward_gradient"]) if "tangent" in r else torch.zeros(action_dim) for r in batch])
    if shuffle is not None:
        rows = torch.nonzero(has).reshape(-1)
        if len(rows) > 1:
            perm = rows[torch.as_tensor(shuffle.permutation(len(rows)))]
            tangent[rows], reward_gradient[rows] = tangent[perm], reward_gradient[perm]
    return dict(tangent=tangent, reward_gradient=reward_gradient, valid=has.float())


def online(env, args):
    results = []
    eval_seeds = [9000+k for k in range(args.eval_episodes)]
    for weight in ([0.0, args.beta] if args.online_weights is None else args.online_weights):
        agent = agent_for(env.obs_dim, args.seed+50, weight)
        rng = np.random.default_rng(args.seed+60)
        shuffle_rng = np.random.default_rng(args.seed+70)
        # The TD replay keeps every transition; the mechanics sidecar is bounded and evicts on its own,
        # so an old row keeps learning values after its tangent is gone (design: replay and target freshness).
        replay, sidecar = [], collections.deque()
        t0, eval_s = time.monotonic(), 0.0
        obs = env.reset(args.seed)
        last_stats, progress = {}, []
        for i in range(args.online_steps):
            action = rng.uniform(-1, 1, 3) if i < 64 else agent.act(obs[None], deterministic=False)[0]
            out = env.step(action, capture=weight > 0)
            row = dict(obs=obs, action=action, reward=out["reward"], next_obs=out["obs"])
            if weight:
                row.update(tangent=out["tangent"].astype(np.float32), reward_gradient=out["reward_gradient"])
                sidecar.append(row)
                while len(sidecar) > args.tangent_rows:
                    old = sidecar.popleft()
                    del old["tangent"], old["reward_gradient"]
            replay.append(row)
            if args.replay_rows and len(replay) > args.replay_rows:
                del replay[:len(replay)-args.replay_rows]
            if len(replay) >= 64:
                for _ in range(args.updates_per_step):
                    batch = [replay[k] for k in rng.integers(0, len(replay), 32)]
                    kw = sidecar_batch(batch, env.obs_dim, shuffle=shuffle_rng if args.shuffle_labels else None) if weight else {}
                    last_stats = agent.update_state_batch(tensor([r["obs"] for r in batch]), tensor([r["action"] for r in batch]),
                        tensor([[r["reward"]] for r in batch]), tensor([r["next_obs"] for r in batch]), torch.ones(32, 1), **kw)
            if (i+1) % 64 == 0:
                print(json.dumps({"online_weight": weight, "steps": i+1, "stats": last_stats}), flush=True)
            if args.eval_every and (i+1) % args.eval_every == 0 and i+1 < args.online_steps:
                # The snapshot follows this decision's advance (a dump straight after a recover is not restorable).
                t1 = time.monotonic()
                snap = env.snapshot()
                episodes = evaluate(env, agent, eval_seeds, args.eval_steps)
                env.restore(snap)
                eval_s += time.monotonic()-t1
                progress.append(dict(steps=i+1, training_s=time.monotonic()-t0-eval_s, evaluation=episodes))
                print(json.dumps({"online_weight": weight, "steps": i+1, "evaluation": episodes}), flush=True)
            obs = env.reset(args.seed+i+1) if out["done"] else out["obs"]
        duration = time.monotonic()-t0-eval_s
        evaluation = evaluate(env, agent, eval_seeds, args.eval_steps)
        result = dict(weight=weight, steps=args.online_steps, training_s=duration, evaluation=evaluation, progress=progress,
                      stats=last_stats, replay_rows=len(replay), sidecar_rows=len(sidecar),
                      labels="shuffled" if args.shuffle_labels and weight else "paired")
        agent.save(args.out/f"online_beta_{weight}.pt", step=args.online_steps, metadata=result)
        results.append(result)
        print(json.dumps({"online_result": result}), flush=True)
    return results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--friction", type=float, default=0)
    p.add_argument("--velocity-tolerance", type=float, default=.001)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshots", type=int, default=4)
    p.add_argument("--guided-steps-min", type=int, default=0,
                   help="guided steps before every probe snapshot, on top of 4*(i mod 4); use 1 so no snapshot dumps straight after a recover")
    p.add_argument("--export-modes", nargs="+", default=["last_iterate"],
                   choices=["last_iterate", "converged", "converged_raw"],
                   help="backend export modes to capture each probe decision with; the first is the run's mode")
    p.add_argument("--eps", type=float, nargs="+", default=[.03, .1])
    p.add_argument("--fd-repeats", type=int, default=2)
    p.add_argument("--collect", type=int, default=128)
    p.add_argument("--fit-updates", type=int, default=1000)
    p.add_argument("--online-steps", type=int, default=256)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=0, help="evaluate every N online steps from a snapshot (0: only at the end)")
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--replay-rows", type=int, default=0, help="TD replay bound (0: keep every transition)")
    p.add_argument("--tangent-rows", type=int, default=1024, help="bounded mechanics sidecar; older rows learn values only")
    p.add_argument("--beta", type=float, default=.1)
    p.add_argument("--refit-weights", type=float, nargs="+", default=[0.0, 0.01, 0.1, 1.0])
    p.add_argument("--shuffle-labels", action="store_true",
                   help="online control: permute each batch's mechanics among its valid rows (everything else matched)")
    p.add_argument("--online-weights", type=float, nargs="+", default=None,
                   help="online arms to run in this process (default: 0 and --beta); one arm per process runs the pair in parallel")
    p.add_argument("--phase", choices=["probe", "all", "refit", "online"], default="all",
                   help="refit re-fits critics on --out's saved fixed dataset without a simulator; "
                        "online skips the gate and fixed-teacher fit already recorded for these arguments")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.phase == "refit":
        refit(args)
        return
    t0 = time.monotonic()
    env = IAQLClothEnv(args.out/"world", IAQLEnvConfig(friction=args.friction, velocity_tolerance=args.velocity_tolerance,
                                                        export_mode=args.export_modes[0]))
    report = dict(environment=env.describe(), seed=args.seed, state_activation="silu",
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    path = args.out/"report.json"
    def save():
        report["elapsed_s"] = time.monotonic()-t0
        path.write_text(json.dumps(report, indent=2))
    if args.phase != "online":
        report["probe"] = probe(env, agent_for(env.obs_dim, args.seed), args)
        save()
        if args.phase == "probe":
            return
        if report["probe"]["accepted_fraction"] < .75:
            report["blocked_stage"] = "physics gate: inspect differences before trusting derivative labels"
            save()
            print(json.dumps({"gate_failed": report["blocked_stage"]}), flush=True)
            return
        rows = collect(env, args.collect, args.seed)
        report["fit"] = fit(env, rows, args)
        save()
    report["online"] = online(env, args)
    save()


if __name__ == "__main__":
    main()
