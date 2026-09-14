"""Level 3, third experiment: the physics direction as an actor's training signal, critic frozen.

The greedy walks showed that a trained critic's state gradient, pushed through the solver's
one-decision Jacobian, is a usable direction at the elbow where the same critic's action
derivative is not. This script asks whether a parametric policy can absorb that direction from a
few hundred transitions: from a restored elbow state it collects transitions with the stochastic
policy and with the proxy direction plus noise, stores for each the physics direction
``g_u = ∂V(x')/∂u`` at the action taken (the critic differentiated through the observation, the
last frame's device solve), then fine-tunes three copies of the actor from the same weights on the
same batches with the critic frozen:

- ``sac``: SAC's own actor loss, ``α log π − min Q(s, π(s))`` through the frozen critic;
- ``phys``: ``−β · unit(g_u) · μ(s)``, the stored physics direction on the policy's mean action
  (a deterministic-policy-gradient surrogate with the simulator's Jacobian in place of the
  critic's action derivative), ``β`` set once so its first gradient matches the SAC loss's in norm;
- ``both``: their sum.

Each actor is then run deterministically for twelve decisions from the elbow state it was tuned
at and from a second state of the same cell it never saw, against the untouched actor.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from uipc_manip import physics_gradient_actor as actor_exp
from uipc_manip import physics_gradient_adjoint as adjoint
from uipc_manip import physics_gradient_probe as probe


# ------------------------------------------------------------------------ collection
def collect(env, snap: dict, agent, capture, feature, layout: dict, episodes: int, horizon: int, mode: str,
            proxy_dir: np.ndarray | None, noise: float, rng: np.random.Generator, box_m: float, box_rad: float) -> list[dict]:
    """Transitions from the restored state: ``mode`` 'policy' samples the stochastic policy, 'proxy'
    walks the proxy direction at the box with Gaussian action noise. Every transition carries the
    physics direction at the action taken."""
    import torch

    rows = []
    for ep in range(episodes):
        probe.restore(env, snap)
        for t in range(horizon):
            obs = env.observation()[0]
            if mode == "policy":
                u = agent.act(obs[None], deterministic=False)[0].astype(np.float64)
            else:
                u = actor_exp.unit_action(proxy_dir, env, box_m, box_rad) + rng.normal(0.0, noise, 6)
                u[3] = 0.0
                u = np.clip(u, -1.0, 1.0)
            out = probe.decision(env, u)
            positions = env.positions()[0].astype(np.float64)
            t0 = time.time()
            vg = actor_exp.value_and_gradient(agent, capture, positions)
            g_u, residual, _ = actor_exp.last_frame_gradient(feature, layout, vg["gradient"], {"offsets": np.asarray(env._offsets[0])}, env)
            executed = out["executed_m"] / out["commanded_m"] if out["commanded_m"] > 1e-9 else 1.0
            rows.append({"obs": obs.astype(np.float32), "action": u.astype(np.float32), "g_u": g_u, "g_norm": float(np.linalg.norm(g_u)),
                         "executed_fraction": float(executed), "device_residual": residual, "value_next": vg["value"],
                         "upperarm_ratio": out["upperarm_ratio"], "net_normal_n": out["net_normal_n"], "mode": mode, "episode": ep, "step": t,
                         "gradient_s": time.time() - t0})
        print(f"[finetune-collect] {mode} episode {ep + 1}/{episodes}: coverage {rows[-1]['upperarm_ratio']:.3f} "
              f"V {rows[-1]['value_next']:.2f} |g_u| {np.mean([r['g_norm'] for r in rows[-horizon:]]):.3g} "
              f"({np.mean([r['gradient_s'] for r in rows[-horizon:]]):.2f} s/gradient)", flush=True)
    return rows


# ------------------------------------------------------------------------ fine-tuning
def actor_losses(agent, actor, obs_batch, g_batch, beta: float):
    """The SAC actor loss through the frozen critic and the physics-direction loss, on one batch."""
    import torch

    from uipc_manip.sac import frozen_parameters

    mu, pi, log_pi, _ = actor(obs_batch)
    with frozen_parameters(agent.critic):
        q1, q2 = agent.critic(obs_batch, pi)
    alpha = agent._alpha_at(0).detach()
    sac = (alpha * log_pi - torch.min(q1, q2)).mean()
    phys = -(beta * (g_batch * mu).sum(dim=-1)).mean()
    return sac, phys


def grad_norm_of(actor, loss) -> float:
    import torch

    actor.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    total = 0.0
    for p in actor.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    actor.zero_grad(set_to_none=True)
    return float(np.sqrt(total))


def finetune(agent, rows: list[dict], arms: tuple[str, ...], updates: int, batch_size: int, seed: int, gate: float) -> dict:
    """Three copies of the actor from the same weights, the same batch sequence, critic frozen."""
    import torch

    device = agent.device
    obs = torch.as_tensor(np.stack([r["obs"] for r in rows]), device=device)
    g = np.stack([r["g_u"] for r in rows]).astype(np.float32)
    keep = np.array([r["executed_fraction"] >= gate and np.isfinite(r["device_residual"]) and r["g_norm"] > 0 for r in rows])
    g_unit = np.where(keep[:, None], g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12), 0.0)
    g_t = torch.as_tensor(g_unit, device=device)
    n = obs.shape[0]
    rng = np.random.default_rng(seed)
    order = [rng.integers(0, n, size=batch_size) for _ in range(updates)]

    # β once: the physics term's first gradient matches the SAC term's in norm, on the untouched actor.
    base = copy.deepcopy(agent.actor)
    base.train(True)
    idx = torch.as_tensor(order[0], device=device)
    batch = agent._unpack(obs[idx])
    sac0, phys0 = actor_losses(agent, base, batch, g_t[idx], 1.0)
    norm_sac, norm_phys = grad_norm_of(base, sac0), grad_norm_of(base, phys0)
    beta = norm_sac / max(norm_phys, 1e-12)
    del base
    result = {"beta": beta, "first_grad_norm_sac": norm_sac, "first_grad_norm_phys_unit_beta": norm_phys, "kept": int(keep.sum()), "rows": n,
              "arms": {}}
    print(f"[finetune] {n} transitions, {int(keep.sum())} pass the gate; β={beta:.3g} (SAC grad {norm_sac:.3g}, physics grad at β=1 {norm_phys:.3g})", flush=True)
    for arm in arms:
        actor = copy.deepcopy(agent.actor)
        actor.train(True)
        opt = torch.optim.Adam(actor.parameters(), lr=agent.cfg.actor_lr, betas=(agent.cfg.actor_beta, 0.999))
        trace = []
        t0 = time.time()
        for k in range(updates):
            idx = torch.as_tensor(order[k], device=device)
            batch = agent._unpack(obs[idx])
            sac, phys = actor_losses(agent, actor, batch, g_t[idx], beta)
            loss = {"sac": sac, "phys": phys, "both": sac + phys}[arm]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            max_norm = float(agent.cfg.grad_clip_max_norm) if agent.cfg.grad_clip_max_norm > 0 else float("inf")
            gn = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm))
            opt.step()
            if k % 50 == 0 or k == updates - 1:
                trace.append({"update": k, "sac_loss": float(sac.item()), "phys_loss": float(phys.item()), "grad_norm": gn})
        actor.train(False)
        result["arms"][arm] = {"actor": actor, "trace": trace, "seconds": time.time() - t0}
        print(f"[finetune] arm {arm}: {updates} updates in {time.time() - t0:.0f}s; sac loss {trace[0]['sac_loss']:.2f} -> {trace[-1]['sac_loss']:.2f}, "
              f"phys loss {trace[0]['phys_loss']:.3f} -> {trace[-1]['phys_loss']:.3f}", flush=True)
    return result


# ------------------------------------------------------------------------ evaluation
def deterministic_walk(env, snap: dict, agent, actor, steps: int) -> dict:
    import torch

    probe.restore(env, snap)
    rows = []
    for s in range(steps):
        obs = env.observation()[0]
        with torch.no_grad():
            batch = agent._unpack(torch.as_tensor(obs.reshape(1, -1), device=agent.device))
            mu, _, _, _ = actor(batch, compute_pi=False, compute_log_pi=False)
        out = probe.decision(env, mu.cpu().numpy().reshape(-1).astype(np.float64))
        rows.append(out)
    v = actor_exp.value(agent, env.observation()[0])
    return {"final_upperarm": rows[-1]["upperarm_ratio"], "delta_upperarm": rows[-1]["upperarm_ratio"] - snap["measure"]["upperarm_ratio"],
            "delta_axis_m": rows[-1]["upperarm_axis_m"] - snap["measure"]["upperarm_axis_m"], "final_value": v,
            "mean_net_normal_n": float(np.mean([r["net_normal_n"] for r in rows])), "executed_m": float(sum(r["executed_m"] for r in rows)),
            "executed_rad": float(sum(r["executed_rad"] for r in rows))}


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--garment", default="tshirt_26")
    p.add_argument("--body", type=int, default=14049)
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--train-state", default="elbow")
    p.add_argument("--eval-states", nargs="+", default=["elbow", "stall"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--policy-episodes", type=int, default=12)
    p.add_argument("--proxy-episodes", type=int, default=12)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--noise", type=float, default=0.3, help="action noise on the proxy episodes (action units)")
    p.add_argument("--updates", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--arms", nargs="+", default=("sac", "phys", "both"))
    p.add_argument("--gate", type=float, default=0.5, help="least executed fraction for a transition's physics direction to count")
    p.add_argument("--eval-repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stall-window", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = probe.build_env(args.garment, args.body, args.region, out / "work")
    record = {"garment": args.garment, "body": args.body, "checkpoint": str(args.checkpoint), "train_state": args.train_state,
              "policy_episodes": args.policy_episodes, "proxy_episodes": args.proxy_episodes, "horizon": args.horizon, "noise": args.noise,
              "updates": args.updates, "batch_size": args.batch_size, "gate": args.gate, "seed": args.seed}
    record_path = out / f"{args.garment}_{args.body}.finetune.json"

    def save():
        record["total_s"] = time.time() - t0
        record_path.write_text(json.dumps(record, indent=1, default=float) + "\n")

    try:
        import torch

        torch.manual_seed(args.seed)
        feature = adjoint.adjoint_feature(env)
        if feature is None:
            raise RuntimeError("this build has no LinearSystemAdjointFeature")
        agent = actor_exp.load_agent(Path(args.checkpoint), int(env.spec.point_budget), int(env.action_dim), args.device)
        capture = actor_exp.ObservationCapture(env)
        layout = adjoint.cloth_layout(env, 1.0)
        rec = json.loads(Path(args.probe_json).read_text())["snapshots"][0]
        g = rec["gradients"]
        key = str(g["epsilons"][len(g["epsilons"]) // 2])
        proxy_dir = np.zeros(6)
        proxy_dir[:3] = np.asarray(g["per_epsilon"][key]["mean"]["opening_axis_m"], dtype=np.float64)
        r = rec["rotation_gradients"]
        rkey = str(r["epsilons"][len(r["epsilons"]) // 2])
        proxy_dir[list(r["axes"])] = np.asarray(r["per_epsilon"][rkey]["mean"]["opening_axis_m"], dtype=np.float64)
        box_m, box_rad = float(env.cfg.max_translation), float(env.cfg.max_rotation)
        snaps, _ = probe.drive_to_snapshots(env, args.stall_window, 0.01, args.max_steps)
        train_snap = next(s for s in snaps if s["name"] == args.train_state)
        record["train_episode_step"] = train_snap["episode_step"]
        rng = np.random.default_rng(args.seed)

        t1 = time.time()
        rows = collect(env, train_snap, agent, capture, feature, layout, args.policy_episodes, args.horizon, "policy", None, 0.0, rng, box_m, box_rad)
        rows += collect(env, train_snap, agent, capture, feature, layout, args.proxy_episodes, args.horizon, "proxy", proxy_dir, args.noise, rng, box_m, box_rad)
        record["collect_s"] = time.time() - t1
        record["transitions"] = [{k: v for k, v in r.items() if k not in ("obs", "action", "g_u")} | {"action": r["action"].tolist(), "g_u": np.asarray(r["g_u"]).tolist()} for r in rows]
        record["collect_summary"] = {mode: {"coverage_end_mean": float(np.mean([r["upperarm_ratio"] for r in rows if r["mode"] == mode and r["step"] == args.horizon - 1])),
                                            "g_norm_mean": float(np.mean([r["g_norm"] for r in rows if r["mode"] == mode])),
                                            "gradient_s_mean": float(np.mean([r["gradient_s"] for r in rows if r["mode"] == mode]))}
                                     for mode in ("policy", "proxy")}
        save()

        tuned = finetune(agent, rows, tuple(args.arms), args.updates, args.batch_size, args.seed, args.gate)
        record["finetune"] = {k: v for k, v in tuned.items() if k != "arms"} | {"arms": {a: {"trace": v["trace"], "seconds": v["seconds"]} for a, v in tuned["arms"].items()}}
        save()

        actors = {"initial": agent.actor} | {a: v["actor"] for a, v in tuned["arms"].items()}
        record["evaluation"] = {}
        for name in args.eval_states:
            snap = next((s for s in snaps if s["name"] == name), None)
            if snap is None:
                continue
            record["evaluation"][name] = {"episode_step": snap["episode_step"], "actors": {}}
            for aname, actor in actors.items():
                reps = [deterministic_walk(env, snap, agent, actor, args.horizon) for _ in range(args.eval_repeats)]
                keys = ("final_upperarm", "delta_axis_m", "final_value", "mean_net_normal_n", "executed_m", "executed_rad")
                summary = {k: float(np.mean([r[k] for r in reps])) for k in keys} | {"std": {k: float(np.std([r[k] for r in reps])) for k in keys}, "repeats": reps}
                record["evaluation"][name]["actors"][aname] = summary
                print(f"[finetune-eval] {name}@{snap['episode_step']} {aname}: coverage {snap['measure']['upperarm_ratio']:.3f} -> "
                      f"{summary['final_upperarm']:.3f} ± {summary['std']['final_upperarm']:.3f} axis +{summary['delta_axis_m']*1e3:.1f}mm "
                      f"V {summary['final_value']:.2f} F={summary['mean_net_normal_n']:.0f}N exec={summary['executed_m']*1e3:.0f}mm "
                      f"rot={np.rad2deg(summary['executed_rad']):.0f}deg", flush=True)
                save()
    finally:
        save()
        env.close()


if __name__ == "__main__":
    main()
