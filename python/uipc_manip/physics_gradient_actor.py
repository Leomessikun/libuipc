"""Level 3, second experiment: the critic says where, the adjoint says how.

A trained SAC checkpoint supplies a value of the next state, ``V(x') = min(Q1, Q2)(s', μ(s'))``, and
the solver supplies how the next state moves with the command, so the actor's gradient is
``∂V/∂u = (∂V/∂x')ᵀ ∂x'/∂u``: the one-decision adjoint chain of ``physics_gradient_adjoint`` (six
exported systems, host factorisation) or the frame's own device solve on the last system alone.
``∂V/∂x'`` is the exact derivative of the critic through the environment's observation function
(camera visibility, voxel centroids, tool-relative packing), rebuilt in torch from the discrete
choices the environment made at ``x'``.

At each state the script compares that physics gradient with central differences of ``V`` over the
command, with SAC's own actor signal ``∂Q(s, a)/∂a``, and with the one-step proxy direction of the
rotation probe; then it walks twelve decisions greedily along each signal, recomputed every
decision, and reads the coverage — the same yardstick as the trajectory optimisation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from uipc_manip import physics_gradient_adjoint as adjoint
from uipc_manip import physics_gradient_probe as probe
from uipc_manip import physics_gradient_trajopt as trajopt
from uipc_manip.obs import FLAG_DEFORMABLE, ObsSpec


# ------------------------------------------------------------------------ the checkpoint
def load_agent(checkpoint: Path, point_budget: int, action_dim: int, device: str):
    from uipc_manip.sac import SACAgent, SACConfig

    payload = SACAgent.read_checkpoint(checkpoint)
    cfg = SACConfig.from_dict(payload["sac_config"])
    saved = payload.get("protocol", {})
    if "critic_action_mode" not in saved:
        # Checkpoints written before the dense critic carry the latent one (the reference's rejected
        # baseline, agent_docs/performance/2026-09-12-critic-architecture-defect.md).
        cfg = SACConfig.from_dict({**cfg.to_dict(), "critic_action_mode": "latent"})
    agent = SACAgent(ObsSpec(int(point_budget)), int(action_dim), cfg, device)
    ours = agent.protocol()
    if any(saved.get(k) != v for k, v in ours.items() if k in saved):
        raise ValueError(f"checkpoint protocol {saved} differs from this agent's {ours} on a shared key")
    agent.load(checkpoint, load_optimizers=False, strict_protocol=False)
    agent.train(False)
    agent.critic_action_mode = cfg.critic_action_mode
    return agent


# ------------------------------------------------------------------------ the observation, differentiable in the cloth
class ObservationCapture:
    """Runs the environment's own observation once and records the discrete choices it made — which
    cloth vertices were visible and which voxel each fell in — so the same observation can be rebuilt
    in torch as a function of the cloth positions."""

    def __init__(self, env):
        self.env = env
        self.voxel_size = float(env._batched_obs.cfg.voxel_size_m)

    def capture(self, positions: np.ndarray) -> dict:
        import torch

        from uipc_manip import dressing_obs

        calls = []
        original = dressing_obs.voxel_centroids_batched

        def hooked(pts, select, voxel_size, n_envs):
            calls.append((pts.detach().cpu().numpy().copy(), select.detach().cpu().numpy().copy()))
            return original(pts, select, voxel_size, n_envs)

        dressing_obs.voxel_centroids_batched = hooked
        try:
            flat = self.env.observation([positions])[0]
        finally:
            dressing_obs.voxel_centroids_batched = original
        if len(calls) != 2:
            raise RuntimeError(f"expected the arm and the cloth voxel calls, saw {len(calls)}")
        pts, select = calls[1]
        pts, select = pts[0], select[0]
        # The second call selects the visible cloth points of the padded cloud (arm, cloth, body
        # occluders); each selected row is one cloth vertex, found again by position.
        from scipy.spatial import cKDTree

        rows = np.nonzero(select)[0]
        dist, visible = cKDTree(positions.astype(np.float32)).query(pts[rows])
        if rows.size and (dist.max() > 1e-5 or np.unique(visible).size != visible.size):
            raise RuntimeError(f"selected cloud rows do not map one-to-one onto cloth vertices (max distance {dist.max():.2e})")
        visible = np.asarray(visible, dtype=np.int64)
        vox = np.floor(positions[visible].astype(np.float32) / self.voxel_size).astype(np.int64) + (1 << 15)
        key = (vox[:, 0] << 32) | (vox[:, 1] << 16) | vox[:, 2]
        uniq, inverse = np.unique(key, return_inverse=True)
        spec = self.env.spec
        pos, feat, valid, extra = spec.unpack_numpy(flat)
        deformable = np.nonzero(valid & (feat[:, FLAG_DEFORMABLE] > 0.5))[0]
        if deformable.shape[0] != uniq.shape[0]:
            raise RuntimeError(f"{deformable.shape[0]} deformable points in the observation, {uniq.shape[0]} voxel "
                               "centroids: the budget subsampled the cloud, which this capture does not follow")
        tool = self.env._anchor[0].astype(np.float32)
        centroids = np.zeros((uniq.shape[0], 3), dtype=np.float64)
        counts = np.zeros(uniq.shape[0])
        np.add.at(centroids, inverse, positions[visible])
        np.add.at(counts, inverse, 1.0)
        centroids /= counts[:, None]
        if not np.allclose(pos[deformable] + tool[None, :], centroids, atol=2e-5):
            raise RuntimeError("rebuilt voxel centroids do not match the observation's deformable points")
        return {"flat": flat, "visible": visible, "inverse": inverse, "counts": counts, "deformable_rows": deformable, "tool": tool}

    def torch_observation(self, cap: dict, x: "torch.Tensor", agent):
        """``(pos, feat, valid, extra)`` for the agent with the deformable rows a function of ``x`` [n, 3]."""
        import torch

        flat = torch.as_tensor(cap["flat"], device=agent.device).unsqueeze(0)
        pos, feat, valid, extra = agent.spec.unpack_torch(flat)
        visible = torch.as_tensor(cap["visible"], device=agent.device)
        inverse = torch.as_tensor(cap["inverse"], device=agent.device)
        counts = torch.as_tensor(cap["counts"], device=agent.device, dtype=x.dtype)
        sums = torch.zeros((counts.shape[0], 3), device=agent.device, dtype=x.dtype)
        sums = sums.index_add(0, inverse, x[visible])
        centroids = sums / counts.unsqueeze(-1)
        tool = torch.as_tensor(cap["tool"], device=agent.device, dtype=x.dtype)
        rows = torch.as_tensor(cap["deformable_rows"], device=agent.device)
        pos = pos.clone().to(x.dtype)
        pos[0, rows] = centroids - tool[None, :]
        if agent._cut_padding:
            used = valid.any(dim=0).nonzero()
            m = int(used.max()) + 1 if used.numel() else 1
            pos, feat, valid = pos[:, :m], feat[:, :m], valid[:, :m]
        return pos.to(torch.float32), feat, valid, extra


def value_and_gradient(agent, capture: ObservationCapture, positions: np.ndarray) -> dict:
    """``V(x') = min Q(s', μ(s'))`` and ``∂V/∂x'`` on every cloth vertex (3n), through the observation."""
    import torch

    cap = capture.capture(positions)
    x = torch.as_tensor(positions, device=agent.device, dtype=torch.float32).requires_grad_(True)
    obs = capture.torch_observation(cap, x, agent)
    mu, _, _, _ = agent.actor(obs, compute_pi=False, compute_log_pi=False)
    q1, q2 = agent.critic(obs, mu)
    v = torch.min(q1, q2).sum()
    v.backward()
    return {"value": float(v.item()), "q1": float(q1.item()), "q2": float(q2.item()), "mu": mu.detach().cpu().numpy().reshape(-1),
            "gradient": x.grad.detach().cpu().numpy().astype(np.float64).reshape(-1), "visible": int(cap["visible"].shape[0]),
            "voxels": int(cap["counts"].shape[0])}


def value(agent, flat_obs: np.ndarray) -> float:
    import torch

    with torch.no_grad():
        batch = agent._unpack(torch.as_tensor(np.asarray(flat_obs, dtype=np.float32).reshape(1, -1), device=agent.device))
        mu, _, _, _ = agent.actor(batch, compute_pi=False, compute_log_pi=False)
        q1, q2 = agent.critic(batch, mu)
        return float(torch.min(q1, q2).item())


def sac_action_gradient(agent, flat_obs: np.ndarray, action: np.ndarray | None = None) -> dict:
    """SAC's own actor signal ``∂ min Q(s, a)/∂a``, at the policy's mean action ``μ(s)`` or at ``action``
    (at the hold action it estimates the same quantity as the adjoint, from the learned action
    dependence instead of the solver's Jacobian)."""
    import torch

    batch = agent._unpack(torch.as_tensor(np.asarray(flat_obs, dtype=np.float32).reshape(1, -1), device=agent.device))
    with torch.no_grad():
        mu, _, _, _ = agent.actor(batch, compute_pi=False, compute_log_pi=False)
    a = mu.clone() if action is None else torch.as_tensor(np.asarray(action, dtype=np.float32).reshape(1, -1), device=agent.device)
    a = a.requires_grad_(True)
    q1, q2 = agent.critic(batch, a)
    torch.min(q1, q2).sum().backward()
    return {"mu": mu.cpu().numpy().reshape(-1), "dQ_da": a.grad.cpu().numpy().astype(np.float64).reshape(-1), "q": float(torch.min(q1, q2).item())}


# ------------------------------------------------------------------------ the physics actor gradient
def last_frame_gradient(feature, layout: dict, g: np.ndarray, frame: dict, env) -> tuple[np.ndarray, float, float]:
    """``∂L/∂u`` through the last frame's system alone, solved on the device: the cheap variant."""
    off, cnt, n = layout["dof_offset"], layout["dof_count"], layout["n"]
    rhs = np.zeros(int(feature.dof_count()))
    rhs[off:off + cnt] = g
    t0 = time.time()
    lam_full, residual = feature.solve(rhs, 1e-6, 32)
    seconds = time.time() - t0
    lam = np.asarray(lam_full)[off:off + cnt].reshape(n, 3)[layout["anchor_idx"]]
    held = lam * (layout["strength"] * layout["mass"][layout["anchor_idx"]])[:, None]
    grad = np.zeros(6)
    grad[:3] = held.sum(axis=0) * float(env.cfg.max_translation)
    grad[3:] = np.cross(frame["offsets"], held).sum(axis=0) * float(env.cfg.max_rotation)
    if getattr(env.cfg, "clip_rotation_to_yz", False):
        grad[3] = 0.0
    return grad, float(residual), seconds


def physics_gradient(env, snap: dict, u: np.ndarray, feature, layout: dict, agent, capture: ObservationCapture) -> dict:
    """One decision ``u`` from the restored state with its six systems; ``V`` at the result and
    ``∂V/∂u`` by the chain (host) and by the last frame alone (device)."""
    roll = trajopt.rollout(env, snap, u[None, :], feature)
    cap = capture.capture(roll["positions"])
    vg = value_and_gradient(agent, capture, roll["positions"])
    t0 = time.time()
    chain = trajopt.chain_gradient(roll["frames"], layout, vg["gradient"], u[None, :], env)
    chain_s = time.time() - t0
    last, residual, last_s = last_frame_gradient(feature, layout, vg["gradient"], roll["frames"][-1], env)
    return {"value": vg["value"], "chain": chain["per_action"][0], "last_frame": last, "device_residual": residual,
            "chain_s": chain_s, "device_s": last_s, "rollout_s": roll["seconds"], "positions": roll["positions"],
            "measure": roll["measure"], "visible": vg["visible"], "voxels": vg["voxels"],
            "gradient_norm": float(np.linalg.norm(vg["gradient"])), "executed_m": roll["decisions"][0]["executed_m"], "cap": cap}


def frozen_value(agent, capture: ObservationCapture, cap: dict, positions: np.ndarray) -> float:
    """``V`` at ``positions`` seen through the visibility and voxel membership captured at another
    state: the differentiable part of the observation alone."""
    import torch

    with torch.no_grad():
        x = torch.as_tensor(positions, device=agent.device, dtype=torch.float32)
        obs = capture.torch_observation(cap, x, agent)
        mu, _, _, _ = agent.actor(obs, compute_pi=False, compute_log_pi=False)
        q1, q2 = agent.critic(obs, mu)
        return float(torch.min(q1, q2).item())


def finite_difference(env, snap: dict, u: np.ndarray, agent, eps_m: float, eps_rad: float,
                      capture: ObservationCapture | None = None, cap: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Central differences of ``V(x'(u))`` on the five live action components, through the full
    observation and, when a capture is given, through the observation with its discrete choices
    frozen at ``cap`` (visibility and voxel membership of the unperturbed decision)."""
    max_t, max_r = float(env.cfg.max_translation), float(env.cfg.max_rotation)
    fd = np.zeros(6)
    fd_frozen = np.zeros(6)
    for k in range(6):
        if k == 3:
            continue
        e = eps_m / max_t if k < 3 else eps_rad / max_r
        sides, frozen = [], []
        for sign in (1.0, -1.0):
            a = u.copy()
            a[k] += sign * e
            roll = trajopt.rollout(env, snap, a[None, :])
            sides.append(value(agent, env.observation([roll["positions"]])[0]))
            if capture is not None and cap is not None:
                frozen.append(frozen_value(agent, capture, cap, roll["positions"]))
        fd[k] = (sides[0] - sides[1]) / (2.0 * e)
        if frozen:
            fd_frozen[k] = (frozen[0] - frozen[1]) / (2.0 * e)
    return fd, fd_frozen


# ------------------------------------------------------------------------ walks
def unit_action(direction: np.ndarray, env, step_m: float, step_rad: float) -> np.ndarray:
    """An action of the given translation and rotation magnitudes along a 6-D direction."""
    a = np.zeros(6)
    t, r = np.asarray(direction[:3], dtype=np.float64), np.asarray(direction[3:], dtype=np.float64).copy()
    if getattr(env.cfg, "clip_rotation_to_yz", False):
        r[0] = 0.0
    if np.linalg.norm(t) > 0:
        a[:3] = t / np.linalg.norm(t) * (step_m / float(env.cfg.max_translation))
    if np.linalg.norm(r) > 0:
        a[3:] = r / np.linalg.norm(r) * (step_rad / float(env.cfg.max_rotation))
    return np.clip(a, -1.0, 1.0)


def walk(env, snap: dict, steps: int, choose, agent, layout: dict) -> dict:
    """``choose(step, snapshot) -> action`` from the restored state; every step is snapshotted so a
    chooser may simulate ahead and come back. Returns the coverage and value trace."""
    probe.restore(env, snap)
    rows = []
    current = snap
    for s in range(steps):
        if s > 0:
            # Dumped after a decision, never straight after a recover (the backend's contact
            # buffers are then out of step with the recovered state and a later recover trips
            # an assertion in the contact system).
            current = probe.take_snapshot(env, f"walk{s}", int(env._episode_step))
        action = choose(s, current)
        probe.restore(env, current)
        out = probe.decision(env, action)
        v = value(agent, env.observation()[0])
        rows.append({"step": s, "action": np.asarray(action).tolist(), "upperarm_ratio": out["upperarm_ratio"],
                     "upperarm_axis_m": out["upperarm_axis_m"], "net_normal_n": out["net_normal_n"], "value": v,
                     "executed_m": out["executed_m"], "executed_rad": out["executed_rad"]})
    first, last = snap["measure"], rows[-1]
    return {"rows": rows, "delta_upperarm": last["upperarm_ratio"] - first["upperarm_ratio"], "final_upperarm": last["upperarm_ratio"],
            "delta_axis_m": last["upperarm_axis_m"] - first["upperarm_axis_m"], "final_value": last["value"],
            "mean_net_normal_n": float(np.mean([r["net_normal_n"] for r in rows])), "executed_m": float(sum(r["executed_m"] for r in rows)),
            "executed_rad": float(sum(r["executed_rad"] for r in rows))}


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--garment", default="tshirt_26")
    p.add_argument("--body", type=int, default=14049)
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--states", nargs="+", default=["elbow"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe-json", default=None, help="Level 1 record of this cell, for the proxy direction at the elbow")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eps-mm", type=float, default=2.0)
    p.add_argument("--eps-deg", type=float, default=1.0)
    p.add_argument("--fd-draws", type=int, default=2)
    p.add_argument("--walk-steps", type=int, default=12)
    p.add_argument("--walks", nargs="*", default=("physics", "sac", "sac_hold", "policy", "proxy", "expert"))
    p.add_argument("--skip-at", action="store_true", help="skip the gradient comparisons, walk only")
    p.add_argument("--stall-window", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = probe.build_env(args.garment, args.body, args.region, out / "work")
    record = {"garment": args.garment, "body": args.body, "checkpoint": str(args.checkpoint), "build_s": time.time() - t0,
              "eps_mm": args.eps_mm, "eps_deg": args.eps_deg, "walk_steps": args.walk_steps, "states": {}}
    record_path = out / f"{args.garment}_{args.body}.actor.json"

    def save():
        record["total_s"] = time.time() - t0
        record_path.write_text(json.dumps(record, indent=1, default=float) + "\n")
    try:
        feature = adjoint.adjoint_feature(env)
        if feature is None:
            raise RuntimeError("this build has no LinearSystemAdjointFeature")
        agent = load_agent(Path(args.checkpoint), int(env.spec.point_budget), int(env.action_dim), args.device)
        record["agent_updates"] = int(getattr(agent, "updates", -1))
        capture = ObservationCapture(env)
        layout = adjoint.cloth_layout(env, 1.0)
        proxy_dir = None
        if args.probe_json:
            rec = json.loads(Path(args.probe_json).read_text())["snapshots"][0]
            g = rec["gradients"]
            key = str(g["epsilons"][len(g["epsilons"]) // 2])
            proxy_dir = np.zeros(6)
            proxy_dir[:3] = np.asarray(g["per_epsilon"][key]["mean"]["opening_axis_m"], dtype=np.float64)
            r = rec["rotation_gradients"]
            rkey = str(r["epsilons"][len(r["epsilons"]) // 2])
            proxy_dir[list(r["axes"])] = np.asarray(r["per_epsilon"][rkey]["mean"]["opening_axis_m"], dtype=np.float64)
            proxy_dir[:3] /= np.linalg.norm(proxy_dir[:3])
            proxy_dir[3:] /= np.linalg.norm(proxy_dir[3:])
        snaps, _ = probe.drive_to_snapshots(env, args.stall_window, 0.01, args.max_steps)
        box_m, box_rad = float(env.cfg.max_translation), float(env.cfg.max_rotation)

        for name in args.states:
            snap = next((s for s in snaps if s["name"] == name), None)
            if snap is None:
                print(f"[actor] no {name} state on this cell", flush=True)
                continue
            entry = {"episode_step": snap["episode_step"], "measure": snap["measure"]}
            probe.restore(env, snap)
            obs0 = env.observation()[0]
            sac = sac_action_gradient(agent, obs0)
            entry["value_here"] = sac["q"]
            entry["policy_mu"] = sac["mu"].tolist()
            entry["sac_dQ_da"] = sac["dQ_da"].tolist()

            # The physics gradient at hold and at the policy's own action, the differences beside it.
            entry["at"] = {}
            for label, u in (() if args.skip_at else (("hold", np.zeros(6)), ("policy", sac["mu"].astype(np.float64)))):
                pg = physics_gradient(env, snap, u, feature, layout, agent, capture)
                sac_at = sac_action_gradient(agent, obs0, u)  # the learned action dependence at the same action
                pairs = [finite_difference(env, snap, u, agent, args.eps_mm * 1e-3, np.deg2rad(args.eps_deg), capture, pg["cap"])
                         for _ in range(args.fd_draws)]
                draws = [pr[0] for pr in pairs]
                frozen_draws = [pr[1] for pr in pairs]
                fd = np.mean(draws, axis=0)
                fd_frozen = np.mean(frozen_draws, axis=0)
                row = {"value_after": pg["value"], "dV_dx_norm": pg["gradient_norm"], "visible_vertices": pg["visible"], "voxels": pg["voxels"],
                       "chain": pg["chain"].tolist(), "last_frame": pg["last_frame"].tolist(), "finite_difference": fd.tolist(),
                       "fd_draws": [d.tolist() for d in draws], "device_residual": pg["device_residual"],
                       "finite_difference_frozen": fd_frozen.tolist(), "fd_frozen_draws": [d.tolist() for d in frozen_draws],
                       "cos_chain_fd_frozen": probe.cosine(pg["chain"], fd_frozen),
                       "ratio_chain_fd_frozen": float(np.linalg.norm(pg["chain"]) / np.linalg.norm(fd_frozen)) if np.linalg.norm(fd_frozen) > 0 else None,
                       "fd_frozen_draw_cosine": probe.cosine(frozen_draws[0], frozen_draws[1]) if len(frozen_draws) > 1 else None,
                       "sac_dQ_da_at": sac_at["dQ_da"].tolist(), "cos_sac_at_fd": probe.cosine(sac_at["dQ_da"], fd),
                       "cos_sac_at_chain": probe.cosine(sac_at["dQ_da"], pg["chain"]),
                       "chain_s": pg["chain_s"], "device_s": pg["device_s"], "rollout_s": pg["rollout_s"], "executed_m": pg["executed_m"],
                       "cos_chain_fd": probe.cosine(pg["chain"], fd), "cos_last_fd": probe.cosine(pg["last_frame"], fd),
                       "cos_chain_fd_translation": probe.cosine(pg["chain"][:3], fd[:3]), "cos_chain_fd_rotation": probe.cosine(pg["chain"][4:], fd[4:]),
                       "ratio_chain_fd": float(np.linalg.norm(pg["chain"]) / np.linalg.norm(fd)) if np.linalg.norm(fd) > 0 else None,
                       "cos_sac_fd": probe.cosine(sac["dQ_da"], fd), "cos_sac_chain": probe.cosine(sac["dQ_da"], pg["chain"]),
                       "fd_draw_cosine": probe.cosine(draws[0], draws[1]) if len(draws) > 1 else None}
                if proxy_dir is not None:
                    row["cos_chain_proxy"] = probe.cosine(pg["chain"], proxy_dir)
                    row["cos_fd_proxy"] = probe.cosine(fd, proxy_dir)
                    row["cos_sac_proxy"] = probe.cosine(sac["dQ_da"], proxy_dir)
                entry["at"][label] = row
                record["states"][name] = entry
                save()
                print(f"[actor] {args.garment}/{args.body} {name}@{snap['episode_step']} at {label}: V={pg['value']:.3f} "
                      f"|dV/dx|={pg['gradient_norm']:.3g} over {pg['visible']} visible vertices in {pg['voxels']} voxels; "
                      f"chain vs fd cos={row['cos_chain_fd']:.3f} (t {row['cos_chain_fd_translation']:.3f}, r {row['cos_chain_fd_rotation']:.3f}) "
                      f"ratio={row['ratio_chain_fd']}; frozen-observation fd: cos={row['cos_chain_fd_frozen']:.3f} ratio={row['ratio_chain_fd_frozen']} "
                      f"draws={row['fd_frozen_draw_cosine']}; last-frame vs fd cos={row['cos_last_fd']:.3f}; SAC dQ/da at mu vs fd cos={row['cos_sac_fd']:.3f}, "
                      f"at this action vs fd cos={row['cos_sac_at_fd']:.3f} vs chain cos={row['cos_sac_at_chain']:.3f}; "
                      f"fd draws cos={row['fd_draw_cosine']}; "
                      + (f"proxy: chain {row['cos_chain_proxy']:.2f} fd {row['cos_fd_proxy']:.2f} sac {row['cos_sac_proxy']:.2f}; " if proxy_dir is not None else "")
                      + f"chain {pg['chain_s']:.1f}s device {pg['device_s']*1e3:.0f}ms residual {pg['device_residual']:.1e}", flush=True)

            # Twelve greedy decisions along each signal, recomputed at every step.
            entry["walks"] = {}

            def physics_chooser(step, current):
                pg = physics_gradient(env, current, np.zeros(6), feature, layout, agent, capture)
                return unit_action(pg["chain"], env, box_m, box_rad)

            def sac_chooser(step, current):
                probe.restore(env, current)
                return unit_action(sac_action_gradient(agent, env.observation()[0])["dQ_da"], env, box_m, box_rad)

            def sac_hold_chooser(step, current):
                # ∂Q/∂a at the hold action: the learned counterpart of the adjoint at u = 0.
                probe.restore(env, current)
                return unit_action(sac_action_gradient(agent, env.observation()[0], np.zeros(6))["dQ_da"], env, box_m, box_rad)

            def policy_chooser(step, current):
                probe.restore(env, current)
                return sac_action_gradient(agent, env.observation()[0])["mu"]

            def expert_chooser(step, current):
                probe.restore(env, current)
                return np.asarray(probe.heuristic(env).actions()[0], dtype=np.float64)

            choosers = {"physics": physics_chooser, "sac": sac_chooser, "sac_hold": sac_hold_chooser, "policy": policy_chooser, "expert": expert_chooser}
            if proxy_dir is not None:
                choosers["proxy"] = lambda step, current: unit_action(proxy_dir, env, box_m, box_rad)
            for wname in args.walks:
                if wname not in choosers:
                    continue
                t1 = time.time()
                w = walk(env, snap, args.walk_steps, choosers[wname], agent, layout)
                w["seconds"] = time.time() - t1
                entry["walks"][wname] = w
                save()
                print(f"[actor-walk] {name}: {wname}: coverage {first_cov(snap):.3f} -> {w['final_upperarm']:.3f} "
                      f"axis +{w['delta_axis_m']*1e3:.1f}mm V {entry['value_here']:.2f} -> {w['final_value']:.2f} "
                      f"F={w['mean_net_normal_n']:.0f}N exec={w['executed_m']*1e3:.0f}mm rot={np.rad2deg(w['executed_rad']):.0f}deg ({w['seconds']:.0f}s)", flush=True)
            record["states"][name] = entry
    finally:
        save()
        env.close()


def first_cov(snap: dict) -> float:
    return float(snap["measure"]["upperarm_ratio"])


if __name__ == "__main__":
    main()
