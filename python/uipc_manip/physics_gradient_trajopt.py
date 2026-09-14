"""Level 3, first experiment: open-loop trajectory optimisation with the physics gradient.

From a restored state, ``h`` decisions ``a_1..a_h`` (6-D actions: translation and rotation of the
gripper, the x-rotation clipped by the environment) are improved by projected gradient ascent on a
terminal objective ``L(x_end)``: the reference's upper-arm coverage (how far past the elbow the
sleeve's leading triangle sits on the shoulder→elbow ray) plus the axis reading (the opening's
centroid on the elbow→shoulder axis), both in metres, so that the objective is smooth before the
sleeve reaches the upper arm and is the reward once it has.

The gradient is the adjoint chain of ``physics_gradient_adjoint`` over all ``6h`` frames: the
systems the solver assembled are exported by the ``LinearSystemAdjointFeature`` after every frame,
the reverse pass ``H_f λ_f = ĝ_f`` runs backwards through the BDF1 inertia coupling, and the
commands enter through the soft position constraint's aim positions of the held vertices —
translation through the anchor, rotation through the held offsets (``δaim = δθ × offset``, carried
to later frames by the rotations executed in between). Every derivative is taken with respect to
the *executed* command: a substep the tether or the no-move rule refused contributes nothing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from uipc_manip import physics_gradient_adjoint as adjoint
from uipc_manip import physics_gradient_probe as probe
from uipc_manip.dressing_reward import line_triangles

ROTATION_AXES = (4, 5)  # the environment clips the rotation about x


# ------------------------------------------------------------------------ rigid bookkeeping
def kabsch(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """The rotation R with after ≈ R before for two sets of offsets about the same anchor."""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    u, _, vt = np.linalg.svd(before.T @ after)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    return vt.T @ np.diag([1.0, 1.0, d]) @ u.T


def rotation_angle(r: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


# ------------------------------------------------------------------------ the terminal objective
def coverage_objective(env, positions: np.ndarray) -> tuple[float, np.ndarray, bool]:
    """The reference's upper-arm distance [m] (ray from the shoulder toward the elbow, first sleeve
    triangle hit, measured back from the elbow) and its gradient on the three vertices of that
    triangle (3n), by central differences of the one-triangle intersection; zero when no polygon
    triangle sits on the upper arm."""
    cell = env.cells[0]
    tri_idx = np.asarray(cell.polygon_triangles(), dtype=np.int64)
    shoulder, elbow = np.asarray(cell.shoulder, dtype=np.float64), np.asarray(cell.elbow, dtype=np.float64)
    d = elbow - shoulder
    d = d / np.linalg.norm(d)

    def progress(tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hit, pts = line_triangles(shoulder, d, tris)
        prog = np.sum((pts - elbow[None, :]) * (-d)[None, :], axis=1)
        return prog, hit & (prog >= 0.0)

    n = positions.shape[0]
    g = np.zeros((n, 3))
    prog, valid = progress(positions[tri_idx])
    if not valid.any():
        return 0.0, g.reshape(-1), False
    best = int(np.argmin(np.where(valid, prog, np.inf)))
    value = float(prog[best])
    tri = tri_idx[best]
    base = positions[tri].astype(np.float64)
    eps = 1e-6
    for v in range(3):
        for c in range(3):
            sides = []
            for sign in (1.0, -1.0):
                p = base.copy()
                p[v, c] += sign * eps
                pr, va = progress(p[None])
                sides.append(pr[0] if va[0] else np.nan)
            g[tri[v], c] = (sides[0] - sides[1]) / (2.0 * eps)
    g[~np.isfinite(g)] = 0.0
    return value, g.reshape(-1), True


def objective(env, positions: np.ndarray, layout: dict) -> dict:
    """L = axis reading + coverage distance [m], with ∂L/∂x on the cloth (3n)."""
    axis_value = probe.upperarm_axis(env, [positions])
    cov_value, cov_grad, on_upperarm = coverage_objective(env, positions)
    return {"value": float(axis_value + cov_value), "axis_m": float(axis_value), "coverage_m": cov_value,
            "on_upperarm": on_upperarm, "gradient": adjoint.axis_gradient(layout) + cov_grad}


# ------------------------------------------------------------------------ rollouts
def rollout(env, snap: dict, actions: np.ndarray, feature=None) -> dict:
    """``h`` decisions from the restored state. With ``feature`` every frame's assembled system is
    exported and kept, with the anchor and held offsets the frame was solved against."""
    actions = np.asarray(actions, dtype=np.float64)
    restore_error = probe.restore(env, snap)
    frames: list[dict] = []
    original = env._sim_step

    def hooked():
        original()
        entry = {"anchor": env._anchor[0].copy(), "offsets": np.asarray(env._offsets[0]).copy()}
        if feature is not None:
            rows, cols, values, _ = feature.export_system()
            entry["H"] = adjoint.system_to_matrix(rows, cols, values, int(feature.dof_count()))
        frames.append(entry)

    decisions = []
    env._sim_step = hooked
    t0 = time.time()
    try:
        for t in range(actions.shape[0]):
            start_anchor, start_offsets = env._anchor[0].copy(), np.asarray(env._offsets[0]).copy()
            first = len(frames)
            out = probe.decision(env, actions[t])
            for k, fr in enumerate(frames[first:]):
                fr["decision"] = t
                fr["substep"] = k + 1
                fr["start_anchor"] = start_anchor
                fr["start_offsets"] = start_offsets
            decisions.append(out)
    finally:
        env._sim_step = original
    positions = env.positions()[0].astype(np.float64)
    return {"restore_error_m": restore_error, "frames": frames, "decisions": decisions, "positions": positions,
            "seconds": time.time() - t0, "measure": probe.measure(env)}


def summarise(env, roll: dict, layout: dict) -> dict:
    obj = objective(env, roll["positions"], layout)
    rows = roll["decisions"]
    return {"objective_m": obj["value"], "axis_m": obj["axis_m"], "coverage_m": obj["coverage_m"], "on_upperarm": obj["on_upperarm"],
            "upperarm_ratio": float(roll["measure"]["upperarm_ratio"]), "net_normal_n": float(roll["measure"]["net_normal_n"]),
            "mean_net_normal_n": float(np.mean([r["net_normal_n"] for r in rows])),
            "executed_m": float(sum(r["executed_m"] for r in rows)), "commanded_m": float(sum(r["commanded_m"] for r in rows)),
            "executed_rad": float(sum(r["executed_rad"] for r in rows)), "commanded_rad": float(sum(r["commanded_rad"] for r in rows)),
            "seconds": roll["seconds"]}


# ------------------------------------------------------------------------ the chain over h decisions
def executed_ratios(frames: list[dict], actions: np.ndarray, env) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Per frame: the executed fraction of its decision's commanded translation and rotation so far
    (k/6 when the whole command executed, less where substeps were refused), and the rotation R_f
    of the held offsets since the start of the rollout."""
    max_t, max_r = float(env.cfg.max_translation), float(env.cfg.max_rotation)
    repeat = int(env.cfg.action_repeat)
    trans = np.zeros(len(frames))
    rot = np.zeros(len(frames))
    R = []
    offsets_0 = frames[0]["start_offsets"]
    for i, fr in enumerate(frames):
        t, k = fr["decision"], fr["substep"]
        cmd_t = np.asarray(actions[t, :3]) * max_t
        cmd_r = probe.commanded_rotation(env, actions[t])
        moved = np.linalg.norm(fr["anchor"] - fr["start_anchor"])
        trans[i] = moved / np.linalg.norm(cmd_t) if np.linalg.norm(cmd_t) > 1e-12 else k / repeat
        turned = rotation_angle(kabsch(fr["start_offsets"], fr["offsets"]))
        rot[i] = turned / np.linalg.norm(cmd_r) if np.linalg.norm(cmd_r) > 1e-12 else k / repeat
        R.append(kabsch(offsets_0, fr["offsets"]))
    return trans, rot, R


def chain_gradient(frames: list[dict], layout: dict, g_final: np.ndarray, actions: np.ndarray, env) -> dict:
    """∂L/∂a_t for every decision (h × 6, action units) through the adjoint chain of all frames;
    each frame's system is factorised when the reverse pass reaches it and dropped afterwards."""
    off, cnt, n = layout["dof_offset"], layout["dof_count"], layout["n"]
    m3 = np.repeat(layout["mass"], 3)
    anchor = layout["anchor_idx"]
    k_held = (layout["strength"] * layout["mass"][anchor])[:, None]
    F = len(frames)
    h = actions.shape[0]
    repeat = int(env.cfg.action_repeat)
    lam_next = np.zeros(cnt)
    lam_next2 = np.zeros(cnt)
    dL_daim = [None] * F  # per frame: (held, 3) = K_I λ_{f,I}
    t0 = time.time()
    for f in range(F - 1, -1, -1):
        ghat = np.zeros(frames[f]["H"].shape[0])
        ghat[off:off + cnt] = g_final if f == F - 1 else 2.0 * m3 * lam_next - m3 * lam_next2
        lu = scipy.sparse.linalg.splu(frames[f]["H"].tocsc())
        lam = lu.solve(ghat)[off:off + cnt]
        dL_daim[f] = lam.reshape(n, 3)[anchor] * k_held
        lam_next2, lam_next = lam_next, lam
    reverse_s = time.time() - t0

    trans_ratio, rot_ratio, R = executed_ratios(frames, actions, env)
    # The executed fraction of a whole decision carries its command into every later frame.
    last_of = {}
    for i, fr in enumerate(frames):
        last_of[fr["decision"]] = i
    grad_m = np.zeros((h, 3))
    grad_rad = np.zeros((h, 3))
    for t in range(h):
        R_end = R[last_of[t]]
        for i, fr in enumerate(frames):
            if fr["decision"] < t:
                continue
            held = dL_daim[i]
            torque = np.cross(fr["offsets"], held).sum(axis=0)  # Σ_I offset_I × K_I λ_I
            if fr["decision"] == t:
                grad_m[t] += trans_ratio[i] * held.sum(axis=0)
                grad_rad[t] += rot_ratio[i] * torque
            else:
                grad_m[t] += trans_ratio[last_of[t]] * held.sum(axis=0)
                grad_rad[t] += rot_ratio[last_of[t]] * ((R[i] @ R_end.T).T @ torque)
    grad_action = np.zeros((h, 6))
    grad_action[:, :3] = grad_m * float(env.cfg.max_translation)
    grad_action[:, 3:] = grad_rad * float(env.cfg.max_rotation)
    if getattr(env.cfg, "clip_rotation_to_yz", False):
        grad_action[:, 3] = 0.0
    return {"per_action": grad_action, "per_metre": grad_m, "per_radian": grad_rad, "reverse_s": reverse_s, "dL_daim": dL_daim,
            "executed_translation_ratio": trans_ratio.tolist(), "executed_rotation_ratio": rot_ratio.tolist(),
            "frames": F, "decisions": h}


# ------------------------------------------------------------------------ checks against differences
def finite_difference_checks(env, snap: dict, actions: np.ndarray, layout: dict, chain_per_action: np.ndarray,
                             decisions: list[int], eps_m: float, eps_rad: float, repeats: int = 1) -> list[dict]:
    """Central differences of L after all h decisions with respect to the chosen decisions' five live
    action components, against the chain's rows; with ``repeats`` > 1 the differences are drawn again
    and their agreement with themselves is the noise floor."""
    h = actions.shape[0]
    max_t, max_r = float(env.cfg.max_translation), float(env.cfg.max_rotation)
    out = []
    for t in decisions:
        draws = []
        for _ in range(repeats):
            fd = np.zeros(6)
            for k in range(6):
                if k == 3:
                    continue
                e = eps_m / max_t if k < 3 else eps_rad / max_r
                sides = []
                for sign in (1.0, -1.0):
                    a = actions.copy()
                    a[t, k] += sign * e
                    sides.append(objective(env, rollout(env, snap, a)["positions"], layout)["value"])
                fd[k] = (sides[0] - sides[1]) / (2.0 * e)
            draws.append(fd)
        fd = np.mean(draws, axis=0)
        chain = np.asarray(chain_per_action[t])
        row = {"decision": int(t), "lag": int(h - 1 - t), "chain": chain.tolist(), "finite_difference": fd.tolist(),
               "draws": [d.tolist() for d in draws],
               "cosine_translation": probe.cosine(chain[:3], fd[:3]), "cosine_rotation": probe.cosine(chain[4:], fd[4:]),
               "cosine_all": probe.cosine(chain, fd),
               "ratio_translation": float(np.linalg.norm(chain[:3]) / np.linalg.norm(fd[:3])) if np.linalg.norm(fd[:3]) > 0 else None,
               "ratio_rotation": float(np.linalg.norm(chain[4:]) / np.linalg.norm(fd[4:])) if np.linalg.norm(fd[4:]) > 0 else None}
        if repeats > 1:
            row["draw_cosine_translation"] = probe.cosine(draws[0][:3], draws[1][:3])
            row["draw_cosine_rotation"] = probe.cosine(draws[0][4:], draws[1][4:])
        out.append(row)
        print(f"[trajopt-check] decision {t + 1} of {h} (lag {h - 1 - t}): translation cos={row['cosine_translation']:.3f} "
              f"ratio={row['ratio_translation'] if row['ratio_translation'] is None else round(row['ratio_translation'], 3)} "
              f"rotation cos={row['cosine_rotation']:.3f} ratio={row['ratio_rotation'] if row['ratio_rotation'] is None else round(row['ratio_rotation'], 3)} "
              f"all cos={row['cosine_all']:.3f} chain={np.round(chain, 4).tolist()} fd={np.round(fd, 4).tolist()}"
              + (f" draws cos t={row['draw_cosine_translation']:.2f} r={row['draw_cosine_rotation']:.2f}" if repeats > 1 else ""), flush=True)
    return out


# ------------------------------------------------------------------------ initial trajectories
def initial_actions(kind: str, h: int, env, probe_json: Path | None, step_m: float, step_rad: float) -> np.ndarray:
    """Constant actions: hold, the 5-D axis-proxy direction from a probe record, or its translation only."""
    a = np.zeros((h, 6))
    if kind == "hold":
        return a
    if kind in ("axis5d", "axis3d"):
        if probe_json is None:
            raise ValueError("--init axis5d/axis3d needs --probe-json")
        rec = json.loads(Path(probe_json).read_text())
        snap = rec["snapshots"][0]
        g = snap["gradients"]
        key = str(g["epsilons"][len(g["epsilons"]) // 2])
        t_dir = np.asarray(g["per_epsilon"][key]["mean"]["opening_axis_m"], dtype=np.float64)
        a[:, :3] = t_dir / np.linalg.norm(t_dir) * (step_m / float(env.cfg.max_translation))
        if kind == "axis5d":
            r = snap["rotation_gradients"]
            rkey = str(r["epsilons"][len(r["epsilons"]) // 2])
            r_dir = np.asarray(r["per_epsilon"][rkey]["mean"]["opening_axis_m"], dtype=np.float64)
            a[:, list(r["axes"])] = r_dir / np.linalg.norm(r_dir) * (step_rad / float(env.cfg.max_rotation))
        return np.clip(a, -1.0, 1.0)
    raise ValueError(f"unknown init {kind}")


def expert_rollout(env, snap: dict, h: int) -> tuple[np.ndarray, dict]:
    """The scripted expert's h decisions from the restored state, closed loop; returns the actions
    it took and a rollout record like ``rollout``'s (without systems)."""
    restore_error = probe.restore(env, snap)
    a = np.zeros((h, 6))
    decisions = []
    t0 = time.time()
    for t in range(h):
        a[t] = np.asarray(probe.heuristic(env).actions()[0], dtype=np.float64)
        decisions.append(probe.decision(env, a[t]))
    return a, {"restore_error_m": restore_error, "frames": [], "decisions": decisions,
               "positions": env.positions()[0].astype(np.float64), "seconds": time.time() - t0, "measure": probe.measure(env)}


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--garment", default="tshirt_26")
    p.add_argument("--body", type=int, default=14049)
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--state", default="elbow", help="elbow, passed or stall")
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=12, help="decisions per trajectory")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--init", choices=("hold", "axis5d", "axis3d", "expert"), default="axis5d")
    p.add_argument("--probe-json", default=None, help="Level 1 record of this state, for the axis5d/axis3d initial direction")
    p.add_argument("--init-step-mm", type=float, default=4.0)
    p.add_argument("--init-step-deg", type=float, default=2.5)
    p.add_argument("--step", type=float, default=0.15, help="largest action change per accepted iteration (action units)")
    p.add_argument("--min-step", type=float, default=0.02)
    p.add_argument("--check-first-decision", type=float, default=0.0,
                   help="Compare the chain's rows with central differences of L after h decisions, this many mm "
                        "(and degrees, divided by two), on the first, middle and last decision unless --check-decisions "
                        "says which; 0 skips the check.")
    p.add_argument("--check-decisions", type=int, nargs="*", default=None, help="0-based decisions to difference")
    p.add_argument("--check-repeats", type=int, default=1, help="draw the differences this many times (noise floor)")
    p.add_argument("--check-only", action="store_true", help="stop after the checks on the initial trajectory")
    p.add_argument("--baselines", nargs="*", default=("expert", "hold"))
    p.add_argument("--stall-window", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--constraint-factor", type=float, default=1.0)
    p.add_argument("--final-repeats", type=int, default=3)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = probe.build_env(args.garment, args.body, args.region, out / "work")
    record = {"garment": args.garment, "body": args.body, "state": args.state, "horizon": args.horizon, "build_s": time.time() - t0,
              "dt": float(env.cfg.dt), "action_repeat": int(env.cfg.action_repeat), "max_translation_m": float(env.cfg.max_translation),
              "max_rotation_rad": float(env.cfg.max_rotation), "init": args.init, "step": args.step}
    try:
        feature = adjoint.adjoint_feature(env)
        if feature is None:
            raise RuntimeError("this build has no LinearSystemAdjointFeature")
        snaps, _ = probe.drive_to_snapshots(env, args.stall_window, 0.01, args.max_steps)
        snap = next(s for s in snaps if s["name"] == args.state)
        record["episode_step"] = snap["episode_step"]
        record["state_measure"] = snap["measure"]
        layout = adjoint.cloth_layout(env, args.constraint_factor)
        record["mass_source"] = layout["mass_source"]

        h = args.horizon
        if args.init == "expert":
            actions, _ = expert_rollout(env, snap, h)
        else:
            actions = initial_actions(args.init, h, env, Path(args.probe_json) if args.probe_json else None,
                                      args.init_step_mm * 1e-3, np.deg2rad(args.init_step_deg))

        # Baselines from the same state.
        record["baselines"] = {}
        if "expert" in args.baselines:
            a_exp, roll_exp = expert_rollout(env, snap, h)
            record["baselines"]["expert"] = summarise(env, roll_exp, layout)
            record["baselines"]["expert"]["actions"] = a_exp.tolist()
        if "hold" in args.baselines:
            record["baselines"]["hold"] = summarise(env, rollout(env, snap, np.zeros((h, 6))), layout)
        for name, base in record["baselines"].items():
            print(f"[trajopt] baseline {name}: L={base['objective_m']:.4f} axis={base['axis_m']:.4f} cov={base['coverage_m']:.4f} "
                  f"up={base['upperarm_ratio']:.3f} F={base['mean_net_normal_n']:.0f}N exec={base['executed_m']*1e3:.1f}mm", flush=True)

        # The first trajectory, with its gradient.
        roll = rollout(env, snap, actions, feature)
        current = summarise(env, roll, layout)
        obj = objective(env, roll["positions"], layout)
        grad = chain_gradient(roll["frames"], layout, obj["gradient"], actions, env)
        record["iterations"] = [{"iteration": 0, **current, "accepted": True, "step": None,
                                 "gradient_norm_per_decision": np.linalg.norm(grad["per_action"], axis=1).tolist(),
                                 "reverse_s": grad["reverse_s"]}]
        print(f"[trajopt] it 0: L={current['objective_m']:.4f} axis={current['axis_m']:.4f} cov={current['coverage_m']:.4f} "
              f"up={current['upperarm_ratio']:.3f} F={current['mean_net_normal_n']:.0f}N exec={current['executed_m']*1e3:.1f}mm "
              f"|g|={np.linalg.norm(grad['per_action']):.3g} (rollout {roll['seconds']:.0f}s, reverse {grad['reverse_s']:.0f}s)", flush=True)

        if args.check_first_decision > 0:
            decisions = args.check_decisions if args.check_decisions else sorted({0, h // 2, h - 1})
            record["decision_checks"] = {"epsilon_mm": args.check_first_decision, "epsilon_deg": args.check_first_decision / 2.0,
                                         "repeats": args.check_repeats,
                                         "rows": finite_difference_checks(env, snap, actions, layout, grad["per_action"], decisions,
                                                                          args.check_first_decision * 1e-3,
                                                                          np.deg2rad(args.check_first_decision / 2.0), args.check_repeats)}
            if args.check_only:
                record["final_actions"] = actions.tolist()
                return

        # Projected gradient ascent with backtracking on the measured objective.
        step = args.step
        for it in range(1, args.iterations + 1):
            direction = grad["per_action"] / max(np.abs(grad["per_action"]).max(), 1e-12)
            accepted = False
            trials = []
            while step >= args.min_step:
                trial_actions = np.clip(actions + step * direction, -1.0, 1.0)
                trial_roll = rollout(env, snap, trial_actions, feature)
                trial = summarise(env, trial_roll, layout)
                trials.append({"step": step, "objective_m": trial["objective_m"]})
                if trial["objective_m"] > current["objective_m"]:
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                record["iterations"].append({"iteration": it, **current, "accepted": False, "step": step, "trials": trials})
                print(f"[trajopt] it {it}: no ascent step above {args.min_step} improved L; stopping", flush=True)
                break
            actions, roll, current = trial_actions, trial_roll, trial
            obj = objective(env, roll["positions"], layout)
            grad = chain_gradient(roll["frames"], layout, obj["gradient"], actions, env)
            record["iterations"].append({"iteration": it, **current, "accepted": True, "step": step, "trials": trials,
                                         "gradient_norm_per_decision": np.linalg.norm(grad["per_action"], axis=1).tolist(),
                                         "reverse_s": grad["reverse_s"]})
            print(f"[trajopt] it {it}: L={current['objective_m']:.4f} axis={current['axis_m']:.4f} cov={current['coverage_m']:.4f} "
                  f"up={current['upperarm_ratio']:.3f} F={current['mean_net_normal_n']:.0f}N exec={current['executed_m']*1e3:.1f}mm "
                  f"rot={np.rad2deg(current['executed_rad']):.1f}deg step={step:.3f} trials={len(trials)}", flush=True)
            step = min(step * 1.25, args.step)
        record["final_actions"] = actions.tolist()

        # The final trajectory repeated, for the run-to-run spread.
        finals = [summarise(env, rollout(env, snap, actions), layout) for _ in range(args.final_repeats)]
        record["final_repeats"] = finals
        keys = ("objective_m", "axis_m", "coverage_m", "upperarm_ratio", "mean_net_normal_n", "executed_m")
        record["final_mean"] = {k: float(np.mean([f[k] for f in finals])) for k in keys}
        record["final_std"] = {k: float(np.std([f[k] for f in finals])) for k in keys}
        fm, fs = record["final_mean"], record["final_std"]
        first = record["iterations"][0]
        print(f"[trajopt] {args.garment}/{args.body} {args.state}@{snap['episode_step']} h={h}: "
              f"L {first['objective_m']:.4f} -> {fm['objective_m']:.4f} ± {fs['objective_m']:.4f}, "
              f"coverage {first['upperarm_ratio']:.3f} -> {fm['upperarm_ratio']:.3f} ± {fs['upperarm_ratio']:.3f}, "
              f"force {first['mean_net_normal_n']:.0f} -> {fm['mean_net_normal_n']:.0f} N, "
              f"executed {first['executed_m']*1e3:.0f} -> {fm['executed_m']*1e3:.0f} mm; baselines "
              + ", ".join(f"{k}: L={v['objective_m']:.4f} up={v['upperarm_ratio']:.3f}" for k, v in record["baselines"].items()), flush=True)
    finally:
        record["total_s"] = time.time() - t0
        (out / f"{args.garment}_{args.body}.{args.state}.h{args.horizon}.{args.init}.json").write_text(
            json.dumps(record, indent=1, default=float) + "\n")
        env.close()


if __name__ == "__main__":
    main()
