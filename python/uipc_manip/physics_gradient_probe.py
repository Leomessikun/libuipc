"""Finite-difference physics gradients at stuck dressing states: repeatability, locality, usefulness.

    PYTHONPATH=python python -m uipc_manip.physics_gradient_probe --garment tshirt_392 --bodies 14046 14049 \\
        --out output/uipc_manip/physics_gradient_probe

Level 1 of the physics-gradient line: before any solver change, measure with the black-box
simulator whether "which way should the gripper move" is a well-defined quantity at the states
where dressing fails. The scripted expert drives one cell until the sleeve reaches the elbow and
until its upper-arm progress stalls there; each of those states is dumped (``World.dump`` writes
the frame to disk, ``recover(frame)`` brings it back), and from the restored state the probe
measures three things:

1. **Repeatability** — the same command from the same restored state, several times: how much
   do the outcomes (coverage, arm contact force, contact energy) and the cloth positions differ?
   This is the noise floor every gradient below is read against.
2. **Locality** — central differences of coverage, contact energy and arm force with respect to
   the gripper translation, at several step sizes and repeated: the cosine similarity between
   repeats (noise) and between step sizes (is it a gradient or a step-size artefact?).
3. **Usefulness** — from the restored state, walk several decisions along the coverage gradient,
   along minus the force and contact-energy gradients, along both at once (their normalised sum,
   and release-then-advance), along the expert's own commands, along random directions and
   holding still: does the direction the physics gives raise coverage or lower force better than
   the alternatives over a short horizon? Every outcome carries the translation the environment
   actually executed, since its tether and no-move collision rules can shorten or drop a command.

Everything is written as JSON; nothing is interpreted here. The base command is the hold
(zero translation, zero rotation) unless ``--base expert`` centres the differences on the
expert's own next command.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z")


# ------------------------------------------------------------------------ world
def build_env(garment: str, body: int, region: int, work: Path):
    from uipc_manip import pretrain_wang, train_sac
    from uipc_manip.dressing_env import GenesisIPCDressingEnv
    from uipc_manip.dressing_live import LiveCellFactory
    from uipc_manip.genesis_env import _ensure_genesis

    saved = sys.argv
    sys.argv = ["probe"]
    try:
        _, targs, _ = pretrain_wang.prepare(
            ["teacher", "--region", str(region), "--garments", garment, "--num-envs", "1",
             "--obs-mode", "wang_static_arm", "--no-obs-augment", "--work-dir", str(work)])
    finally:
        sys.argv = saved
    cfg = train_sac.dressing_config(targs)
    _ensure_genesis(cfg.logging_level)
    factory = LiveCellFactory(cfg.live)
    cfg = replace(cfg, cells=((garment, int(body)),), decision_watchdog=False, contact_force_readout=True)
    return GenesisIPCDressingEnv(cfg, num_envs=1, cell_factory=factory)


def contact_energy(env) -> float | None:
    """Total contact energy of the world from the exporter, or None when the backend has none."""
    from uipc import view
    from uipc.core import ContactSystemFeature
    from uipc.geometry import Geometry

    feature = env._world.features().find(ContactSystemFeature)
    if feature is None:
        return None
    total = 0.0
    for prim in feature.contact_primitive_types():
        geom = Geometry()
        feature.contact_energy(prim, geom)
        if geom.instances().size() == 0:
            continue
        slot = None
        for name in ("energy", "E", "energies"):
            slot = geom.instances().find(name)
            if slot is not None:
                break
        if slot is None:
            raise RuntimeError(f"Contact energy geometry of {prim} carries no known energy attribute")
        total += float(np.asarray(view(slot), dtype=np.float64).sum())
    return total


def frame_stats(env) -> dict:
    engine = getattr(env.scene.sim.coupler, "_ipc_engine", None)
    if engine is None:
        return {}
    try:
        stats = engine.frame_stats()
        if not isinstance(stats, dict):
            stats = json.loads(str(stats))
        return {k: stats.get(k) for k in ("frame", "newton_iterations", "linear_solver_iterations", "converged")}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"error": repr(exc)}


def opening_axis(env, positions) -> float:
    """Arc length [m] of the sleeve opening's centroid along finger -> elbow -> shoulder.

    Wang's upper-arm ratio is exactly zero until the opening passes the elbow, so its gradient is
    zero on the whole approach; this reading is continuous through the elbow.
    """
    cell = env.cells[0]
    c = positions[0][cell.opening_idx].mean(axis=0)
    total, best = 0.0, None
    for a, b in ((cell.finger, cell.elbow), (cell.elbow, cell.shoulder)):
        d = b - a
        length = float(np.linalg.norm(d))
        t = float(np.clip((c - a) @ d / (length * length), 0.0, 1.0))
        dist = float(np.linalg.norm(c - (a + t * d)))
        if best is None or dist < best[0]:
            best = (dist, total + t * length)
        total += length
    return best[1]


def arm_clearance(env) -> float:
    """Distance [m] from the gripper anchor to the nearest arm-shell point; the environment drops
    any move that would bring it under ``no_move_collision_threshold`` (12 mm)."""
    cell = env.cells[0]
    return float(np.min(np.linalg.norm(cell.arm_points - env._anchor[0][None, :], axis=1)))


def upperarm_axis(env, positions) -> float:
    """Signed projection [m] of the opening centroid onto the elbow -> shoulder axis, unclipped.

    ``opening_axis`` clips its projection onto both segments, which flattens it at the elbow corner;
    this reading is linear there (negative below the elbow).
    """
    cell = env.cells[0]
    c = positions[0][cell.opening_idx].mean(axis=0)
    d = cell.shoulder - cell.elbow
    return float((c - cell.elbow) @ d / float(np.linalg.norm(d)))


def measure(env) -> dict:
    positions = env.positions()
    pr = env._progress(positions)[0]
    force = env._arm_force_summaries()[0]
    out = {
        "upperarm_ratio": float(pr.upperarm_ratio), "forearm_ratio": float(pr.forearm_ratio), "reward": float(pr.reward),
        "opening_axis_m": opening_axis(env, positions),
        "net_normal_n": float(force.get("net_normal_n", np.nan)), "summed_normal_n": float(force.get("summed_normal_n", np.nan)),
        "peak_vertex_normal_n": float(force.get("peak_vertex_normal_n", np.nan)),
        "contact_energy": contact_energy(env),
        "tracking_error": float(env._tracking_error(0, positions)),
        "anchor": env._anchor[0].tolist(),
        "arm_clearance_m": arm_clearance(env),
        "upperarm_axis_m": upperarm_axis(env, positions),
    }
    out.update({f"solver_{k}": v for k, v in frame_stats(env).items()})
    return out


# ------------------------------------------------------------------------ snapshots
def heuristic(env):
    if getattr(env, "_heuristic", None) is None:
        env.scripted_actions()
    return env._heuristic


def take_snapshot(env, name: str, step: int) -> dict:
    if not env._world.dump():
        raise RuntimeError("World.dump failed")
    h = heuristic(env)
    return {
        "name": name, "frame": int(env._world.frame()), "episode_step": int(step),
        "anchor": env._anchor.copy(), "offsets": [o.copy() for o in env._offsets],
        "last_progress": list(env._last_progress), "privileged": env._privileged.copy(),
        "heuristic": {k: getattr(h, k).copy() for k in ("stage", "_steps", "_align_steps", "_best_upper")},
        "positions": [p.copy() for p in env.positions()],
        "rng_states": [copy.deepcopy(r.bit_generator.state) for r in env.rngs],
        "measure": measure(env),
    }


def restore(env, snap: dict) -> float:
    """Bring the world and the environment's bookkeeping back; returns the largest position error [m]."""
    if not env._world.recover(snap["frame"]):
        raise RuntimeError(f"World.recover({snap['frame']}) failed")
    env._world.retrieve()
    if int(env._world.frame()) != int(snap["frame"]):
        raise RuntimeError(f"Recovered frame {env._world.frame()} is not {snap['frame']}")
    env._anchor = snap["anchor"].copy()
    env._offsets = [o.copy() for o in snap["offsets"]]
    env._update_targets()
    env._last_progress = list(snap["last_progress"])
    env._privileged = snap["privileged"].copy()
    env._episode_step = int(snap["episode_step"])
    for rng, state in zip(env.rngs, snap.get("rng_states", []), strict=False):
        rng.bit_generator.state = copy.deepcopy(state)
    env._decision_times.clear()
    h = heuristic(env)
    for k, v in snap["heuristic"].items():
        getattr(h, k)[:] = v
    restored = env.positions()
    return float(max(np.abs(a - b).max() for a, b in zip(restored, snap["positions"], strict=True)))


def rotation_angle(before: np.ndarray, after: np.ndarray) -> float:
    """Angle [rad] of the rigid rotation taking the held offsets ``before`` to ``after`` (Kabsch)."""
    before, after = np.asarray(before, dtype=np.float64), np.asarray(after, dtype=np.float64)
    if before.shape[0] < 3:
        a, b = before[0], after[0]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))) if na > 0 and nb > 0 else 0.0
    u, _, vt = np.linalg.svd(before.T @ after)
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


def commanded_rotation(env, action: np.ndarray) -> np.ndarray:
    """The rotation vector [rad] the environment derives from an action (one axis may be clipped)."""
    rot = np.asarray(action[3:6], dtype=np.float64) * float(env.cfg.max_rotation)
    if getattr(env.cfg, "clip_rotation_to_yz", False):
        rot[0] = 0.0
    return rot


def decision(env, action: np.ndarray) -> dict:
    """One decision; the outcome carries the commanded and the executed gripper translation [m] and
    rotation [rad], the latter after the environment's tether and no-move collision rules."""
    action = np.asarray(action, dtype=np.float32)
    before, offsets_before = env._anchor[0].copy(), np.asarray(env._offsets[0]).copy()
    _, reward, done, infos = env.step(action[None])
    if infos[0].get("sim_error"):
        raise RuntimeError(f"simulator error: {infos[0].get('error')}")
    out = measure(env)
    out["reward_step"] = float(reward[0])
    out["commanded_m"] = float(np.linalg.norm(action[:3]) * float(env.cfg.max_translation))
    out["executed_m"] = float(np.linalg.norm(env._anchor[0] - before))
    out["commanded_rad"] = float(np.linalg.norm(commanded_rotation(env, action)))
    out["executed_rad"] = rotation_angle(offsets_before, np.asarray(env._offsets[0]))
    for key in ("grasp_valid", "valid_grasp_success", "collision_rejected_substeps", "tether_rejected_substeps"):
        if key in infos[0]:
            out[key] = infos[0][key]
    return out


# ------------------------------------------------------------------------ the three questions
def repeatability(env, snap: dict, base: np.ndarray, repeats: int) -> dict:
    rows, positions = [], []
    for _ in range(repeats):
        err = restore(env, snap)
        rows.append({"restore_error_m": err, **decision(env, base)})
        positions.append(env.positions()[0])
    keys = ("upperarm_ratio", "opening_axis_m", "net_normal_n", "summed_normal_n", "contact_energy")
    spread = {k: float(np.std([r[k] for r in rows])) if all(r[k] is not None for r in rows) else None for k in keys}
    pos = np.stack(positions)
    return {"rows": rows, "spread": spread,
            "position_spread_max_m": float(np.abs(pos - pos.mean(0)).max()),
            "restore_error_max_m": float(max(r["restore_error_m"] for r in rows))}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


TRANSLATION_AXES = (0, 1, 2)
ROTATION_AXES = (4, 5)  # the environment clips the rotation about x to zero


def gradients(env, snap: dict, base: np.ndarray, epsilons: list[float], repeats: int,
              axes: tuple[int, ...] = TRANSLATION_AXES) -> dict:
    """Central differences of the outcome quantities with respect to the gripper command on ``axes``:
    translation axes (0-2) in metres, rotation axes (3-5) in radians."""
    rotation = axes[0] >= 3
    cap = float(env.cfg.max_rotation) if rotation else float(env.cfg.max_translation)
    exec_key = "executed_rad" if rotation else "executed_m"
    n = len(axes)
    keys = ("upperarm_ratio", "opening_axis_m", "upperarm_axis_m", "net_normal_n", "summed_normal_n", "contact_energy", "reward")
    per_eps = {}
    for eps in epsilons:
        step = eps / cap
        grads = []
        for _ in range(repeats):
            plus, minus = {}, {}
            for i, axis in enumerate(axes):
                for sign, store in ((+1.0, plus), (-1.0, minus)):
                    action = base.copy()
                    action[axis] += sign * step
                    restore(env, snap)
                    store[i] = decision(env, action)
            grad = {}
            for k in keys:
                if any(plus[a][k] is None or minus[a][k] is None for a in range(n)):
                    grad[k] = None
                    continue
                grad[k] = [(plus[a][k] - minus[a][k]) / (2.0 * eps) for a in range(n)]  # per commanded unit
            grad["outcomes"] = {k: {"plus": [plus[a][k] for a in range(n)], "minus": [minus[a][k] for a in range(n)]} for k in keys}
            # The environment may drop or shorten a move (tether, no-move collision); the difference per
            # executed unit is the derivative of the physics, the one above of the environment.
            executed = {"plus": [plus[a][exec_key] for a in range(n)], "minus": [minus[a][exec_key] for a in range(n)]}
            grad["executed"] = executed
            grad["executed_m" if not rotation else "executed_rad"] = executed
            grad["per_executed_unit"] = {
                k: (None if grad[k] is None else [
                    ((plus[a][k] - minus[a][k]) / (executed["plus"][a] + executed["minus"][a]))
                    if executed["plus"][a] + executed["minus"][a] > 0.0 else None for a in range(n)])
                for k in keys}
            if not rotation:
                grad["per_executed_metre"] = grad["per_executed_unit"]
            grads.append(grad)
        per_eps[str(eps)] = {
            "gradients": grads,
            "mean": {k: (np.mean([g[k] for g in grads], axis=0).tolist() if grads[0][k] is not None else None) for k in keys},
            "mean_executed": {s: np.mean([g["executed"][s] for g in grads], axis=0).tolist() for s in ("plus", "minus")},
            "repeat_cosine": {k: ([cosine(grads[i][k], grads[j][k]) for i in range(len(grads)) for j in range(i + 1, len(grads))]
                                  if grads[0][k] is not None else None) for k in keys},
        }
        if not rotation:
            per_eps[str(eps)]["mean_executed_m"] = per_eps[str(eps)]["mean_executed"]
    eps_keys = [str(e) for e in epsilons]
    locality = {k: [[cosine(per_eps[a]["mean"][k], per_eps[b]["mean"][k]) for b in eps_keys] for a in eps_keys]
                for k in keys if per_eps[eps_keys[0]]["mean"][k] is not None}
    out = {"axes": list(axes), "unit": "rad" if rotation else "m", "epsilons": list(epsilons), "per_epsilon": per_eps,
           "locality_cosine": locality, "base_action": base.tolist()}
    if not rotation:
        out["epsilons_m"] = list(epsilons)
    return out


def walk(env, snap: dict, direction, steps: int, repeats: int = 1) -> dict:
    """``direction(step) -> action``; returns the outcome after every decision of the first run and,
    with ``repeats > 1``, the end-of-walk summaries of every run (for states whose decision is noisy)."""
    runs = []
    for _ in range(repeats):
        err = restore(env, snap)
        rows = []
        for s in range(steps):
            rows.append(decision(env, np.clip(direction(s), -1.0, 1.0)))
        first, last = snap["measure"], rows[-1]
        runs.append({"restore_error_m": err, "rows": rows,
                     "delta_upperarm": last["upperarm_ratio"] - first["upperarm_ratio"],
                     "delta_opening_axis_m": last["opening_axis_m"] - first["opening_axis_m"],
                     "delta_upperarm_axis_m": last["upperarm_axis_m"] - first["upperarm_axis_m"],
                     "max_opening_axis_m": max(r["opening_axis_m"] for r in rows),
                     "max_upperarm": max(r["upperarm_ratio"] for r in rows),
                     "delta_net_normal_n": last["net_normal_n"] - first["net_normal_n"],
                     "mean_net_normal_n": float(np.mean([r["net_normal_n"] for r in rows])),
                     "executed_m": float(sum(r["executed_m"] for r in rows)),
                     "commanded_m": float(sum(r["commanded_m"] for r in rows)),
                     "executed_rad": float(sum(r.get("executed_rad", 0.0) for r in rows)),
                     "commanded_rad": float(sum(r.get("commanded_rad", 0.0) for r in rows)),
                     "final_tracking_error": last["tracking_error"], "final_arm_clearance_m": last["arm_clearance_m"]})
    out = dict(runs[0])
    if repeats > 1:
        summary_keys = ("delta_upperarm", "delta_opening_axis_m", "delta_upperarm_axis_m", "delta_net_normal_n", "mean_net_normal_n", "executed_m")
        out["repeats"] = [{k: r[k] for k in summary_keys} for r in runs]
        out["repeat_mean"] = {k: float(np.mean([r[k] for r in runs])) for k in summary_keys}
        out["repeat_std"] = {k: float(np.std([r[k] for r in runs])) for k in summary_keys}
    return out


WALKS = ("coverage_gradient", "axis_gradient", "minus_force_gradient", "minus_energy_gradient", "combined_gradient",
         "release_then_advance", "rotation_gradient", "full_gradient", "axis_gradient_expert_rotation", "expert", "hold",
         "random")


def usefulness(env, snap: dict, grad: dict, steps: int, magnitude_m: float, seed: int, repeats: int = 1,
               walks: tuple[str, ...] = WALKS, rot_grad: dict | None = None, rot_magnitude_rad: float = 0.0) -> dict:
    max_t = float(env.cfg.max_translation)
    scale = magnitude_m / max_t
    eps_key = str(grad["epsilons"][len(grad["epsilons"]) // 2])
    g_cov = np.asarray(grad["per_epsilon"][eps_key]["mean"]["upperarm_ratio"], dtype=np.float64)
    g_force = np.asarray(grad["per_epsilon"][eps_key]["mean"]["net_normal_n"], dtype=np.float64)
    g_energy = grad["per_epsilon"][eps_key]["mean"]["contact_energy"]
    rng = np.random.default_rng(seed)

    def fixed(v):
        a = np.zeros(6)
        a[:3] = _unit(v) * scale
        return lambda s: a

    def fixed6(t, r, rot_axes):
        """Translation direction ``t`` at the walk's magnitude plus rotation direction ``r`` on ``rot_axes``
        at ``rot_magnitude_rad`` per decision; either may be zero."""
        a = np.zeros(6)
        if t is not None and np.linalg.norm(t) > 0:
            a[:3] = _unit(np.asarray(t, dtype=np.float64)) * scale
        if r is not None and np.linalg.norm(r) > 0:
            a[list(rot_axes)] = _unit(np.asarray(r, dtype=np.float64)) * (rot_magnitude_rad / float(env.cfg.max_rotation))
        return lambda s: a

    def with_expert_rotation(t):
        a_t = fixed(t)(0)

        def direction(s):
            a = a_t.copy()
            a[3:] = np.asarray(heuristic(env).actions()[0], dtype=np.float64)[3:]
            return a
        return direction

    def expert(s):
        return heuristic(env).actions()[0]

    def sequenced(first, then, switch):
        a_first, a_then = fixed(first)(0), fixed(then)(0)
        return lambda s: a_first if s < switch else a_then

    out = {"gradient_epsilon_m": float(eps_key), "step_magnitude_m": magnitude_m, "steps": steps, "walk_repeats": repeats,
           "release_steps": steps // 3, "random_seed": seed, "walks": {}}
    g_axis = grad["per_epsilon"][eps_key]["mean"]["opening_axis_m"]
    if "coverage_gradient" in walks:
        out["walks"]["coverage_gradient"] = walk(env, snap, fixed(g_cov), steps, repeats)
    if "axis_gradient" in walks and g_axis is not None:
        out["walks"]["axis_gradient"] = walk(env, snap, fixed(np.asarray(g_axis)), steps, repeats)
    if "minus_force_gradient" in walks:
        out["walks"]["minus_force_gradient"] = walk(env, snap, fixed(-g_force), steps, repeats)
    if "minus_energy_gradient" in walks and g_energy is not None:
        out["walks"]["minus_energy_gradient"] = walk(env, snap, fixed(-np.asarray(g_energy)), steps, repeats)
    # Two ways of using both one-step gradients at once: their normalised sum, and release first
    # (down the force gradient), then advance (up the coverage gradient).
    if "combined_gradient" in walks and np.linalg.norm(g_cov) > 0 and np.linalg.norm(g_force) > 0:
        out["walks"]["combined_gradient"] = walk(env, snap, fixed(_unit(g_cov) + _unit(-g_force)), steps, repeats)
    if "release_then_advance" in walks and np.linalg.norm(g_cov) > 0 and np.linalg.norm(g_force) > 0:
        out["walks"]["release_then_advance"] = walk(env, snap, sequenced(-g_force, g_cov, steps // 3), steps, repeats)
    # Rotation: the coverage gradient where it exists, else the axis reading's; same choice for translation.
    if rot_grad is not None:
        r_key = str(rot_grad["epsilons"][len(rot_grad["epsilons"]) // 2])
        r_cov = np.asarray(rot_grad["per_epsilon"][r_key]["mean"]["upperarm_ratio"], dtype=np.float64)
        r_axis = rot_grad["per_epsilon"][r_key]["mean"]["opening_axis_m"]
        r_axis = None if r_axis is None else np.asarray(r_axis, dtype=np.float64)
        r_dir, r_from = (r_cov, "upperarm_ratio") if np.linalg.norm(r_cov) > 0 else (r_axis, "opening_axis_m")
        t_dir, t_from = (g_cov, "upperarm_ratio") if np.linalg.norm(g_cov) > 0 else (
            (np.asarray(g_axis, dtype=np.float64), "opening_axis_m") if g_axis is not None else (None, None))
        rot_axes = tuple(rot_grad["axes"])
        out["rotation"] = {"epsilon_rad": float(r_key), "magnitude_rad": rot_magnitude_rad, "axes": list(rot_axes),
                           "rotation_direction_from": r_from, "translation_direction_from": t_from}
        if "rotation_gradient" in walks and r_dir is not None and np.linalg.norm(r_dir) > 0:
            out["walks"]["rotation_gradient"] = walk(env, snap, fixed6(None, r_dir, rot_axes), steps, repeats)
        if "full_gradient" in walks and r_dir is not None and np.linalg.norm(r_dir) > 0 and t_dir is not None:
            out["walks"]["full_gradient"] = walk(env, snap, fixed6(t_dir, r_dir, rot_axes), steps, repeats)
    if "axis_gradient_expert_rotation" in walks and g_axis is not None:
        out["walks"]["axis_gradient_expert_rotation"] = walk(env, snap, with_expert_rotation(np.asarray(g_axis)), steps, repeats)
    if "expert" in walks:
        out["walks"]["expert"] = walk(env, snap, expert, steps, repeats)
    if "hold" in walks:
        out["walks"]["hold"] = walk(env, snap, lambda s: np.zeros(6), steps, repeats)
    if "random" in walks:
        for i in range(3):
            out["walks"][f"random_{i}"] = walk(env, snap, fixed(rng.standard_normal(3)), steps, repeats)
    return out


# ------------------------------------------------------------------------ driving to the elbow
def drive_to_snapshots(env, stall_window: int, stall_delta: float, max_steps: int) -> tuple[list[dict], list[dict]]:
    """Run the expert; dump the state when the sleeve reaches the elbow and when progress stalls there."""
    env.reset([0])
    h = heuristic(env)
    snaps, trace = [], []
    elbow_step, passed, best, best_step = None, False, 0.0, 0
    for step in range(max_steps):
        m = measure(env)
        trace.append({"step": step, "stage": h.stage_names()[0],
                      **{k: m[k] for k in ("upperarm_ratio", "forearm_ratio", "opening_axis_m", "net_normal_n", "contact_energy")}})
        if elbow_step is None and m["forearm_ratio"] >= 0.95:
            # The sleeve has reached the elbow; the stall clock starts here, not at the episode start.
            elbow_step, best_step = step, step
            snaps.append(take_snapshot(env, "elbow", step))
        if elbow_step is not None:
            if not passed and m["upperarm_ratio"] >= 0.1:
                passed = True
                snaps.append(take_snapshot(env, "passed", step))
            if m["upperarm_ratio"] > best + stall_delta:
                best, best_step = m["upperarm_ratio"], step
            elif step - best_step >= stall_window:
                snaps.append(take_snapshot(env, "stall", step))
                break
        if h.stage_names()[0] == "done" and elbow_step is not None:
            snaps.append(take_snapshot(env, "done", step))
            break
        decision(env, h.actions()[0])
    return snaps, trace


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--garment", default="tshirt_392")
    p.add_argument("--bodies", type=int, nargs="+", default=[14046])
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--out", required=True)
    p.add_argument("--epsilons-mm", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    p.add_argument("--repeats", type=int, default=3, help="Gradient repeats per step size.")
    p.add_argument("--rot-epsilons-deg", type=float, nargs="+", default=[0.5, 1.0, 2.5],
                   help="Rotation step sizes for the differences on the two live rotation axes; empty list disables.")
    p.add_argument("--walk-rot-deg", type=float, default=2.5, help="Rotation per decision along a walked rotation direction.")
    p.add_argument("--repeatability", type=int, default=5, help="Identical decisions from the restored state.")
    p.add_argument("--walk-steps", type=int, default=12)
    p.add_argument("--walk-mm", type=float, default=4.0, help="Translation per decision along a walked direction.")
    p.add_argument("--walk-repeats", type=int, default=1, help="Runs per walked direction (for states whose decision is noisy).")
    p.add_argument("--walks", nargs="+", default=list(WALKS), choices=WALKS, help="Which walks to run.")
    p.add_argument("--states", nargs="+", default=None, help="Probe only these snapshots (elbow, passed, stall, done).")
    p.add_argument("--base", choices=("hold", "expert"), default="hold")
    p.add_argument("--stall-window", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    epsilons = [e * 1.0e-3 for e in args.epsilons_mm]
    for body in args.bodies:
        t0 = time.time()
        env = build_env(args.garment, body, args.region, out / "work")
        record = {"garment": args.garment, "body": int(body), "max_translation_m": float(env.cfg.max_translation),
                  "dt": float(env.cfg.dt), "action_repeat": int(env.cfg.action_repeat), "build_s": time.time() - t0, "snapshots": []}
        try:
            snaps, trace = drive_to_snapshots(env, args.stall_window, 0.01, args.max_steps)
            record["trace"] = trace
            for snap in snaps:
                if args.states and snap["name"] not in args.states:
                    continue
                t1 = time.time()
                base = np.zeros(6)
                if args.base == "expert":
                    restore(env, snap)
                    base = np.asarray(heuristic(env).actions()[0], dtype=np.float64)
                entry = {"name": snap["name"], "frame": snap["frame"], "episode_step": snap["episode_step"], "state": snap["measure"]}
                entry["repeatability"] = repeatability(env, snap, base, args.repeatability)
                entry["gradients"] = gradients(env, snap, base, epsilons, args.repeats)
                rot_grad = None
                if args.rot_epsilons_deg:
                    rot_grad = gradients(env, snap, base, [np.deg2rad(d) for d in args.rot_epsilons_deg], args.repeats, ROTATION_AXES)
                    entry["rotation_gradients"] = rot_grad
                # The random directions are drawn per state; the first three cells drew the same three at every state.
                entry["usefulness"] = usefulness(env, snap, entry["gradients"], args.walk_steps, args.walk_mm * 1.0e-3,
                                                 [args.seed, int(body), int(snap["episode_step"])], args.walk_repeats, tuple(args.walks),
                                                 rot_grad, float(np.deg2rad(args.walk_rot_deg)))
                entry["seconds"] = time.time() - t1
                record["snapshots"].append(entry)
                summary = {k: entry["gradients"]["per_epsilon"][str(epsilons[-1])]["mean"][k] for k in ("upperarm_ratio", "opening_axis_m", "net_normal_n")}
                print(f"[probe] {args.garment}/{body} {snap['name']}@{snap['episode_step']}: up={snap['measure']['upperarm_ratio']:.3f} "
                      f"axis={snap['measure']['opening_axis_m']:.3f} restore_err={entry['repeatability']['restore_error_max_m']:.2e} "
                      f"spread_axis={entry['repeatability']['spread']['opening_axis_m']:.2e} "
                      f"grad_up/m={np.round(summary['upperarm_ratio'], 2).tolist()} grad_axis={np.round(summary['opening_axis_m'], 3).tolist()} "
                      f"grad_force/m={np.round(summary['net_normal_n'], 1).tolist()} "
                      + (f"rot_grad_up/rad={np.round(rot_grad['per_epsilon'][str(rot_grad['epsilons'][-1])]['mean']['upperarm_ratio'], 2).tolist()} "
                         f"rot_grad_axis/rad={np.round(rot_grad['per_epsilon'][str(rot_grad['epsilons'][-1])]['mean']['opening_axis_m'], 3).tolist()} "
                         if rot_grad is not None else "") +
                      f"walks_axis_mm={ {k: round(v['delta_opening_axis_m'] * 1e3, 1) for k, v in entry['usefulness']['walks'].items()} } "
                      f"walks_up={ {k: round(v['delta_upperarm'], 3) for k, v in entry['usefulness']['walks'].items()} } "
                      f"walks_travel_mm={ {k: round(v['executed_m'] * 1e3, 1) for k, v in entry['usefulness']['walks'].items()} } ({entry['seconds']:.0f}s)", flush=True)
        finally:
            suffix = "" if not args.states else "." + "_".join(args.states)
            (out / f"{args.garment}_{body}{suffix}.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
            env.close()


if __name__ == "__main__":
    main()
