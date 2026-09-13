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
   along minus the contact-energy gradient, along the expert's own commands, along random
   directions and holding still: does the direction the physics gives raise coverage or lower
   force better than the alternatives over a short horizon?

Everything is written as JSON; nothing is interpreted here. The base command is the hold
(zero translation, zero rotation) unless ``--base expert`` centres the differences on the
expert's own next command.
"""

from __future__ import annotations

import argparse
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


def measure(env) -> dict:
    positions = env.positions()
    pr = env._progress(positions)[0]
    force = env._arm_force_summaries()[0]
    out = {
        "upperarm_ratio": float(pr.upperarm_ratio), "forearm_ratio": float(pr.forearm_ratio), "reward": float(pr.reward),
        "net_normal_n": float(force.get("net_normal_n", np.nan)), "summed_normal_n": float(force.get("summed_normal_n", np.nan)),
        "peak_vertex_normal_n": float(force.get("peak_vertex_normal_n", np.nan)),
        "contact_energy": contact_energy(env),
        "tracking_error": float(env._tracking_error(0, positions)),
        "anchor": env._anchor[0].tolist(),
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
    env._decision_times.clear()
    h = heuristic(env)
    for k, v in snap["heuristic"].items():
        getattr(h, k)[:] = v
    restored = env.positions()
    return float(max(np.abs(a - b).max() for a, b in zip(restored, snap["positions"], strict=True)))


def decision(env, action: np.ndarray) -> dict:
    _, reward, done, infos = env.step(np.asarray(action, dtype=np.float32)[None])
    if infos[0].get("sim_error"):
        raise RuntimeError(f"simulator error: {infos[0].get('error')}")
    out = measure(env)
    out["reward_step"] = float(reward[0])
    return out


# ------------------------------------------------------------------------ the three questions
def repeatability(env, snap: dict, base: np.ndarray, repeats: int) -> dict:
    rows, positions = [], []
    for _ in range(repeats):
        err = restore(env, snap)
        rows.append({"restore_error_m": err, **decision(env, base)})
        positions.append(env.positions()[0])
    keys = ("upperarm_ratio", "net_normal_n", "summed_normal_n", "contact_energy")
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


def gradients(env, snap: dict, base: np.ndarray, epsilons_m: list[float], repeats: int) -> dict:
    """Central differences of the outcome quantities with respect to the gripper translation."""
    max_t = float(env.cfg.max_translation)
    keys = ("upperarm_ratio", "net_normal_n", "summed_normal_n", "contact_energy", "reward")
    per_eps = {}
    for eps in epsilons_m:
        step = eps / max_t
        grads = []
        for _ in range(repeats):
            plus, minus = {}, {}
            for axis in range(3):
                for sign, store in ((+1.0, plus), (-1.0, minus)):
                    action = base.copy()
                    action[axis] += sign * step
                    restore(env, snap)
                    store[axis] = decision(env, action)
            grad = {}
            for k in keys:
                if any(plus[a][k] is None or minus[a][k] is None for a in range(3)):
                    grad[k] = None
                    continue
                grad[k] = [(plus[a][k] - minus[a][k]) / (2.0 * eps) for a in range(3)]  # per metre
            grads.append(grad)
        per_eps[str(eps)] = {
            "gradients": grads,
            "mean": {k: (np.mean([g[k] for g in grads], axis=0).tolist() if grads[0][k] is not None else None) for k in keys},
            "repeat_cosine": {k: ([cosine(grads[i][k], grads[j][k]) for i in range(len(grads)) for j in range(i + 1, len(grads))]
                                  if grads[0][k] is not None else None) for k in keys},
        }
    eps_keys = [str(e) for e in epsilons_m]
    locality = {k: [[cosine(per_eps[a]["mean"][k], per_eps[b]["mean"][k]) for b in eps_keys] for a in eps_keys]
                for k in keys if per_eps[eps_keys[0]]["mean"][k] is not None}
    return {"epsilons_m": epsilons_m, "per_epsilon": per_eps, "locality_cosine": locality, "base_action": base.tolist()}


def walk(env, snap: dict, direction, steps: int) -> dict:
    """``direction(step) -> action``; returns the outcome after every decision."""
    err = restore(env, snap)
    rows = []
    for s in range(steps):
        rows.append(decision(env, np.clip(direction(s), -1.0, 1.0)))
    first, last = snap["measure"], rows[-1]
    return {"restore_error_m": err, "rows": rows,
            "delta_upperarm": last["upperarm_ratio"] - first["upperarm_ratio"],
            "max_upperarm": max(r["upperarm_ratio"] for r in rows),
            "delta_net_normal_n": last["net_normal_n"] - first["net_normal_n"],
            "mean_net_normal_n": float(np.mean([r["net_normal_n"] for r in rows]))}


def usefulness(env, snap: dict, grad: dict, steps: int, magnitude_m: float, seed: int) -> dict:
    max_t = float(env.cfg.max_translation)
    scale = magnitude_m / max_t
    eps_key = str(grad["epsilons_m"][len(grad["epsilons_m"]) // 2])
    g_cov = np.asarray(grad["per_epsilon"][eps_key]["mean"]["upperarm_ratio"], dtype=np.float64)
    g_force = np.asarray(grad["per_epsilon"][eps_key]["mean"]["net_normal_n"], dtype=np.float64)
    g_energy = grad["per_epsilon"][eps_key]["mean"]["contact_energy"]
    rng = np.random.default_rng(seed)

    def fixed(v):
        a = np.zeros(6)
        a[:3] = _unit(v) * scale
        return lambda s: a

    def expert(s):
        return heuristic(env).actions()[0]

    out = {"gradient_epsilon_m": float(eps_key), "step_magnitude_m": magnitude_m, "steps": steps, "walks": {}}
    out["walks"]["coverage_gradient"] = walk(env, snap, fixed(g_cov), steps)
    out["walks"]["minus_force_gradient"] = walk(env, snap, fixed(-g_force), steps)
    if g_energy is not None:
        out["walks"]["minus_energy_gradient"] = walk(env, snap, fixed(-np.asarray(g_energy)), steps)
    out["walks"]["expert"] = walk(env, snap, expert, steps)
    out["walks"]["hold"] = walk(env, snap, lambda s: np.zeros(6), steps)
    for i in range(3):
        out["walks"][f"random_{i}"] = walk(env, snap, fixed(rng.standard_normal(3)), steps)
    return out


# ------------------------------------------------------------------------ driving to the elbow
def drive_to_snapshots(env, stall_window: int, stall_delta: float, max_steps: int) -> tuple[list[dict], list[dict]]:
    """Run the expert; dump the state when the sleeve reaches the elbow and when progress stalls there."""
    env.reset([0])
    h = heuristic(env)
    snaps, trace = [], []
    elbow_step, best, best_step = None, 0.0, 0
    for step in range(max_steps):
        m = measure(env)
        trace.append({"step": step, "stage": h.stage_names()[0], **{k: m[k] for k in ("upperarm_ratio", "forearm_ratio", "net_normal_n", "contact_energy")}})
        if elbow_step is None and m["forearm_ratio"] >= 0.95:
            elbow_step = step
            snaps.append(take_snapshot(env, "elbow", step))
        if elbow_step is not None:
            if m["upperarm_ratio"] > best + stall_delta:
                best, best_step = m["upperarm_ratio"], step
            elif step - best_step >= stall_window and len(snaps) < 2:
                snaps.append(take_snapshot(env, "stall", step))
                break
        if h.stage_names()[0] == "done" and len(snaps) < 2 and elbow_step is not None:
            snaps.append(take_snapshot(env, "done", step))
            break
        decision(env, h.actions()[0])
    return snaps, trace


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--garment", default="tshirt_392")
    p.add_argument("--bodies", type=int, nargs="+", default=[14046])
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--out", required=True)
    p.add_argument("--epsilons-mm", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    p.add_argument("--repeats", type=int, default=3, help="Gradient repeats per step size.")
    p.add_argument("--repeatability", type=int, default=5, help="Identical decisions from the restored state.")
    p.add_argument("--walk-steps", type=int, default=8)
    p.add_argument("--walk-mm", type=float, default=4.0, help="Translation per decision along a walked direction.")
    p.add_argument("--base", choices=("hold", "expert"), default="hold")
    p.add_argument("--stall-window", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
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
                t1 = time.time()
                base = np.zeros(6)
                if args.base == "expert":
                    restore(env, snap)
                    base = np.asarray(heuristic(env).actions()[0], dtype=np.float64)
                entry = {"name": snap["name"], "frame": snap["frame"], "episode_step": snap["episode_step"], "state": snap["measure"]}
                entry["repeatability"] = repeatability(env, snap, base, args.repeatability)
                entry["gradients"] = gradients(env, snap, base, epsilons, args.repeats)
                entry["usefulness"] = usefulness(env, snap, entry["gradients"], args.walk_steps, args.walk_mm * 1.0e-3, args.seed)
                entry["seconds"] = time.time() - t1
                record["snapshots"].append(entry)
                summary = {k: entry["gradients"]["per_epsilon"][str(epsilons[1])]["mean"][k] for k in ("upperarm_ratio", "net_normal_n")}
                print(f"[probe] {args.garment}/{body} {snap['name']}@{snap['episode_step']}: up={snap['measure']['upperarm_ratio']:.3f} "
                      f"restore_err={entry['repeatability']['restore_error_max_m']:.2e} spread_up={entry['repeatability']['spread']['upperarm_ratio']:.2e} "
                      f"grad_up/m={np.round(summary['upperarm_ratio'], 2).tolist()} grad_force/m={np.round(summary['net_normal_n'], 1).tolist()} "
                      f"walks={ {k: round(v['delta_upperarm'], 3) for k, v in entry['usefulness']['walks'].items()} } ({entry['seconds']:.0f}s)", flush=True)
        finally:
            (out / f"{args.garment}_{body}.json").write_text(json.dumps(record, indent=1, default=float) + "\n")
            env.close()


if __name__ == "__main__":
    main()
