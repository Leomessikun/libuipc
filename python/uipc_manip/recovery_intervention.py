"""Turn a recovery decision into full-episode dressing, or show that it does not.

A local comparison can rank recovery macros correctly and still leave the episode
where it was: the sleeve is freed, the policy walks back into the same stall, and the
final coverage is unchanged. This runs complete episodes in which a stall detector
that reads only deployment-time quantities hands control to a macro for a few
decisions and then hands it back, against the same policy with the detector disabled.

The detector is the garment's response to the command: the opening's motion per metre
of commanded translation, averaged over a window. It uses the same observation the
actor receives, so an arm that uses it is deployable.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from . import decision_features as df


class StallDetector:
    """Fires when the garment stops responding to the commanded motion.

    ``response`` is metres of garment-centroid motion per metre of commanded tool
    translation, averaged over ``window`` decisions. Commands below ``min_command_m``
    do not count: standing still is not a stall.
    """

    def __init__(self, window: int = 8, threshold: float = 0.15, min_command_m: float = 0.002,
                 cooldown: int = 20):
        if window < 2 or threshold <= 0 or cooldown < 0:
            raise ValueError("Need a window of at least two decisions, a positive threshold and a cooldown")
        self.window, self.threshold, self.min_command_m, self.cooldown = window, threshold, min_command_m, cooldown
        self.moved: deque[float] = deque(maxlen=window)
        self.commanded: deque[float] = deque(maxlen=window)
        self._previous: np.ndarray | None = None
        self._blocked = 0

    def reset(self) -> None:
        self.moved.clear()
        self.commanded.clear()
        self._previous = None
        self._blocked = 0

    def update(self, centroid: np.ndarray, commanded_m: float) -> float | None:
        """Record one decision; returns the current response, or None while filling up."""
        centroid = np.asarray(centroid, dtype=np.float64)
        if self._previous is not None:
            self.moved.append(float(np.linalg.norm(centroid - self._previous)))
            self.commanded.append(float(commanded_m))
        self._previous = centroid
        self._blocked = max(0, self._blocked - 1)
        return self.response()

    def response(self) -> float | None:
        if len(self.moved) < self.window:
            return None
        commanded = np.asarray(self.commanded)
        keep = commanded >= self.min_command_m
        if not keep.any():
            return None
        return float(np.asarray(self.moved)[keep].sum() / commanded[keep].sum())

    def triggered(self) -> bool:
        response = self.response()
        return bool(response is not None and response < self.threshold and self._blocked == 0)

    def fired(self) -> None:
        """Arm the cooldown and forget the window, so one stall triggers once."""
        self._blocked = self.cooldown
        self.moved.clear()
        self.commanded.clear()
        self._previous = None


def observation_centroid(flat: np.ndarray, point_budget: int) -> np.ndarray:
    """The garment centroid the detector tracks, from the deployed observation alone."""
    features = df.observation_features(flat, point_budget)
    tool = np.asarray(df.unpack_observation(flat, point_budget)[3][0:3], dtype=np.float64)
    return features[2:5] + tool


def episode_summary(trace: list[dict], sustained: int = 12, threshold: float = 0.7,
                    grasp_limit: float = 0.02) -> dict:
    """Outcome of one full episode under the project's coverage-and-grasp rule."""
    if len(trace) < sustained:
        raise ValueError("Episode is shorter than the sustained window")
    upper = np.asarray([row["upperarm_ratio"] for row in trace], dtype=np.float64)
    tracking = np.asarray([row["tracking_error"] for row in trace], dtype=np.float64)
    sustained_coverage = float(upper[-sustained:].min())
    valid = bool(tracking.max() <= grasp_limit)
    return dict(final_coverage=float(upper[-1]), sustained_coverage=sustained_coverage,
                max_coverage=float(upper.max()), max_tracking_error=float(tracking.max()),
                whole_episode_grasp_valid=valid,
                success=bool(sustained_coverage >= threshold and valid))


def run_arms(cfg, checkpoint, out, *, macro_name: str, window: int, slots: int, seed: int, step_m: float,
             detector: dict, arms=("control", "intervene", "random")) -> dict:
    """Full episodes with and without the detector, from the same seeds.

    ``control`` never intervenes, ``intervene`` always runs ``macro_name`` when the
    detector fires, and ``random`` runs a uniformly drawn direction macro instead, so
    a gain that comes from interrupting the policy at all is separated from a gain
    that comes from this particular recovery.
    """
    import gc
    import json
    import time
    from dataclasses import replace

    from . import decision_branches as db
    from .dressing_env import GenesisIPCDressingEnv
    from .dressing_live import LiveCellFactory
    from .physics_gradient_actor import load_agent

    macro = next(m for m in db.MACROS if m["name"] == macro_name)
    alternatives = [m for m in db.MACROS if m["kind"] == "direction"]
    out.mkdir(parents=True)
    started = time.perf_counter()
    env = GenesisIPCDressingEnv(replace(cfg, workspace=str(out / "assets")), num_envs=slots,
                                cell_factory=LiveCellFactory(cfg.live))
    result = dict(completed=False, env=cfg.to_dict(), checkpoint=str(checkpoint), macro=macro_name,
                  window=window, slots=slots, seed=seed, macro_step_m=step_m, detector=detector,
                  physical_decisions=0, episodes=[])

    def save():
        result["seconds"] = time.perf_counter() - started
        (out / "result.json").write_text(json.dumps(result, indent=1) + "\n")

    try:
        agent = load_agent(checkpoint, env.spec.point_budget, env.action_dim, "cuda")
        seeds = [seed + i for i in range(slots)]
        for arm in arms:
            obs = env.reset(seeds)
            detectors = [StallDetector(**detector) for _ in range(slots)]
            rng = np.random.default_rng(seed)
            plans: list[tuple[int, dict, dict] | None] = [None] * slots
            traces: list[list[dict]] = [[] for _ in range(slots)]
            triggers = [0] * slots
            for _ in range(cfg.horizon):
                policy_action = agent.act(obs, deterministic=True)
                action = policy_action.copy()
                tools = env._anchor.copy()
                for i in range(slots):
                    commanded = float(np.linalg.norm(policy_action[i, :3]) * cfg.max_translation)
                    detectors[i].update(observation_centroid(obs[i], env.spec.point_budget), commanded)
                    if plans[i] is None and arm in ("intervene", "random") and detectors[i].triggered():
                        cell = env.cells[i]
                        chosen = macro if arm == "intervene" else alternatives[int(rng.integers(len(alternatives)))]
                        plans[i] = (0, db.slot_directions(tools[i], cell.finger, cell.elbow, cell.shoulder), chosen)
                        detectors[i].fired()
                        triggers[i] += 1
                    if plans[i] is not None:
                        t, directions, chosen = plans[i]
                        action[i] = db.macro_action(chosen, t, window, directions, policy_action[i],
                                                    step_m, cfg.max_translation)
                        plans[i] = (t + 1, directions, chosen) if t + 1 < window else None
                obs, _, done, rows = env.step(np.clip(action, -1.0, 1.0).astype(np.float32), reset_on_done=False)
                result["physical_decisions"] += slots
                if any(r.get("sim_error") for r in rows):
                    raise RuntimeError(f"Simulator failure in the {arm} arm")
                for i in range(slots):
                    traces[i].append({k: rows[i][k] for k in ("upperarm_ratio", "forearm_ratio", "tracking_error",
                                                              "grasp_valid", "commanded_translation_m",
                                                              "accepted_anchor_translation_m")})
            for i in range(slots):
                summary = episode_summary(traces[i])
                result["episodes"].append(dict(arm=arm, slot=i, seed=seeds[i], garment=env.cells[i].garment,
                                               human=int(env.cells[i].human), triggers=triggers[i],
                                               **summary, trace=traces[i]))
                print(json.dumps({k: v for k, v in result["episodes"][-1].items() if k != "trace"}), flush=True)
            save()
        result["completed"] = True
        save()
        return result
    finally:
        env.close()
        gc.collect()


def main():
    import argparse
    import json
    from dataclasses import replace
    from pathlib import Path

    from . import train_sac
    from .sac import SACAgent

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cell", default="tshirt_26:14046")
    p.add_argument("--macro", default="retreat_outward")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=5501)
    p.add_argument("--macro-step-m", type=float, default=0.008)
    p.add_argument("--detector-window", type=int, default=8)
    p.add_argument("--detector-threshold", type=float, default=0.15)
    p.add_argument("--detector-cooldown", type=int, default=20)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = SACAgent.read_checkpoint(args.checkpoint)
    targs = train_sac.build_parser().parse_args(["--eval-only"])
    train_sac.restore_resume_args(targs, ["--eval-only"], payload)
    train_sac.resolve_defaults(targs)
    garment, body = args.cell.rsplit(":", 1)
    cfg = replace(train_sac.dressing_config(targs), cells=((garment, int(body)),) * args.slots,
                  contact_force_readout=False, decision_watchdog=False)
    result = run_arms(cfg, args.checkpoint, args.out, macro_name=args.macro, window=args.window, slots=args.slots,
                      seed=args.seed, step_m=args.macro_step_m,
                      detector=dict(window=args.detector_window, threshold=args.detector_threshold,
                                    cooldown=args.detector_cooldown))
    for arm in ("control", "intervene", "random"):
        rows = [e for e in result["episodes"] if e["arm"] == arm]
        print(json.dumps(dict(arm=arm, episodes=len(rows), successes=sum(e["success"] for e in rows),
                              mean_sustained=float(np.mean([e["sustained_coverage"] for e in rows])),
                              grasp_valid=sum(e["whole_episode_grasp_valid"] for e in rows),
                              triggers=sum(e["triggers"] for e in rows))), flush=True)


if __name__ == "__main__":
    main()
