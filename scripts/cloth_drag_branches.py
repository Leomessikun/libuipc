"""The same decision diagnostic, on a task where reinforcement learning does work.

Every measurement of decision structure in this project has been made on dressing,
where every method has failed, so a null result there cannot distinguish "this task has
no learnable decision at these states" from "this diagnostic always says no". The
cloth-drag benchmark is the control: the same simulator, the same solver, a
three-dimensional action, and a SAC policy whose return rises from -4 to +13 over
20,000 transitions, so the task demonstrably has something to learn.

The protocol is the dressing one. At a state the policy visits, the world is
snapshotted and a fixed macro replaces the policy for a short window before the same
policy resumes; each macro is repeated with identical commands. What comes out is the
consequence of the best macro, the spread of a macro against itself, whether the winner
depends on the state, and how much choosing per state is worth.

The macros are the natural directions of this task: toward the goal, away from it, two
perpendicular to it, up, and the policy's own action at the macros' magnitude.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

MACROS = ("policy", "policy_scaled", "toward", "away", "perpendicular_a", "perpendicular_b", "up")


def unit(v: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.asarray(fallback, dtype=np.float64)


def frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The goal direction and two directions perpendicular to it."""
    toward = unit(direction, (1.0, 0.0, 0.0))
    seed = np.array([0.0, 0.0, 1.0]) if abs(toward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = unit(np.cross(toward, seed))
    return toward, a, unit(np.cross(toward, a))


def macro_action(name: str, policy_action: np.ndarray, directions, scale: float) -> np.ndarray:
    toward, perp_a, perp_b = directions
    if name == "policy":
        return np.asarray(policy_action, dtype=np.float64)
    if name == "policy_scaled":
        return unit(policy_action, (1.0, 0.0, 0.0)) * scale
    vector = {"toward": toward, "away": -toward, "perpendicular_a": perp_a,
              "perpendicular_b": perp_b, "up": np.array([0.0, 0.0, 1.0])}[name]
    return vector * scale


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--snapshot-steps", default="20,50,80")
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--follow", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--slots", type=int, default=8)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--friction", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=1.0, help="Macro command as a fraction of the limit")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    steps = sorted(int(s) for s in args.snapshot_steps.split(","))
    if args.out.exists():
        raise FileExistsError(args.out)
    if steps[-1] + args.window + args.follow >= args.horizon:
        raise ValueError("The last branch must end before the horizon")
    from uipc_manip.iaql_benchmark import agent_for
    from uipc_manip.iaql_env import IAQLClothEnv, IAQLEnvConfig

    args.out.mkdir(parents=True)
    started = time.perf_counter()
    env = IAQLClothEnv(args.out / "world", IAQLEnvConfig(friction=args.friction, num_slots=args.slots,
                                                         horizon=args.horizon))
    result = dict(completed=False, checkpoint=str(args.checkpoint), environment=env.describe(),
                  macros=list(MACROS), snapshot_steps=steps, window=args.window, follow=args.follow,
                  repeats=args.repeats, slots=args.slots, seed=args.seed, scale=args.scale,
                  states=[], records=[], physical_decisions=0)

    arrays: dict[str, np.ndarray] = {}

    def save():
        result["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        np.savez_compressed(args.out / "states.npz", **arrays)

    try:
        agent = agent_for(env.obs_dim, seed=args.seed, device=args.device, checkpoint=args.checkpoint)
        agent.train(False)
        obs = env.reset(args.seed)
        step = 0
        for target in steps:
            while step < target:
                obs = env.step(agent.act(obs, deterministic=True))["obs"]
                result["physical_decisions"] += args.slots
                step += 1
            snapshot = env.snapshot()
            observation = np.asarray(env.observation()).reshape(args.slots, -1)
            positions = env.all_positions()
            markers = positions[:, env.marker].mean(1)
            directions = [frame(env.goals[i] - markers[i]) for i in range(args.slots)]
            ids = []
            for i in range(args.slots):
                sid = f"seed{args.seed}__slot{i}__step{target}"
                ids.append(sid)
                result["states"].append(dict(state=sid, slot=i, step=target,
                                             distance=float(env.all_distances()[i]),
                                             goal=[float(x) for x in env.goals[i]],
                                             marker=[float(x) for x in markers[i]]))
                # The fourth measurement — whether the choice is identifiable from what the
                # policy sees — needs the state's own observation, as on dressing.
                arrays[f"{sid}__observation"] = observation[i]
            for repeat in range(args.repeats):
                for macro in MACROS:
                    env.restore(snapshot)
                    branch = env.observation()
                    traces: list[list[dict]] = [[] for _ in range(args.slots)]
                    for t in range(args.window + args.follow):
                        policy_action = np.asarray(agent.act(branch, deterministic=True)).reshape(args.slots, 3)
                        if t < args.window:
                            action = np.stack([macro_action(macro, policy_action[i], directions[i], args.scale)
                                               for i in range(args.slots)])
                        else:
                            action = policy_action
                        out = env.step(np.clip(action, -1, 1))
                        branch = out["obs"]
                        result["physical_decisions"] += args.slots
                        distances = np.asarray(out["distance"]).reshape(-1)
                        rewards = np.asarray(out["reward"]).reshape(-1)
                        for i in range(args.slots):
                            traces[i].append(dict(distance=float(distances[i]), reward=float(rewards[i])))
                    for i in range(args.slots):
                        distance = np.asarray([row["distance"] for row in traces[i]])
                        result["records"].append(dict(
                            state=ids[i], macro=macro, repeat=repeat, slot=i, step=target,
                            # Higher is better, to match the dressing analysis: the negative of the
                            # worst distance over the branch's last twelve decisions.
                            sustained_coverage=float(-distance[-12:].max()),
                            final_distance=float(distance[-1]), best_distance=float(distance.min()),
                            reward_sum=float(sum(row["reward"] for row in traces[i]))))
                    print(json.dumps(dict(step=target, macro=macro, repeat=repeat,
                                          final=[round(r["final_distance"], 4)
                                                 for r in result["records"][-args.slots:]])), flush=True)
                    save()
            env.restore(snapshot)
            obs = env.observation()
            step = target
        result["completed"] = True
        save()
        print(json.dumps(dict(states=len(result["states"]), branches=len(result["records"]),
                              decisions=result["physical_decisions"],
                              seconds=round(result["seconds"], 1))), flush=True)
    finally:
        env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    main()
