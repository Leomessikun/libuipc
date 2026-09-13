"""Cost of the RLT temporal stack and of one windowed update, on whatever device is free.

Usage: PYTHONPATH=python python -m uipc_manip.rlt_cost_probe [--device cpu|cuda] [--points 400]

Times, per call after warm-up:
  1. the temporal stack alone (no point clouds): run(), branch() and a 300-step stream at the
     production width, for window lengths 8, 32 and 300 and batch sizes 1 and 64;
  2. one decision for 25 streams through the acting path of each arm (raw-frame rollout state);
  3. one SAC update with the real heads (PointNet++ at the production widths, `points` per cloud)
     for the single-frame agent, the H4 frame history, and the RLT history at H = 4 and 8, with the
     batch chosen so every arm sees the same number of learning positions.
Prints JSON lines; nothing is written.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from uipc_manip.obs import ObsSpec
from uipc_manip.replay import FlatReplayBuffer
from uipc_manip.rlt import RLTConfig, RecurrentLoopedTransformer
from uipc_manip.sac import SACAgent, SACConfig


def timed(fn, repeats, device):
    fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def temporal_stack(device):
    cfg = RLTConfig()
    model = RecurrentLoopedTransformer(56, cfg).to(device).eval()
    for length in (8, 32, 300):
        for batch in (1, 64):
            tokens = torch.randn(batch, length, 56, device=device)
            valid = torch.ones(batch, length, dtype=torch.bool, device=device)
            with torch.no_grad():
                run_s = timed(lambda: model.run(tokens, valid), 3, device)
                run = model.run(tokens, valid)
                branch_s = timed(lambda: model.branch(run, tokens), 3, device)
            grad_s = None
            if batch == 64:
                def train_step():
                    model.zero_grad()
                    model.run(tokens, valid).s.sum().backward()
                grad_s = timed(train_step, 3, device)
            print(json.dumps({"probe": "temporal", "length": length, "batch": batch, "run_s": run_s, "branch_s": branch_s,
                              "run_backward_s": grad_s, "params": sum(p.numel() for p in model.parameters())}))
    state = model.init_state(1, 300, device)
    token = torch.randn(1, 56, device=device)
    with torch.no_grad():
        model.reset_state(state)
        def stream():
            model.reset_state(state)
            for _ in range(300):
                model.step(state, token)
        per_episode = timed(stream, 2, device)
    print(json.dumps({"probe": "stream", "steps": 300, "per_step_s": per_episode / 300}))


def updates(device, points):
    spec, action_dim = ObsSpec(points), 3
    positions = 64  # learning positions per update in every arm
    arms = [
        ("single", dict(history_length=1), positions),
        ("frames_h4", dict(history_length=4), positions),
        ("rlt_h4", dict(history_length=4, history_kind="rlt"), positions // 4),
        ("rlt_h8", dict(history_length=8, history_kind="rlt"), positions // 8),
    ]
    for name, extra, batch in arms:
        cfg = SACConfig(batch_size=batch, actor_type="wang-flow", critic_action_mode="dense", **extra)
        torch.manual_seed(0)
        agent = SACAgent(spec, action_dim, cfg, device)
        replay = FlatReplayBuffer(spec.dim, action_dim, 4096, batch, device, sequence=True)
        rng = np.random.default_rng(0)
        for episode in range(4):
            for step in range(64):
                block = np.zeros((points, 7), dtype=np.float32)
                block[:, :3] = rng.standard_normal((points, 3)) * 0.1
                block[:, 3] = 1.0  # every point a cloth point ...
                block[0, 3], block[0, 6] = 0.0, 1.0  # ... except one tool point
                obs = np.concatenate([block.reshape(-1), np.zeros(7, dtype=np.float32)])
                replay.add(obs, rng.uniform(-1, 1, action_dim), 0.0, obs, False, stream_id=0, episode_id=episode,
                           episode_step=step, episode_end=step == 63)
        seconds = timed(lambda: agent.update(replay), 2, device)
        print(json.dumps({"probe": "update", "arm": name, "windows": batch, "learning_positions": positions,
                          "frames_per_update": batch * (int(cfg.history_length) + 1 if extra.get("history_kind") == "rlt" else int(cfg.history_length)),
                          "update_s": seconds, "per_position_s": seconds / positions}))


def acting(device, points, num_envs=25):
    """One decision for `num_envs` streams: the single-frame policy, and the H4/H8 windows that
    collection re-encodes from raw frames every decision (`RolloutHistory`)."""
    from uipc_manip.history import act_with, rollout_state

    spec, action_dim = ObsSpec(points), 3
    rng = np.random.default_rng(0)
    block = np.zeros((num_envs, points, 7), dtype=np.float32)
    block[:, :, :3] = rng.standard_normal((num_envs, points, 3)) * 0.1
    block[:, :, 3] = 1.0
    block[:, 0, 3], block[:, 0, 6] = 0.0, 1.0
    obs = np.concatenate([block.reshape(num_envs, -1), np.zeros((num_envs, 7), dtype=np.float32)], axis=1)
    for name, extra in [("single", dict(history_length=1)), ("frames_h4", dict(history_length=4)),
                        ("rlt_h4", dict(history_length=4, history_kind="rlt")), ("rlt_h8", dict(history_length=8, history_kind="rlt"))]:
        torch.manual_seed(0)
        agent = SACAgent(spec, action_dim, SACConfig(actor_type="wang-flow", critic_action_mode="dense", **extra), device)
        history = rollout_state(agent, num_envs)
        for _ in range(8):  # fill the window so every frame is re-encoded
            act_with(agent, obs, False, history)
        seconds = timed(lambda: act_with(agent, obs, False, history), 5, device)
        print(json.dumps({"probe": "act", "arm": name, "num_envs": num_envs, "decision_s": seconds,
                          "frames_encoded": num_envs * int(agent.cfg.history_length)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--points", type=int, default=400)
    p.add_argument("--skip-updates", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    print(json.dumps({"probe": "env", "device": str(device), "threads": torch.get_num_threads(), "torch": torch.__version__}))
    temporal_stack(device)
    acting(device, args.points)
    if not args.skip_updates:
        updates(device, args.points)


if __name__ == "__main__":
    main()
