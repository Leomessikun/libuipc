"""IQL/BC controls using the existing dressing actor, dense critic and saved replay.

This is an implementation of Implicit Q-Learning (Kostrikov et al., ICLR 2022),
not a new IPC algorithm. No simulator or teacher labels are needed. Policy
outputs and observation packing remain compatible with SAC deployment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import _broadcast_action, reuse_neighbourhoods
from .sac import soft_update


def exact_successors(obs, next_obs, not_dones):
    """Recover conservative links in a chronological flat snapshot.

    Accept only unique, bitwise-equal observations, a later row, a nonterminal
    predecessor and a unique incoming edge. These are observation-contiguous
    segments, not certified simulator episode identities. Never infer adjacency
    from row order or a guessed vector-environment count.
    """
    if obs.shape != next_obs.shape or obs.ndim != 2 or len(not_dones) != len(obs):
        raise ValueError("Observation and mask shapes disagree")
    digest = lambda row: hashlib.blake2b(row.tobytes(), digest_size=16).digest()
    keys = {}
    for i, row in enumerate(obs):
        key = digest(row)
        keys[key] = -1 if key in keys else i
    links = np.full(len(obs), -1, dtype=np.int64)
    for i, row in enumerate(next_obs):
        j = keys.get(digest(row), -1)
        if j > i and float(not_dones[i, 0]) > 0 and np.isfinite(row).all() and np.array_equal(row, obs[j]):
            links[i] = j
    destinations, counts = np.unique(links[links >= 0], return_counts=True)
    links[np.isin(links, destinations[counts > 1])] = -1
    return links


def segment_ids(links):
    """Connected components of a forward-only one-predecessor transition graph."""
    links = np.asarray(links, dtype=np.int64)
    live = links >= 0
    if (links[live] >= len(links)).any() or (links[live] <= np.flatnonzero(live)).any():
        raise ValueError("Successors must be later rows")
    if len(np.unique(links[live])) != live.sum():
        raise ValueError("A segment cannot have multiple predecessors")
    incoming = np.zeros(len(links), dtype=bool)
    incoming[links[live]] = True
    ids = np.full(len(links), -1, dtype=np.int64)
    for component, start in enumerate(np.flatnonzero(~incoming)):
        row = int(start)
        while row >= 0:
            ids[row] = component
            row = int(links[row])
    return ids


def expectile_loss(difference, expectile):
    return (torch.where(difference > 0, expectile, 1 - expectile) * difference.square()).mean()


def advantage_weights(advantage, inverse_temperature, max_weight):
    # Clamp before exp so a divergent critic cannot overflow the weights.
    return (inverse_temperature * advantage.detach()).clamp(max=math.log(max_weight)).exp()


class ObservationValue(nn.Module):
    """Reuse the saved critic's architecture, initialized at its zero-action Q1.

    The zero command is a constant input, so this is a state value, independent
    of dataset/candidate actions. Its weights are independent of the Q network.
    """
    def __init__(self, critic):
        super().__init__()
        self.encoder = copy.deepcopy(critic.encoder)
        self.head = copy.deepcopy(critic.Q1)
        self.action_mode, self.action_dim, self.use_extra = critic.action_mode, critic.action_dim, critic.use_extra

    def forward(self, obs):
        pos, feat, valid, extra = obs
        zero = pos.new_zeros((len(pos), self.action_dim))
        if self.action_mode == "dense":
            feat = _broadcast_action(feat, zero)
        z = self.encoder(pos, feat, valid)
        parts = [z] if self.action_mode == "dense" else [z, zero]
        if self.use_extra:
            parts.append(extra)
        return self.head(torch.cat(parts, dim=-1))


class ImplicitQLearner:
    """In-support Bellman learning and advantage-weighted log likelihood.

    V fits an upper expectile of target Q on *recorded* actions; Q fits
    r + gamma * mask * V(next observation). The actor imitates dataset actions
    weighted by exp(beta * (target Q - V)). No Q action derivative is used.
    The optional BC control has the identical actor update, with unit weights.
    """
    def __init__(self, agent, *, expectile=0.7, inverse_temperature=3.0, max_weight=100.0):
        if (not 0.5 <= expectile < 1 or not math.isfinite(inverse_temperature)
                or inverse_temperature <= 0 or not math.isfinite(max_weight) or max_weight < 1):
            raise ValueError("Invalid expectile or advantage weighting")
        cfg = agent.cfg
        if cfg.history_length != 1 or cfg.critic_input != "points" or cfg.algo != "sac":
            raise ValueError("This control requires a single-frame scalar point-cloud critic")
        if cfg.physics_actor_weight or cfg.adjoint_weight or cfg.random_shift_scale or cfg.point_jitter_scale:
            raise ValueError("Offline controls require unaugmented data and no physics loss")
        self.agent, self.expectile = agent, float(expectile)
        self.inverse_temperature, self.max_weight = float(inverse_temperature), float(max_weight)
        self.value = ObservationValue(agent.critic).to(agent.device)
        self.optimizer = torch.optim.Adam(self.value.parameters(), lr=cfg.critic_lr,
                                          fused=agent.device.type == "cuda")
        self.updates = 0

    def update(self, batch, *, behavior_cloning=False):
        agent = self.agent
        flat, action, reward, next_flat, mask = batch[:5]
        with reuse_neighbourhoods():
            obs, nxt = agent._unpack(flat), agent._unpack(next_flat)
            stats = {}
            if behavior_cloning:
                weights = torch.ones_like(reward)
            else:
                with torch.no_grad():
                    q1, q2 = agent.critic_target(obs, action)
                    data_q = torch.minimum(q1, q2)
                    target = reward + agent.cfg.discount * mask * self.value(nxt)
                v = self.value(obs)
                advantage = data_q - v
                value_loss = expectile_loss(advantage, self.expectile)
                weights = advantage_weights(advantage, self.inverse_temperature, self.max_weight)
                self.optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                self.optimizer.step()
                q1, q2 = agent.critic(obs, action)
                critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
                agent.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                agent.critic_optimizer.step()
                soft_update(agent.critic, agent.critic_target, agent.cfg.critic_tau)
                stats.update(value_loss=float(value_loss.detach()), critic_loss=float(critic_loss.detach()),
                             data_q=float(data_q.mean()), advantage_std=float(advantage.detach().std()))
            actor_loss = -(weights * agent.actor.action_log_prob(obs, action)).mean()
            agent.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            agent.actor_optimizer.step()
            self.updates += 1
            agent.updates += 1
            stats.update(actor_loss=float(actor_loss.detach()), weight_mean=float(weights.mean()),
                         weight_ess=float(weights.sum().square() / weights.square().sum().clamp_min(1e-12)))
            if not all(math.isfinite(x) for x in stats.values()):
                raise FloatingPointError(f"Nonfinite offline update: {stats}")
            return stats


def main(argv=None):
    from .physics_gradient_actor import load_agent

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("iql", "bc", "sac"), default="iql")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--inverse-temperature", type=float, default=3.0)
    args = p.parse_args(argv)
    if args.steps < 1 or args.batch_size < 1:
        p.error("steps and batch-size must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "result.json").exists():
        raise FileExistsError("Use a new run directory")
    started = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    metadata = json.loads((args.replay / "replay.json").read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    transitions = int(checkpoint["metadata"]["transitions"])
    if transitions != int(metadata["metadata"]["transitions"]):
        raise ValueError("Checkpoint and replay transition counts differ")
    reward_scale = float(metadata["metadata"]["reward_scale"])
    expected_scale = (1 - checkpoint["sac_config"]["discount"]) / 0.01
    if not np.isclose(reward_scale, expected_scale):
        raise ValueError("Replay reward scale differs from the Wang checkpoint contract")
    agent = load_agent(args.checkpoint, int(checkpoint["protocol"]["point_budget"]), metadata["action_dim"], args.device)
    # All arms start with the same saved weights and fresh optimizers: an explicit
    # offline objective switch, not a resumed SAC experiment.
    agent.cfg.batch_size = args.batch_size
    agent.train(True)
    with np.load(args.replay / "replay.npz") as f:
        arrays = {k: f[k] for k in ("obs", "actions", "rewards", "next_obs", "not_dones")}
    links = exact_successors(arrays["obs"], arrays["next_obs"], arrays["not_dones"])
    segments = segment_ids(links)
    # Same complete connected segments across arms; no row-wise validation leak.
    rng = np.random.default_rng(args.seed)
    groups = rng.permutation(np.unique(segments))
    if len(groups) < 2:
        raise ValueError("Need at least two connected segments for a validation split")
    val = np.isin(segments, groups[:max(1, len(groups) // 5)])
    train_rows = torch.as_tensor(np.flatnonzero(~val), device=args.device)
    val_rows = torch.as_tensor(np.flatnonzero(val), device=args.device)
    # Keep the existing corpus on the GPU. This is offline optimization, so the
    # IPC collector cannot serialize learning or consume simulation steps.
    tensors = [torch.as_tensor(arrays[k], device=args.device) for k in arrays]
    learner = ImplicitQLearner(agent, expectile=args.expectile, inverse_temperature=args.inverse_temperature)
    result = {"mode": args.mode, "checkpoint": str(args.checkpoint.resolve()), "replay": str(args.replay.resolve()),
              "new_simulator_transitions": 0, "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
              "expectile": args.expectile, "inverse_temperature": args.inverse_temperature,
              "initialization": "saved SAC weights, fresh optimizers", "reward_scale": reward_scale,
              "rows": len(segments), "segments": len(groups), "matched_links": int((links >= 0).sum()),
              "train_rows": len(train_rows), "validation_rows": len(val_rows), "logs": []}
    del arrays

    class BatchReplay:
        def sample(self, n):
            indices = train_rows[torch.randint(len(train_rows), (n,), device=args.device)]
            return tuple(x[indices] for x in tensors)

    replay = BatchReplay()
    # Fix validation observations once and do not consume training RNG to select them.
    vr = val_rows[:min(1024, len(val_rows))]
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    result["setup_seconds"] = time.perf_counter() - started
    loop_start = time.perf_counter()
    with (args.out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            stats = agent.update(replay) if args.mode == "sac" else learner.update(
                replay.sample(args.batch_size), behavior_cloning=args.mode == "bc")
            if step % 100 == 0 or step == args.steps:
                with torch.no_grad():
                    nll = []
                    for rows in vr.split(args.batch_size):
                        nll.append(-agent.actor.action_log_prob(agent._unpack(tensors[0][rows]), tensors[1][rows]).mean())
                    stats["validation_nll"] = float(torch.stack(nll).mean())
                record = {"step": step, "seconds": time.perf_counter() - loop_start, **stats}
                result["logs"].append(record)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    result["training_seconds"] = time.perf_counter() - loop_start
    result["actor_updates"] = args.steps if args.mode != "sac" else sum(
        i % agent.cfg.actor_update_freq == 0 for i in range(agent.updates - args.steps + 1, agent.updates + 1))
    # This is an offline policy export at the original simulation count, not
    # additional transitions. The sidecar makes the new training objective explicit.
    agent.save(args.out / "policy.pt", step=int(checkpoint["step"]),
               metadata={**checkpoint["metadata"], "offline_rl": {k: v for k, v in result.items() if k != "logs"}})
    if args.mode == "iql":
        torch.save({"value": learner.value.state_dict(), "optimizer": learner.optimizer.state_dict(),
                    "updates": learner.updates}, args.out / "iql_value.pt")
    result["total_seconds"] = time.perf_counter() - started
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return args.out


if __name__ == "__main__":
    main()
