"""Bounded native IPC actor-interface experiment; not a dressing learning benchmark.

Every arm starts from the same checkpoint, including Adam moments. Rollouts share
initial simulator state and Gaussian noise. Report actual reward, without a critic
tail, and keep checkpoint/batch identities rather than treating slots as seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .iaql import soft_targets_detailed
from .iaql_env import IAQLClothEnv, IAQLEnvConfig
from .obs import ObsSpec
from .sac import SACAgent, SACConfig


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def load_agent(path):
    saved = SACAgent.read_checkpoint(path)
    agent = SACAgent(ObsSpec(3), 3, SACConfig.from_dict(saved['sac_config']), 'cpu')
    agent.load(path)
    return agent


def action(agent, obs, noise):
    with torch.no_grad():
        return agent.actor(tensor(obs), noise=noise)[1].numpy()


def update(agent, obs, noise, anchor, gradient, arm, radius, random_direction):
    """Return a clone; never modify the checkpoint or the reference optimizer."""
    if arm.endswith('_freshadam'):
        agent = copy.deepcopy(agent)
        agent.actor_optimizer.state.clear()
        arm = arm.removesuffix('_freshadam')
    candidate = copy.deepcopy(agent)
    reconstructed = tensor(action(candidate, obs, noise))
    if not torch.allclose(reconstructed, anchor, atol=1e-6, rtol=1e-5):
        raise ValueError('Actor does not reconstruct the saved same-action anchor')
    bounded = arm.startswith('bounded_')
    if arm in ('sac', 'mix'):
        candidate.cfg.physics_actor_mode = 'mix'
        candidate.cfg.physics_actor_weight = float(arm == 'mix')
        candidate.cfg.physics_actor_rho = .5
        candidate._update_actor_and_alpha(obs, state=obs, action_noise=noise,
            action_anchor=anchor, require_same_action=True,
            physics=(gradient, torch.ones(len(obs))) if arm == 'mix' else None)
        return candidate, 0
    direction = gradient / gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    if arm in ('random', 'bounded_random'):
        direction = random_direction
    elif arm == 'zero_target':
        direction = torch.zeros_like(direction)
    elif arm == 'negative':
        direction = -direction
    target = (anchor + radius * direction).clamp(-1, 1).detach()
    # Identical projected displacement makes the linear/regression first gradients
    # exactly equivalent at the anchor, including action-boundary effects.
    displacement = target - anchor
    for trial in range(12 if bounded else 1):
        if trial:
            candidate = copy.deepcopy(agent)  # Restore parameters AND Adam state.
        for group in candidate.actor_optimizer.param_groups:
            group['lr'] *= 0.5 ** trial
        pi = candidate.actor(obs, noise=noise)[1]
        if arm == 'raw':
            loss = -(gradient.detach() * (pi-anchor)).sum(-1).mean()
        elif arm == 'linear':
            loss = -(displacement * (pi-anchor)).sum(-1).mean()
        else:
            loss = .5 * (pi-target).square().sum(-1).mean()
        candidate.actor_optimizer.zero_grad()
        loss.backward()
        candidate._clip(candidate.actor)
        candidate.actor_optimizer.step()
        movement = tensor(action(candidate, obs, noise)) - anchor
        if not bounded or float(movement.norm(dim=-1).max()) <= radius:
            return candidate, trial
    return copy.deepcopy(agent), 12  # Rejected; no stale optimizer moments survive.


def rollout(env, snap, agent, noises):
    env.restore(snap)
    total = np.zeros(env.cfg.num_slots)
    for k, noise in enumerate(noises):
        out = env.step(action(agent, env.observation(), noise))
        total += agent.cfg.discount ** k * np.asarray(out['reward'])
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--slots', type=int, default=8)
    parser.add_argument('--batches', type=int, default=2)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--radius', type=float, default=.03)
    parser.add_argument('--arms', nargs='+', choices=['sac', 'mix', 'raw', 'linear', 'target',
        'bounded_target', 'random', 'negative', 'reward_target', 'zero_target', 'zero_target_freshadam',
        'target_freshadam', 'bounded_target_freshadam', 'random_freshadam',
        'bounded_random_freshadam'], default=['sac', 'mix', 'raw', 'linear', 'target',
        'bounded_target', 'random', 'negative', 'reward_target'])
    args = parser.parse_args()
    if args.slots < 2 or args.batches < 1 or args.horizon < 1 or args.radius <= 0:
        parser.error('Require slots>=2, positive batches, horizon and radius')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = IAQLClothEnv(args.out/'world', IAQLEnvConfig(num_slots=args.slots,
        export_mode='converged_raw', tangent_device='cuda'))
    rows = []
    start = time.monotonic()
    arms = args.arms
    for seed in args.seeds:
        path = Path(f'output/iaql/vec20k_s{seed}_actor_trust/online_beta_0.0.pt')
        base = load_agent(path)
        for batch in range(args.batches):
            generator = torch.Generator().manual_seed(78000 + seed*100 + batch)
            draw = lambda: torch.randn(args.slots, 3, generator=generator)
            env.reset(9200 + seed*100 + batch)
            for _ in range(8 + 4*batch):
                env.step(action(base, env.observation(), draw()))
            obs = tensor(env.observation())
            noises = [draw() for _ in range(args.horizon)]
            anchor = tensor(action(base, obs, noises[0]))
            snap = env.snapshot()
            captured = env.step(anchor.numpy(), capture=True)
            _, gradient, info = soft_targets_detailed(base.actor, base.critic_target,
                tensor(captured['obs']), tensor(captured['reward']).reshape(-1,1),
                torch.ones(args.slots,1), tensor(captured['tangent']),
                tensor(captured['reward_gradient']), base.alpha.detach(), base.cfg.discount,
                base.cfg.reward_abs_bound/(1-base.cfg.discount), noise=draw(),
                continuation_trust_kappa=base.cfg.continuation_trust_kappa)
            baseline = rollout(env, snap, base, noises)
            random_direction = draw()
            random_direction /= random_direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            actions = {}
            for arm in arms:
                teacher = tensor(captured['reward_gradient']) if arm == 'reward_target' else gradient
                candidate, backtracks = update(base, obs, noises[0], anchor, teacher,
                    arm, args.radius, random_direction)
                new_action = tensor(action(candidate, obs, noises[0]))
                movement = new_action-anchor
                improvement = rollout(env, snap, candidate, noises)-baseline
                row = dict(seed=seed, batch=batch, checkpoint=str(path), arm=arm,
                    reward_improvement=improvement.tolist(), baseline_reward=baseline.tolist(),
                    movement_norm=movement.norm(dim=-1).tolist(),
                    cosine=torch.nn.functional.cosine_similarity(movement, gradient, dim=-1).tolist(),
                    teacher_dot_movement=(gradient*movement).sum(-1).tolist(), backtracks=backtracks,
                    reward_norm=info['reward_norm'].tolist(), continuation_norm=info['continuation_norm'].tolist())
                actions[arm] = new_action
                rows.append(row)
                with (args.out/'rows.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                print(json.dumps(dict(seed=seed,batch=batch,arm=arm,
                    improvement=float(improvement.mean()),movement_max=float(movement.norm(dim=-1).max()),
                    backtracks=backtracks)), flush=True)
            floor = rollout(env, snap, base, noises)-baseline
            rows.append(dict(seed=seed,batch=batch,arm='repeat_baseline',reward_improvement=floor.tolist(),
                linear_target_action_max_difference=float((actions['linear']-actions['target']).abs().max()) if 'linear' in actions and 'target' in actions else None))
            (args.out/'report.json').write_text(json.dumps(dict(config=vars(args)|{'out':str(args.out)},
                elapsed_s=time.monotonic()-start,rows=rows),indent=2)+'\n')
    print(json.dumps({'finished':str(args.out/'report.json'),'elapsed_s':time.monotonic()-start}),flush=True)


if __name__ == '__main__':
    main()
