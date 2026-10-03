"""Reuse v5 valid prefixes for a bounded interaction-goal proposal pilot.

Run in curl's CPU Python. Whole-garment validation is fixed before fitting.
Low action error is not evidence of goal realization or successful dressing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from dynamic_student import encode, sha256
from motion_observation import history_inputs
from outcome_goals import (InteractionCoordinates, base_action, future_indices, goal_input,
                           proposal_net, residual_action, valid_prefix)
from physical_sleeve import DEFAULT_OBJ, SleeveSections, read_obj
from train_dynamic_student import BASE, PACKAGE


def extract(args):
    sys.path.insert(0, str(PACKAGE))
    import torch
    from uipc_manip.obs import ObsSpec
    from uipc_manip.wang_bridge import ReferencePolicy
    torch.set_num_threads(1)
    policy = ReferencePolicy(BASE, device='cpu', yaw_deg=267.)
    spec = ObsSpec(768)
    args.out.mkdir(parents=True, exist_ok=False)
    audit = json.loads(args.audit.read_text())
    records = []
    for index, row in enumerate(audit['rows']):
        source = Path(row['path']).resolve()
        with np.load(source, allow_pickle=False) as d:
            meta = json.loads(str(d['metadata_json']))
            if meta['checkpoint_sha256'] != sha256(BASE):
                raise ValueError(f'Mixed base policy: {source}')
            end = valid_prefix(d['grasp_valid'], meta)
            if end < args.horizon:
                print('[excluded] ' + str(source), flush=True)
                continue
            obs, cloth = d['obs'][:end + 1], d['positions'][:end + 1]
            tools, actions = d['tcp'][:end + 1], d['actions'][:end]
            landmarks = np.stack([d[k] for k in ('finger', 'elbow', 'shoulder')])
            rest, faces = read_obj(DEFAULT_OBJ.parent / (row['garment'] + '.obj'))
            if not np.array_equal(faces, d['faces']):
                raise ValueError('Rest mesh topology differs from saved cloth')
            sections = SleeveSections(rest * 4., faces, d['opening_idx'])
        coordinates = InteractionCoordinates(sections, landmarks)
        features, bases, desc = [], [], []
        for observation, positions in zip(obs, cloth):
            pos, flags, valid, _ = spec.unpack_numpy(observation)
            feature, logits = encode(policy, pos[valid], flags[valid])
            features.append(feature)
            bases.append(base_action(logits, policy.rotation))
            desc.append(coordinates.describe(positions))
        file = args.out / f'episode_{index:03d}.npz'
        np.savez_compressed(file, features=np.asarray(features), base=np.asarray(bases),
                            descriptor=np.asarray(desc), tools=tools, actions=actions,
                            landmarks=landmarks, rotation=policy.rotation)
        record = dict(path=str(file.resolve()), source=str(source), source_sha256=sha256(source),
                      garment=row['garment'], body=meta['body'], valid_decisions=end,
                      collector_success=bool(meta['accepted']), sections=sections.describe(),
                      arm_length=coordinates.length)
        records.append(record)
        print('[extracted] ' + json.dumps({k: record[k] for k in ('garment', 'body', 'valid_decisions')}), flush=True)
        (args.out / 'manifest.partial.json').write_text(json.dumps(records, indent=2) + '\n')
    manifest = dict(episodes=records, base_checkpoint=str(BASE), base_sha256=sha256(BASE),
                    yaw=267., feature_rng_seed=0, audit_sha256=sha256(args.audit),
                    horizon=args.horizon, validation_garment=args.validation_garment,
                    extraction_sha256=sha256(__file__), geometry_sha256=sha256(Path(__file__).with_name('outcome_goals.py')))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def canonicalize(args):
    """Reuse encoded frames while dropping the non-corresponding .75/.55 ring."""
    original = args.features / 'manifest.json'
    manifest = json.loads(original.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for record in manifest['episodes']:
        if not np.allclose(record['sections']['section_fractions'][:3], [0., .25, .5]):
            raise ValueError('Inconsistent remaining section fractions')
        source = Path(record['path'])
        with np.load(source, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
        if arrays['descriptor'].shape[1:] != (5, 9):
            raise ValueError('Expected the initial five-section cache')
        arrays['descriptor'] = arrays['descriptor'][:, [0, 1, 2, 4]]
        destination = args.out / source.name
        np.savez_compressed(destination, **arrays)
        record.update(parent_cache_sha256=sha256(source), path=str(destination.resolve()))
    manifest.update(parent_manifest_sha256=sha256(original),
                    cache_conversion='Drop variable .75/.55 section; keep cuff/.25/.5/armhole; reuse unchanged encoder outputs',
                    geometry_sha256=sha256(Path(__file__).with_name('outcome_goals.py')))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def dataset(manifest, frames, seed, chunk):
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ('history', 'geometry', 'generic', 'duration', 'base', 'action', 'val', 'episode')}
    for episode, record in enumerate(manifest['episodes']):
        with np.load(record['path'], allow_pickle=False) as d:
            history = history_inputs(d['features'], d['tools'] @ d['rotation'].T, frames)
            for t in range(0, record['valid_decisions'] - chunk + 1, 2):
                # Full-length and shorter remaining paths share the same decoder.
                for h in {min(manifest['horizon'], record['valid_decisions'] - t),
                          int(rng.integers(chunk, min(manifest['horizon'], record['valid_decisions'] - t) + 1))}:
                    future = future_indices(t, h)
                    rows['history'].append(history[t])
                    rows['geometry'].append(goal_input(d['descriptor'][future], d['landmarks'], d['tools'][t], d['rotation']))
                    rows['generic'].append((d['features'][future] - d['features'][t]).reshape(-1))
                    rows['duration'].append([h / manifest['horizon']])
                    rows['base'].append(d['base'][t])
                    rows['action'].append(d['actions'][t:t + chunk])
                    rows['val'].append(record['garment'] == manifest['validation_garment'])
                    rows['episode'].append(episode)
    return {k: np.asarray(v) for k, v in rows.items()}


def train(args):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    code_hashes = dict(geometry_sha256=sha256(Path(__file__).with_name('outcome_goals.py')),
                       train_sha256=sha256(__file__))
    manifest = json.loads((args.features / 'manifest.json').read_text())
    data = dataset(manifest, args.frames, args.seed, args.chunk)
    train_idx, val_idx = np.flatnonzero(~data['val']), np.flatnonzero(data['val'])
    if not len(train_idx) or not len(val_idx):
        raise ValueError('Need disjoint whole-garment train and validation sets')
    args.out.mkdir(parents=True, exist_ok=False)
    base, target = torch.from_numpy(data['base']).float(), torch.from_numpy(data['action']).float()
    summary = {}
    for kind in ('geometry', 'generic', 'no_goal'):
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        goal = data[kind] if kind != 'no_goal' else np.zeros((len(base), 0))
        values = np.concatenate((data['history'], goal, data['duration']), axis=1).astype(np.float32)
        mean, std = values[train_idx].mean(0), np.maximum(values[train_idx].std(0), .05)
        x = torch.from_numpy((values - mean) / std)
        model = proposal_net(x.shape[1], args.chunk)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        # Equal episode sampling, so long or successful trajectories do not dominate.
        episodes = np.unique(data['episode'][train_idx])
        ep_indices = {ep: train_idx[data['episode'][train_idx] == ep] for ep in episodes}
        log = (args.out / f'{kind}.jsonl').open('w')
        for update in range(args.updates + 1):
            if update:
                indices = np.array([rng.choice(ep_indices[ep]) for ep in rng.choice(episodes, 128)])
                action, correction = residual_action(model, x[indices], base[indices])
                loss = (action - target[indices]).square().mean() + .01 * correction.square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite proposal loss')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            if update % 200 == 0 or update == args.updates:
                with torch.no_grad():
                    pred, _ = residual_action(model, x[val_idx], base[val_idx])
                    fitted, _ = residual_action(model, x[train_idx], base[train_idx])
                    shuffled = x[val_idx].clone()
                    if goal.shape[1]:
                        # Same observations, permuted desired outcomes.
                        goal_slice = slice(data['history'].shape[1], -1)
                        shuffled[:, goal_slice] = shuffled[torch.randperm(len(val_idx)), goal_slice]
                    shuffled_pred, _ = residual_action(model, shuffled, base[val_idx])
                row = dict(update=update, train_mse=float((fitted - target[train_idx]).square().mean()),
                           validation_mse=float((pred - target[val_idx]).square().mean()),
                           base_validation_mse=float((base[val_idx, None] - target[val_idx]).square().mean()),
                           shuffled_goal_mse=float((shuffled_pred - target[val_idx]).square().mean()),
                           goal_action_rms=float((pred - shuffled_pred).square().mean().sqrt()))
                log.write(json.dumps(row) + '\n'); log.flush()
                print(f'[train {kind}] ' + json.dumps(row), flush=True)
        log.close()
        config = dict(kind=kind, frames=args.frames, seed=args.seed, updates=args.updates,
                      chunk=args.chunk, feature_manifest=str((args.features / 'manifest.json').resolve()),
                      feature_manifest_sha256=sha256(args.features / 'manifest.json'),
                      base_checkpoint=manifest['base_checkpoint'], base_sha256=manifest['base_sha256'],
                      yaw=manifest['yaw'], horizon=manifest['horizon'], validation_garment=manifest['validation_garment'],
                      train_samples=len(train_idx), validation_samples=len(val_idx), input_dim=x.shape[1],
                      **code_hashes,
                      selection='fixed update count; validation garment excluded from training and source retrieval')
        torch.save(dict(config=config, state_dict=model.state_dict(), mean=mean, std=std), args.out / f'{kind}.pt')
        summary[kind] = dict(config=config, metrics=row)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    f = sub.add_parser('extract')
    f.add_argument('--audit', type=Path, default=Path('agent_docs/performance/2026-10-03-existing-progress-audit.json'))
    f.add_argument('--horizon', type=int, default=24)
    f.add_argument('--validation-garment', default='tshirt_4')
    t = sub.add_parser('train')
    t.add_argument('--features', type=Path, required=True)
    t.add_argument('--frames', type=int, default=4)
    t.add_argument('--chunk', type=int, default=8)
    t.add_argument('--seed', type=int, default=20261003)
    t.add_argument('--updates', type=int, default=1200)
    c = sub.add_parser('canonicalize')
    c.add_argument('--features', type=Path, required=True)
    for command in (f, t, c):
        command.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    dict(extract=extract, train=train, canonicalize=canonicalize)[args.stage](args)


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
