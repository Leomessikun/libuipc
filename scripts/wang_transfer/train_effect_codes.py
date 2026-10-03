"""Prepare matched action/effect partitions and train small flow corrections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from dynamic_student import sha256
from effect_codes import (ACTION_AXES, VARIANTS, build_models, conditioning,
                          context_inputs, draw, effect_vector, kmeans, nearest)
from train_dynamic_student import PACKAGE


def prepare(args):
    from train_outcome_goals import branches
    args.out.mkdir(parents=True, exist_ok=False)
    cache = args.out / 'features'
    branches(SimpleNamespace(features=args.features, out=cache, runs=args.runs, validation_seed=-1))
    manifest = json.loads((cache / 'manifest.json').read_text())
    rows = {k: [] for k in ('context', 'actions', 'effects', 'val', 'episode', 'counterfactual', 'root_start')}
    index = []
    for episode, record in enumerate(manifest['episodes']):
        with np.load(record['path'], allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        start = record.get('start_decision', 0)
        if record.get('counterfactual'):
            # These collectors roll in with the nominal deterministic r1 policy.
            # Context contains preceding states, not actions from the branch.
            data['actions'][:start] = data['base'][:start]
        contexts = context_inputs(data['features'], data['tools'], data['actions'], data['base'], data['rotation'])
        validation = (record['garment'] == manifest['validation_garment']
                      or (record.get('counterfactual', False) and record['body'] == args.validation_body))
        for t in range(start, len(data['actions']) - args.horizon + 1, 1 if record.get('counterfactual') else 4):
            path = data['descriptor'][t:t + args.horizon + 1]
            if not np.isfinite(path).all():
                raise ValueError('Missing context geometry leaked into an effect target')
            rows['context'].append(contexts[t])
            rows['actions'].append(data['actions'][t:t + args.horizon, ACTION_AXES])
            rows['effects'].append(effect_vector(path))
            rows['val'].append(validation); rows['episode'].append(episode)
            rows['counterfactual'].append(record.get('counterfactual', False))
            rows['root_start'].append(bool(record.get('counterfactual') and t == start))
            index.append(dict(episode=episode, time=t, source=record['source'], garment=record['garment'],
                              body=record['body'], validation=validation))
    data = {k: np.asarray(v) for k, v in rows.items()}
    train = np.flatnonzero(~data['val'])
    validation = np.flatnonzero(data['val'])
    if not len(train) or not len(validation):
        raise ValueError('Missing disjoint training or validation episodes')
    rng = np.random.default_rng(20261005)
    fit = []
    later = bool(np.any(data['counterfactual'][train] & ~data['root_start'][train]))
    masks = [(~data['counterfactual'], 1600),
             (data['counterfactual'] & data['root_start'], 800 if later else 1600),
             (data['counterfactual'] & ~data['root_start'], 800)]
    for mask, size in masks:
        eligible = train[mask[train]]
        if not len(eligible):
            continue
        episodes = np.unique(data['episode'][eligible])
        by_episode = {e: eligible[data['episode'][eligible] == e] for e in episodes}
        fit.extend([rng.choice(by_episode[e]) for e in rng.choice(episodes, size)])
    fit = np.asarray(fit)
    action_mean = data['actions'][train].reshape(len(train), -1).mean(0)
    action_std = np.maximum(data['actions'][train].reshape(len(train), -1).std(0), .1)
    action_values = (data['actions'].reshape(len(data['actions']), -1) - action_mean) / action_std
    action_centers = kmeans(action_values[fit], args.codes)
    effect_centers = kmeans(data['effects'][fit], args.codes)
    data['action_codes'] = nearest(action_values, action_centers)
    data['effect_codes'] = nearest(data['effects'], effect_centers)
    data['shuffled_effect_codes'] = data['effect_codes'].copy()
    # Preserve each training root/body's marginal label distribution. Paired
    # replay branches of a body stay together; no validation rows are touched.
    groups = {}
    for i in train:
        row = index[i]
        key = (row['garment'], row['body']) if data['counterfactual'][i] else ('episode', row['episode'])
        groups.setdefault(key, []).append(i)
    for values in groups.values():
        data['shuffled_effect_codes'][values] = data['effect_codes'][rng.permutation(values)]
    np.savez_compressed(args.out / 'dataset.npz', **data)
    np.savez_compressed(args.out / 'partition.npz', action_centers=action_centers,
                        action_mean=action_mean, action_std=action_std, effect_centers=effect_centers)
    root_rows = data['root_start'] & ~data['val']
    summary = dict(rows=len(index), train_rows=len(train), validation_rows=len(validation),
                   counterfactual_train_rows=int(np.sum(data['counterfactual'][train])),
                   counterfactual_validation_rows=int(np.sum(data['counterfactual'][validation])),
                   root_start_effect_counts=np.bincount(data['effect_codes'][root_rows], minlength=args.codes).tolist(),
                   persistent_wrap_event_rows=int(np.sum(np.any(data['effects'][:, 24:32] != 0, axis=1))),
                   effect_counts_train=np.bincount(data['effect_codes'][train], minlength=args.codes).tolist(),
                   effect_counts_validation=np.bincount(data['effect_codes'][validation], minlength=args.codes).tolist())
    result = dict(manifest)
    result.update(horizon=args.horizon, codes=args.codes, frames=4, validation_body=args.validation_body,
                  index=index, summary=summary, partition_fit_rows=fit.tolist(),
                  dataset_sha256=sha256(args.out / 'dataset.npz'), partition_sha256=sha256(args.out / 'partition.npz'),
                  context_contract='Causal cloud/tool history, previous executed actions and current r1 action only',
                  method_contract='Observed-effect k-means initialization; valid branch prefixes only; no task-positive actor labels',
                  validation_contract='Entire tshirt_4 adapter-validation garment and all intervention branches of body 2034; development only')
    (args.out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print('[prepared] ' + json.dumps(summary), flush=True)


def train(args):
    sys.path.insert(0, str(PACKAGE))
    import torch
    torch.set_num_threads(2)
    manifest = json.loads((args.data / 'manifest.json').read_text())
    with np.load(args.data / 'dataset.npz', allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    train_idx, val_idx = np.flatnonzero(~data['val']), np.flatnonzero(data['val'])
    mean, std = data['context'][train_idx].mean(0), np.maximum(data['context'][train_idx].std(0), .05)
    context = torch.from_numpy((data['context'] - mean) / std).float()
    target = torch.from_numpy(data['actions']).float()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'interface.json').write_text(json.dumps(dict(horizon=manifest['horizon'])) + '\n')
    np.savez(args.out / 'partition.npz', **dict(np.load(args.data / 'partition.npz')))
    summary = {}
    config = dict(context_dim=context.shape[1], horizon=manifest['horizon'], codes=manifest['codes'],
                  frames=4, width=args.width, depth=2, seed=args.seed, updates=args.updates,
                  base_checkpoint=manifest['base_checkpoint'], base_sha256=manifest['base_sha256'],
                  yaw=manifest['yaw'], dataset_sha256=manifest['dataset_sha256'],
                  partition_sha256=manifest['partition_sha256'], data=str(args.data.resolve()),
                  sampling='half nominal; half interventions, explicitly upweight exact branch roots when later windows exist',
                  module_sha256=sha256(Path(__file__).with_name('effect_codes.py')),
                  trainer_sha256=sha256(__file__), initialized_from='Frozen r1 point encoder and nominal action; new correction heads')
    # Balance all variants identically, with explicit mass on histories that
    # have different intervention actions rather than unique nominal labels.
    sample_groups = []
    later = bool(np.any(data['counterfactual'][train_idx] & ~data['root_start'][train_idx]))
    masks = [(~data['counterfactual'], 64),
             (data['counterfactual'] & data['root_start'], 32 if later else 64),
             (data['counterfactual'] & ~data['root_start'], 32)]
    for mask, size in masks:
        eligible = train_idx[mask[train_idx]]
        if not len(eligible):
            continue
        episodes = np.unique(data['episode'][eligible])
        sample_groups.append(({ep: eligible[data['episode'][eligible] == ep] for ep in episodes}, size))
    for variant in VARIANTS:
        torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
        models = build_models(torch, config['context_dim'], config['horizon'], config['codes'], args.width, 2)
        optimizer = torch.optim.AdamW(models.parameters(), lr=3e-4, weight_decay=1e-4)
        labels_np = np.zeros(len(context), np.int64) if variant == 'flat' else data[variant + '_codes']
        labels = torch.from_numpy(labels_np).long()
        with (args.out / (variant + '.jsonl')).open('w') as log:
            for update in range(1, args.updates + 1):
                indices = []
                for group, size in sample_groups:
                    keys = list(group)
                    indices.extend([rng.choice(group[k]) for k in rng.choice(keys, size)])
                idx = np.asarray(indices)
                x1, c, lab = target[idx], context[idx], labels[idx]
                cond = conditioning(torch, c, lab, config['codes'], variant == 'flat')
                anchor = models['anchor'](cond).reshape_as(x1)
                noise = torch.randn_like(x1)
                x0 = noise if variant == 'flat' else .7 * noise + .3 * anchor.detach()
                time = torch.rand(len(x1))
                xt = x0 + time[:, None, None] * (x1 - x0)
                loss_fm = (models['flow'](xt, time, cond) - (x1 - x0)).square().mean()
                one = x0 + models['flow'](x0, torch.zeros(len(x0)), cond)
                loss = (loss_fm + .5 * (one - x1).square().mean() + .1 * (anchor - x1).square().mean()
                        + .1 * torch.nn.functional.cross_entropy(models['selector'](c), lab))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite flow training loss')
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(models.parameters(), 5.); optimizer.step()
                if update % 400 == 0 or update == args.updates:
                    with torch.no_grad():
                        ids = val_idx[:2048]
                        probabilities = models['selector'](context[ids])
                        chosen = probabilities.argmax(-1) if variant != 'flat' else torch.zeros(len(ids), dtype=torch.long)
                        noise = torch.randn((len(ids), config['horizon'], 4), generator=torch.Generator().manual_seed(77))
                        pred = draw(torch, models, context[ids], chosen, noise, flat=variant == 'flat')
                        oracle = draw(torch, models, context[ids], labels[ids], noise, flat=variant == 'flat')
                        shuffled = draw(torch, models, context[ids], labels[ids].roll(1), noise, flat=variant == 'flat')
                    row = dict(update=update, loss=float(loss), autonomous_validation_mse=float((pred - target[ids]).square().mean()),
                               oracle_code_validation_mse=float((oracle - target[ids]).square().mean()),
                               shuffled_oracle_code_mse=float((shuffled - target[ids]).square().mean()),
                               code_accuracy=float((chosen == labels[ids]).float().mean()),
                               parameters=sum(p.numel() for p in models.parameters()))
                    log.write(json.dumps(row) + '\n'); log.flush()
                    print('[train] ' + variant + ' ' + json.dumps(row), flush=True)
            payload = dict(config=dict(config, variant=variant), state_dict=models.state_dict(), context_mean=mean,
                           context_std=std, metrics=row)
            torch.save(payload, args.out / f'{variant}.pt')
            summary[variant] = row
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--features', type=Path, required=True)
    a.add_argument('--runs', type=Path, nargs='+', required=True)
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--validation-body', type=int, default=2034)
    a.add_argument('--horizon', type=int, default=24)
    a.add_argument('--codes', type=int, default=8)
    t = sub.add_parser('train')
    t.add_argument('--data', type=Path, required=True)
    t.add_argument('--out', type=Path, required=True)
    t.add_argument('--seed', type=int, default=20261005)
    t.add_argument('--updates', type=int, default=1600)
    t.add_argument('--width', type=int, default=128)
    args = p.parse_args()
    prepare(args) if args.stage == 'prepare' else train(args)
