"""Physical-effect partitions and matched flow proposal models.

Effect coordinates supervise training only. They are never actor observations.
The codebook, generative loss and pilot are not established novelty claims.
"""
from __future__ import annotations

import numpy as np

from motion_observation import history_inputs

ACTION_AXES = [0, 1, 2, 5]
VARIANTS = ('flat', 'action', 'effect', 'shuffled_effect')


def effect_vector(descriptors, valid=None):
    """Describe measured changes, including three-state persistent wrap events.

    Ring coordinates are normalized by arm length. Radius is an opening-size
    proxy, not measured clearance or a comfort quantity. No motion after the
    supplied chunk enters the label. Isolated wrapping flicker is ignored.
    """
    d = np.asarray(descriptors, np.float32)
    if d.ndim != 3 or d.shape[1:] != (4, 9) or len(d) < 4 or not np.isfinite(d).all():
        raise ValueError('Need at least four finite canonical descriptor states')
    initial = d[0, :, 8] > .5
    flags = initial.copy()
    gained, lost = np.zeros(4, bool), np.zeros(4, bool)
    for t in range(2, len(d)):
        recent = d[t - 2:t + 1, :, 8] > .5
        update = np.where(recent.all(0), True, np.where((~recent).all(0), False, flags))
        gained |= update & ~flags
        lost |= ~update & flags
        flags = update
    geometric = np.r_[(d[-1, :, :3] - d[0, :, :3]).reshape(-1) / .05,
                       (d[-1, :, 6] - d[0, :, 6]) / .02,
                       (1. - np.sum(d[-1, :, 3:6] * d[0, :, 3:6], axis=1)) / .2]
    return np.r_[np.clip(geometric, -5., 5.),
                 4. * (flags.astype(float) - initial.astype(float)),
                 4. * gained, 4. * lost,
                 4. * float(True if valid is None else np.all(valid))].astype(np.float32)


def context_inputs(features, tools, actions, bases, rotation, frames=4):
    """Causal point/tool history, executed past actions and current nominal action."""
    history = history_inputs(features, np.asarray(tools) @ rotation.T, frames)
    past = np.zeros((len(features), (frames - 1) * 4), np.float32)
    for t in range(len(features)):
        start = max(0, t - frames + 1)
        values = np.asarray(actions)[start:t, ACTION_AXES].reshape(-1)
        if len(values):
            past[t, -len(values):] = values
    return np.concatenate((history, past, np.asarray(bases)[:, ACTION_AXES]), axis=1).astype(np.float32)


def nearest(values, centers):
    values, centers = np.asarray(values), np.asarray(centers)
    return np.argmin(((values[:, None] - centers[None]) ** 2).sum(-1), axis=1)


def kmeans(values, count=8, seed=7, iterations=40):
    """Small dependency-free k-means++ used only on declared training rows."""
    x = np.asarray(values, np.float32)
    if len(x) < count or not np.isfinite(x).all():
        raise ValueError('Insufficient finite training examples for the codebook')
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(count - 1):
        dist = ((x[:, None] - np.asarray(centers)[None]) ** 2).sum(-1).min(1)
        if dist.sum() < 1e-9:
            raise ValueError('Fewer distinct effects than requested codes')
        centers.append(x[rng.choice(len(x), p=dist / dist.sum())])
    centers = np.asarray(centers)
    for _ in range(iterations):
        labels = nearest(x, centers)
        updated = np.stack([x[labels == k].mean(0) if np.any(labels == k) else centers[k]
                            for k in range(count)])
        if np.max(np.abs(updated - centers)) < 1e-5:
            break
        centers = updated
    return centers.astype(np.float32)


def build_models(torch, context_dim, horizon=8, codes=8, width=128, depth=2):
    """Reuse the existing flow backbone; all four variants have identical shapes."""
    from uipc_manip.flow_policy import build_net
    nn = torch.nn
    selector = nn.Sequential(nn.Linear(context_dim, width), nn.GELU(), nn.Linear(width, codes))
    anchor = nn.Sequential(nn.Linear(context_dim + codes, width), nn.GELU(),
                           nn.Linear(width, horizon * 4))
    flow = build_net(torch, horizon, width, depth)
    flow.cond[0] = nn.Linear(context_dim + codes, width)
    return nn.ModuleDict(dict(selector=selector, anchor=anchor, flow=flow))


def conditioning(torch, context, labels, codes, flat=False):
    z = torch.nn.functional.one_hot(labels.long(), codes).to(context.dtype)
    if flat:
        z = z * 0.
    return torch.cat((context, z), dim=1)


def draw(torch, models, context, labels, noise, *, flat=False, steps=4):
    codes = models['selector'][-1].out_features
    cond = conditioning(torch, context, labels, codes, flat)
    anchor = models['anchor'](cond).reshape_as(noise)
    x = noise if flat else .7 * noise + .3 * anchor
    for k in range(steps):
        t = torch.full((len(x),), k / steps, dtype=x.dtype, device=x.device)
        x = x + models['flow'](x, t, cond) / steps
    return x


class EffectModels:
    def __init__(self, directory):
        import torch
        from pathlib import Path
        from dynamic_student import sha256
        self.torch = torch
        self.models = {}
        for variant in VARIANTS:
            data = torch.load(str(Path(directory) / f'{variant}.pt'), map_location='cpu', weights_only=False)
            cfg = data['config']
            if sha256(Path(__file__)) != cfg['module_sha256']:
                raise ValueError('Effect-model implementation differs from trained checkpoint')
            model = build_models(torch, cfg['context_dim'], cfg['horizon'], cfg['codes'], cfg['width'], cfg['depth'])
            model.load_state_dict(data['state_dict']); model.eval()
            self.models[variant] = model
            self.config = cfg
            self.mean, self.std = np.asarray(data['context_mean']), np.asarray(data['context_std'])

    def sample(self, variant, context, rank, seed):
        torch = self.torch
        model = self.models[variant]
        x = torch.as_tensor(((context - self.mean) / self.std)[None], dtype=torch.float32)
        with torch.no_grad():
            probabilities = model['selector'](x).softmax(-1)[0]
            order = torch.argsort(probabilities, descending=True)
            code = int(order[rank % len(order)]) if variant != 'flat' else 0
            noise = torch.randn((1, self.config['horizon'], 4),
                                generator=torch.Generator().manual_seed(int(seed)))
            actions = draw(torch, model, x, torch.tensor([code]), noise, flat=variant == 'flat')
        return actions[0].numpy(), code, probabilities.numpy()
