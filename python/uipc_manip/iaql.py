"""Paired SAC Bellman values/slopes and derivative regression, independent of IPC."""
from __future__ import annotations

import torch

from .models import gaussian_logprob, squash


def soft_targets(actor, critic, next_state, reward, mask, tangent, reward_gradient,
                 alpha, discount, bound, noise=None):
    """Refresh a scalar/slope pair from a teacher-independent action tangent.

    ``tangent[B,S,A]`` includes all successor state/action paths and uses the
    same units as ``next_state``. Reward derivatives are total, per normalized
    current action. Teacher input derivatives remain live until labels detach.
    """
    from .sac import frozen_parameters

    b, s = next_state.shape
    a = reward_gradient.shape[-1]
    if tangent.shape != (b, s, a) or reward_gradient.shape != (b, a):
        raise ValueError("Incompatible state/action tangent shapes")
    if reward.shape != (b, 1) or mask.shape != (b, 1):
        raise ValueError("Reward and bootstrap mask must be [B,1]")
    with torch.enable_grad(), frozen_parameters(actor), frozen_parameters(critic):
        ns = next_state.detach().requires_grad_(True)
        mu, log_std = actor.head(ns)
        noise = torch.randn_like(mu) if noise is None else noise.detach()
        _, action, log_prob = squash(mu, mu + log_std.exp()*noise, gaussian_logprob(noise, log_std))
        q1, q2 = critic(ns, action)
        raw = reward.detach() + discount*mask.detach()*(torch.minimum(q1, q2)-torch.as_tensor(alpha).detach()*log_prob)
        state_gradient = torch.autograd.grad(raw.sum(), ns)[0]
        gradient = reward_gradient.detach() + torch.einsum("bsa,bs->ba", tangent.detach(), state_gradient)
        inside = (raw.abs() < bound).to(gradient.dtype)
        gradient = gradient * inside
        value = raw.clamp(-bound, bound)
    return value.detach(), gradient.detach()


def derivative_loss(qs, action, target_gradient, valid, scale=1.0):
    """Both scalar Q heads learn signed slopes; invalid rows contribute zero.

    Mask before arithmetic so invalid NaN labels cannot contaminate the loss.
    Smooth zero targets are valid. Normalization is over the entire batch.
    """
    if scale <= 0 or not torch.isfinite(torch.as_tensor(scale)):
        raise ValueError("Gradient scale must be finite and positive")
    if target_gradient.shape != action.shape or valid.shape not in ((len(action),), (len(action), 1)):
        raise ValueError("Gradient labels/masks have incompatible shapes")
    keep = valid.reshape(-1, 1).bool()
    target = torch.where(keep, target_gradient.detach(), torch.zeros_like(target_gradient))
    if not torch.isfinite(target).all():
        raise ValueError("A valid derivative label is nonfinite")
    loss = action.new_zeros(())
    for q in qs:
        pred = torch.autograd.grad(q.sum(), action, create_graph=True, retain_graph=True)[0]
        delta = torch.where(keep, pred-target, torch.zeros_like(pred))
        loss = loss + (delta/scale).square().mean()
    return loss
