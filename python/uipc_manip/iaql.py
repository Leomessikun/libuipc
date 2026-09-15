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
    value, gradient, _ = soft_targets_detailed(actor, critic, next_state, reward, mask, tangent,
                                               reward_gradient, alpha, discount, bound, noise)
    return value, gradient


def soft_targets_detailed(actor, critic, next_state, reward, mask, tangent, reward_gradient,
                          alpha, discount, bound, noise=None, continuation_trust_kappa=0.0):
    """``soft_targets`` with the label split into its reward part ``g_R`` and its continuation
    part ``g_C = γ m Dᵀ∇V̄(s')``, and a continuation trust ``c = exp(−κ d²)`` from the twin
    critics' disagreement ``d = |g_C1 − g_C2| / mean(|g_C1|, |g_C2|)`` (each head's own
    continuation through the same next action): the label is ``g_R + c·g_C``. ``κ = 0`` trusts
    the continuation fully and reproduces ``soft_targets``. Returns ``(value, gradient, info)``
    with per-row ``reward_norm``, ``continuation_norm``, ``trust`` and ``disagreement``."""
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
        entropy = torch.as_tensor(alpha).detach()*log_prob
        scale = discount*mask.detach()
        raw = reward.detach() + scale*(torch.minimum(q1, q2)-entropy)
        # Three separate reverse passes: one grad call over several outputs would return the
        # gradient of their sum, not one gradient per head.
        state_gradient = torch.autograd.grad(raw.sum(), ns, retain_graph=True)[0]
        grad1 = torch.autograd.grad((scale*(q1-entropy)).sum(), ns, retain_graph=True)[0]
        grad2 = torch.autograd.grad((scale*(q2-entropy)).sum(), ns)[0]
        pull = lambda grad: torch.einsum("bsa,bs->ba", tangent.detach(), grad)  # noqa: E731
        continuation = pull(state_gradient)
        c1, c2 = pull(grad1), pull(grad2)
        disagreement = (c1-c2).norm(dim=-1, keepdim=True) / (0.5*(c1.norm(dim=-1, keepdim=True)+c2.norm(dim=-1, keepdim=True))+1e-12)
        trust = torch.exp(-float(continuation_trust_kappa)*disagreement.square()) if continuation_trust_kappa > 0 else torch.ones_like(disagreement)
        g_reward = reward_gradient.detach()
        gradient = g_reward + trust*continuation
        inside = (raw.abs() < bound).to(gradient.dtype)
        gradient = gradient * inside
        value = raw.clamp(-bound, bound)
    info = dict(reward_norm=g_reward.norm(dim=-1).detach(), continuation_norm=continuation.norm(dim=-1).detach(),
                trust=trust.reshape(-1).detach(), disagreement=disagreement.reshape(-1).detach())
    return value.detach(), gradient.detach(), info


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
