"""Local response objectives with fixed IPC teachers and bounded normalized actions.

No learned response replaces the simulator in the actor. The finite-difference
objective uses ordinary network backward; the Jacobian control deliberately
retains the mixed derivative graph. Targets are detached in every variant.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_radius(action, direction, epsilon):
    """Largest feasible symmetric radius, up to epsilon, without clipping actions."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not torch.isfinite(action).all() or (action.abs() > 1).any():
        raise ValueError("actions must be finite and normalized to [-1, 1]")
    margin = (1 - action.abs()) / direction.abs().clamp_min(1e-12)
    return margin.amin(-1).clamp(max=epsilon).detach()


def response_objective(predict, action, response, tangent, direction, *,
                       mode="response", epsilon=.05, weight=1.):
    """``predict(a)`` returns B x points x xyz, tangent is B x points x xyz x action.

    ``counterfactual`` is literal symmetric endpoint regression. ``difference``
    normalizes the central difference by its radius, preserving slope weight.
    All coordinates/targets must already share the same response units.
    """
    if mode not in {"response", "jacobian", "counterfactual", "difference"}:
        raise ValueError(f"unknown response objective {mode}")
    response, tangent, direction = response.detach(), tangent.detach(), direction.detach()
    nominal = predict(action)
    value_loss = F.mse_loss(nominal, response)
    differential = value_loss.new_zeros(())
    radius = symmetric_radius(action, direction, epsilon)
    target = torch.einsum("bpca,ba->bpc", tangent, direction)
    live = radius > 1e-7
    if mode == "jacobian":
        _, jvp = torch.autograd.functional.jvp(predict, action, direction, create_graph=True)
        differential = F.mse_loss(jvp, target)
    elif mode != "response" and live.any():
        shift = radius[:, None] * direction
        plus, minus = predict(action + shift), predict(action - shift)
        r = radius[live, None, None]
        if mode == "counterfactual":
            differential = .5 * (F.mse_loss(plus[live], response[live] + r * target[live])
                                  + F.mse_loss(minus[live], response[live] - r * target[live]))
        else:
            differential = F.mse_loss((plus[live] - minus[live]) / (2 * r), target[live])
    return value_loss + weight * differential, {
        "response_loss": value_loss.detach(), "differential_loss": differential.detach(),
        "radius_mean": radius.mean(), "valid_fraction": live.float().mean(),
    }
