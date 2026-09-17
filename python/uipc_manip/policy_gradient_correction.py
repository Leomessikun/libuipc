"""Mathematical prototype for selectively evaluated branch-return corrections.

This is NOT wired into SAC collection or training. The likelihood-ratio/control-
variate identity is established prior work (Liu et al., ICLR 2018), as is inverse-
probability debiasing. See the research note for the proposed allocation problem
and assumptions; these functions alone are not a novel RL algorithm.
"""
import torch


def branch_return_residual(values, observed, survival):
    """Return sum_l I_l/P_l * (Y_l-Y_{l-1}) for nested branch prefixes.

    values: [batch, levels+1], starting with frozen Q(s,a), then progressively
    longer return estimates; unobserved values may be NaN. observed/survival:
    [batch, levels]. survival contains *marginal reach probabilities*, including
    root query selection, not just the latest conditional continuation coin.

    A collector must choose each continuation probability before seeing that
    segment's outcome and retain positive support. No success filtering, weight
    clipping or selected-row normalization preserves this identity in general.
    All values/probabilities are detached from the policy gradient.
    """
    if (values.ndim != 2 or values.shape[1] < 2 or observed.shape != values[:, 1:].shape
            or survival.shape != observed.shape or observed.dtype != torch.bool
            or not values.is_floating_point() or not survival.is_floating_point()
            or observed.device != values.device or survival.device != values.device):
        raise ValueError("Invalid branch value/mask/probability layout")
    if not torch.isfinite(values[:, 0]).all():
        raise ValueError("Root control-variate values must be finite")
    if (observed[:, 1:] & ~observed[:, :-1]).any():
        raise ValueError("Observed branch levels must form a prefix")
    if (not torch.isfinite(values[:, 1:][observed]).all()
            or not torch.isfinite(survival[observed]).all()
            or (survival[observed] <= 0).any() or (survival[observed] > 1).any()):
        raise ValueError("Observed returns need finite values and probabilities in (0,1]")
    if ((survival[:, 1:] > survival[:, :-1]) & observed[:, 1:]).any():
        raise ValueError("Marginal survival probabilities cannot increase")
    values, survival = values.detach(), survival.detach()
    increment = torch.where(observed, values[:, 1:] - values[:, :-1], 0)
    probability = torch.where(observed, survival, 1)
    return (increment / probability).sum(dim=1)


def branch_score_loss(log_prob_fixed_action, values, observed, survival):
    """Loss to ADD to the existing SAC actor loss, averaged over all roots.

    log_prob_fixed_action must evaluate a detached action drawn by the current
    frozen policy, NOT its reparameterized action with a live sampling graph.
    Returns must use that same frozen policy for continuation and a consistent
    soft-return/time-limit contract. This estimates only the fixed-root,
    fixed-continuation improvement surrogate; it is no full-task guarantee.
    """
    residual = branch_return_residual(values, observed, survival)
    if log_prob_fixed_action.shape != residual.shape:
        raise ValueError("One scalar action log probability is required per root")
    if not torch.isfinite(log_prob_fixed_action).all():
        raise ValueError("Action log probabilities must be finite")
    return -(log_prob_fixed_action * residual).mean()
