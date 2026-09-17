import math

import numpy as np
import pytest
import torch

from uipc_manip.policy_gradient_correction import branch_return_residual, branch_score_loss


def test_nested_query_weighting_recovers_full_return_even_when_most_branches_stop():
    # All three possible outcomes of two Bernoulli continuation decisions.
    probabilities = torch.tensor([.8, .2 * .7, .2 * .3], dtype=torch.float64)
    values = torch.tensor([[2., float('nan'), float('nan')], [2., 5., float('nan')], [2., 5., 11.]])
    observed = torch.tensor([[False, False], [True, False], [True, True]])
    survival = torch.tensor([[.2, .06]] * 3)
    correction = branch_return_residual(values, observed, survival).double()
    assert float((probabilities * correction).sum()) == pytest.approx(11 - 2, abs=1e-6)
    naive = torch.tensor([0., 3., 9.], dtype=torch.float64)
    assert float((probabilities * naive).sum()) != pytest.approx(11 - 2)


def test_score_correction_recovers_gradient_across_discontinuous_reward_with_wrong_critic():
    # Deterministic quadrature, split at the discontinuity. This is a mathematical
    # identity check, not a cloth simulation or an RL benchmark result.
    theta0, sigma, threshold = .2, .7, .25
    z_cut = (threshold - theta0) / sigma
    nodes, weights = np.polynomial.legendre.leggauss(96)
    z, w = [], []
    for low, high in [(-10, z_cut), (z_cut, 10)]:
        part = low + (nodes + 1) * (high - low) / 2
        z.extend(part)
        w.extend(weights * (high - low) / 2 * np.exp(-part**2 / 2) / math.sqrt(2 * math.pi))
    z, weights = torch.tensor(z), torch.tensor(w)
    theta = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    action = theta + sigma * z
    frozen_action = action.detach()
    q = -action  # Its gradient points in exactly the wrong direction.
    target = (frozen_action > threshold).double()
    log_prob = torch.distributions.Normal(theta, sigma).log_prob(frozen_action)
    values = torch.stack((q.detach(), target), dim=1)
    residual = branch_return_residual(values, torch.ones((len(z), 1), dtype=torch.bool),
                                     torch.ones((len(z), 1)))
    objective = (weights * (q + residual * log_prob)).sum()
    estimate = torch.autograd.grad(objective, theta)[0].item()
    exact = math.exp(-z_cut**2 / 2) / (sigma * math.sqrt(2 * math.pi))
    assert estimate == pytest.approx(exact, abs=1e-10)
    assert estimate > 0


def test_loss_preserves_gradient_only_through_fixed_action_log_probability():
    values = torch.tensor([[1., 3.], [2., float('nan')]], requires_grad=True)
    survival = torch.tensor([[.5], [.5]], requires_grad=True)
    log_prob = torch.tensor([-.3, -.4], requires_grad=True)
    loss = branch_score_loss(log_prob, values, torch.tensor([[True], [False]]), survival)
    loss.backward()
    assert values.grad is None and survival.grad is None
    torch.testing.assert_close(log_prob.grad, torch.tensor([-2., 0.]))


@pytest.mark.parametrize("mask,prob", [([False, True], [.5, .25]), ([True, True], [.2, .5]),
                                      ([True, False], [0., .2]), ([True, True], [.5, float('nan')])])
def test_invalid_selection_records_are_rejected(mask, prob):
    with pytest.raises(ValueError):
        branch_return_residual(torch.tensor([[1., 2., 3.]]), torch.tensor([mask]), torch.tensor([prob]))
