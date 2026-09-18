"""Reversing the behavior flow: the inverse of the forward Euler scheme, and its statistics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uipc_manip import flow_reversal as fr


class LinearFlow:
    """A behavior flow whose velocity field is a fixed linear map of the state and time.

    With ``vector(encoded, x, t) = gain * x + bias``, forward Euler has a closed form,
    so the reverse pass can be checked against arithmetic rather than against itself.
    """

    def __init__(self, gain=0.3, bias=0.1, steps=10, action_dim=6, device="cpu"):
        self.cfg = SimpleNamespace(flow_steps=steps)
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.behavior = SimpleNamespace(vector=lambda encoded, x, time: gain * x + bias)
        self.gain, self.bias, self.steps = gain, bias, steps

    @torch.no_grad()
    def flow_actions(self, encoded, noise):
        x = noise
        for k in range(self.steps):
            time = x.new_full((len(x), 1), k / self.steps)
            x = x + self.behavior.vector(encoded, x, time) / self.steps
        return x.clamp(-1, 1)


def test_reverse_flow_undoes_the_forward_scheme_up_to_its_integration_error():
    agent = LinearFlow()
    encoded = torch.zeros(4, 1)
    noise = 0.2 * torch.randn(4, 6, generator=torch.Generator().manual_seed(0))
    forward = agent.flow_actions(encoded, noise)
    recovered = fr.reverse_flow(agent, encoded, forward)
    # Explicit reverse of an explicit forward step: close, but not the identity.
    assert torch.allclose(recovered, noise, atol=2e-2)
    assert not torch.allclose(recovered, noise, atol=1e-6)


def test_the_integration_error_shrinks_as_the_scheme_is_refined():
    encoded = torch.zeros(3, 1)
    noise = torch.full((3, 6), 0.3)
    errors = []
    for steps in (5, 20, 80):
        agent = LinearFlow(steps=steps)
        forward = agent.flow_actions(encoded, noise)
        errors.append(float((fr.reverse_flow(agent, encoded, forward) - noise).norm()))
    assert errors[0] > errors[1] > errors[2]


def test_a_larger_action_needs_a_larger_noise_under_a_contracting_field():
    agent = LinearFlow(gain=0.0, bias=0.0)   # the identity flow: noise equals action
    encoded = torch.zeros(2, 1)
    small = fr.reverse_flow(agent, encoded, torch.full((1, 6), 0.1))
    large = fr.reverse_flow(agent, encoded, torch.full((1, 6), 0.9))
    assert float(large.norm()) > float(small.norm())


def test_chi_percentile_matches_the_standard_normal_length():
    # The median length of a six-dimensional standard normal is about 2.31.
    median = fr.chi_percentile(np.array([2.313]), 6, samples=50_000, seed=1)[0]
    assert 0.45 < median < 0.55
    assert fr.chi_percentile(np.array([0.2]), 6, samples=50_000, seed=1)[0] < 0.01
    assert fr.chi_percentile(np.array([6.0]), 6, samples=50_000, seed=1)[0] > 0.99


def test_reversal_report_returns_one_row_per_pair_and_rejects_a_mismatch():
    agent = LinearFlow()
    observations = np.zeros((5, 1), dtype=np.float32)
    actions = 0.2 * np.ones((5, 6), dtype=np.float32)
    agent.unpack = lambda flat: flat
    agent.behavior._frame_latent = lambda obs: torch.zeros(len(obs), 1)
    report = fr.reversal_report(agent, observations, actions, batch=2)
    for key in ("latent", "latent_norm", "latent_percentile", "reconstruction_error", "noise_round_trip"):
        assert len(report[key]) == 5
    assert (report["reconstruction_error"] >= 0).all()
    with pytest.raises(ValueError):
        fr.reversal_report(agent, observations, actions[:3])


def test_summarize_reports_the_tail_fraction():
    report = dict(latent_norm=np.array([1.0, 2.0, 9.0]), latent_percentile=np.array([0.2, 0.5, 0.999]),
                  reconstruction_error=np.array([0.01, 0.02, 0.03]), noise_round_trip=np.array([0.001] * 3),
                  action_norm=np.array([0.1, 0.2, 0.3]))
    s = fr.summarize(report)
    assert s["count"] == 3
    assert s["fraction_above_99th"] == pytest.approx(1 / 3)
    assert s["latent_norm"]["median"] == pytest.approx(2.0)
