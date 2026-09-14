"""CPU check of the fine-tuning losses: the physics term's gradient on a linear actor is the stored
direction, scaled by β, and the SAC term goes through the frozen critic."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip import physics_gradient_finetune as ft  # noqa: E402


class LinearActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(6))

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        b = obs[0].shape[0]
        mu = self.w.unsqueeze(0).expand(b, 6)
        return mu, mu, torch.zeros(b, 1), torch.zeros(b, 6)


class ZeroCritic(torch.nn.Module):
    def forward(self, obs, action):
        q = (action * action).sum(dim=-1, keepdim=True)  # Q = |a|², so dQ/da = 2a
        return q, q


def test_physics_loss_gradient_is_the_stored_direction():
    agent = SimpleNamespace(critic=ZeroCritic(), _alpha_at=lambda i: torch.zeros(()))
    actor = LinearActor()
    g = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    obs = (torch.zeros(2, 4, 3), None, None, None)
    sac, phys = ft.actor_losses(agent, actor, obs, g, beta=2.0)
    phys.backward()
    # -β · mean over the batch of g · μ: dμ/dw = I, so the gradient is -β · mean(g).
    assert torch.allclose(actor.w.grad, -2.0 * g.mean(dim=0))
    actor.w.grad = None
    with torch.no_grad():
        actor.w[:] = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    sac, _ = ft.actor_losses(agent, actor, obs, g, beta=1.0)
    sac.backward()
    # -Q(s, π): dQ/da = 2a at a = w, so the gradient is -2w.
    assert torch.allclose(actor.w.grad, torch.tensor([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_grad_norm_of_leaves_no_gradient_behind():
    agent = SimpleNamespace(critic=ZeroCritic(), _alpha_at=lambda i: torch.zeros(()))
    actor = LinearActor()
    g = torch.ones(3, 6)
    obs = (torch.zeros(3, 4, 3), None, None, None)
    _, phys = ft.actor_losses(agent, actor, obs, g, beta=1.0)
    assert ft.grad_norm_of(actor, phys) == pytest.approx(float(np.sqrt(6.0)))
    assert actor.w.grad is None
