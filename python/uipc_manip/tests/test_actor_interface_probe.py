"""Check the update comparison and actual-displacement guard independently of IPC."""
import copy

import torch

from uipc_manip.actor_interface_probe import action, update
from uipc_manip.obs import ObsSpec
from uipc_manip.sac import SACAgent, SACConfig


def fixture():
    torch.manual_seed(91)
    agent = SACAgent(ObsSpec(3), 3, SACConfig(actor_type='state',
        critic_input='privileged', privileged_dim=6, hidden_dim=16,
        state_activation='silu', actor_lr=.01), 'cpu')
    obs, noise = torch.randn(4, 6), torch.randn(4, 3)
    # Nonzero Adam history is part of the comparison contract.
    agent.actor_optimizer.zero_grad()
    agent.actor(obs, noise=noise)[1].square().mean().backward()
    agent.actor_optimizer.step()
    anchor = torch.as_tensor(action(agent, obs, noise))
    gradient = torch.randn(4, 3)
    return agent, obs, noise, anchor, gradient


def test_linear_and_target_first_step_match_with_adam_history():
    agent, obs, noise, anchor, gradient = fixture()
    before = copy.deepcopy(agent.actor.state_dict())
    linear, _ = update(agent, obs, noise, anchor, gradient, 'linear', .03, gradient)
    target, _ = update(agent, obs, noise, anchor, gradient, 'target', .03, gradient)
    torch.testing.assert_close(torch.as_tensor(action(linear, obs, noise)),
                               torch.as_tensor(action(target, obs, noise)), atol=1e-6, rtol=1e-6)
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_guard_restores_optimizer_before_reduced_step():
    agent, obs, noise, anchor, gradient = fixture()
    radius = .001
    bounded, trial = update(agent, obs, noise, anchor, gradient, 'bounded_target', radius, gradient)
    assert trial > 0
    movement = torch.as_tensor(action(bounded, obs, noise))-anchor
    assert float(movement.norm(dim=-1).max()) <= radius
    if trial < 12:
        reduced = copy.deepcopy(agent)
        for group in reduced.actor_optimizer.param_groups:
            group['lr'] *= .5 ** trial
        direct, _ = update(reduced, obs, noise, anchor, gradient, 'target', radius, gradient)
        for key, value in bounded.actor.state_dict().items():
            torch.testing.assert_close(value, direct.actor.state_dict()[key], atol=0, rtol=0)


def test_zero_target_can_move_with_old_moments_but_not_fresh_adam():
    agent, obs, noise, anchor, gradient = fixture()
    old, _ = update(agent, obs, noise, anchor, gradient, 'zero_target', .03, gradient)
    fresh, _ = update(agent, obs, noise, anchor, gradient, 'zero_target_freshadam', .03, gradient)
    assert float((torch.as_tensor(action(old, obs, noise))-anchor).abs().max()) > 1e-5
    torch.testing.assert_close(torch.as_tensor(action(fresh, obs, noise)), anchor, atol=0, rtol=0)
