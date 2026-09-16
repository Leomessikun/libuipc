"""Fresh same-action policy improvement, independent of the TD replay size."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uipc_manip.obs import ObsSpec
from uipc_manip.sac import SACAgent, SACConfig


def agent(rho=.5, weight=1., trunk="plain", max_step=0.0):
    torch.manual_seed(42)
    return SACAgent(ObsSpec(3), 3, SACConfig(
        actor_type="state", critic_input="privileged", privileged_dim=4, hidden_dim=16,
        batch_size=8, state_activation="silu", trunk_style=trunk, alpha_fixed=True,
        physics_actor_mode="mix", physics_actor_weight=weight, physics_actor_rho=rho,
        physics_actor_sigma=.5, physics_actor_max_action_step=max_step), "cpu")


def batch(learner):
    obs = torch.randn(8, 4)
    action, noise = learner.sample_state_action(obs)
    reward_gradient = torch.randn_like(action)
    return obs, action, noise, torch.zeros(8, 1), obs + .1, torch.ones(8, 1), reward_gradient


@pytest.mark.parametrize("rho", [0., .5, 1.])
@pytest.mark.parametrize("trunk", ["plain", "residual"])
def test_fresh_parameter_gradient_is_the_requested_mixture(rho, trunk):
    learner = agent(rho, trunk=trunk)
    obs, action, noise, reward, nxt, mask, g = batch(learner)
    critic_before = copy.deepcopy(learner.critic.state_dict())
    _, pi, logpi, _ = learner.actor(obs, noise=noise)
    q1, q2 = learner.critic(obs, pi)
    expected_loss = (learner.alpha.detach() * logpi - (1-rho)*torch.minimum(q1, q2)
                     - rho*(g*pi).sum(-1, keepdim=True)).mean()
    expected = torch.autograd.grad(expected_loss, tuple(learner.actor.parameters()))
    stats = learner.update_fresh_state_actor(obs, action, noise, reward, nxt, mask,
                                            reward_gradient=g, signal="reward")
    for p, grad in zip(learner.actor.parameters(), expected):
        torch.testing.assert_close(p.grad, grad, atol=1e-6, rtol=1e-5)
    for key, value in learner.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)
    assert learner.updates == 0  # actor steps do not advance the critic/target clock
    assert stats["actor_action_distance_max"] < 1e-6
    assert stats["physics_effective_rho"] == pytest.approx(rho)
    if rho > 0:
        assert stats["physics_loss"] == pytest.approx(0, abs=1e-7)
        assert stats["physics_correction_norm"] > 0


def test_stale_or_resampled_actions_are_refused_before_an_optimizer_step():
    learner = agent()
    obs, action, noise, reward, nxt, mask, g = batch(learner)
    before = copy.deepcopy(learner.actor.state_dict())
    with pytest.raises(ValueError, match="differs from its anchor"):
        learner.update_fresh_state_actor(obs, action, noise+1, reward, nxt, mask,
                                        reward_gradient=g, signal="reward")
    for k, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, before[k], atol=0, rtol=0)


def test_td_updates_do_not_update_the_fresh_actor_or_require_sidecar_rows():
    learner = agent()
    obs, action, _, reward, nxt, mask, _ = batch(learner)
    before = copy.deepcopy(learner.actor.state_dict())
    for _ in range(4):
        stats = learner.update_state_batch(obs, action, reward, nxt, mask, update_actor=False)
        assert "actor_loss" not in stats
    assert learner.updates == 4
    for k, value in learner.actor.state_dict().items():
        torch.testing.assert_close(value, before[k], atol=0, rtol=0)


def test_replay_locality_uses_sampled_squashed_action():
    learner = agent()
    obs, action, noise, _, _, _, g = batch(learner)
    # Shift the unsquashed means beyond the action bounds. The saved action still matches pi.
    with torch.no_grad():
        learner.actor.trunk[-1].bias[:3].fill_(3.)
    _, action, _, _ = learner.actor(obs, noise=noise)
    action = action.detach()
    stats = learner._update_actor_and_alpha(obs, state=obs, physics=(g, torch.ones(8)),
                                           action_noise=noise, action_anchor=action)
    assert stats["physics_mix_weight"] == pytest.approx(1.)
    assert stats["physics_effective_rho"] == pytest.approx(.5)


def test_fresh_update_rolls_back_and_bounds_actual_action_step():
    learner = agent(max_step=.001)
    obs, action, noise, reward, nxt, mask, g = batch(learner)
    stats = learner.update_fresh_state_actor(obs, action, noise, reward, nxt, mask,
                                            reward_gradient=100*g, signal="reward")
    assert stats["physics_actual_step_max"] <= .001 + 1e-7


@pytest.mark.parametrize("control", ["random", "zero", "negative"])
def test_controls_transform_teacher_before_correction(control, monkeypatch):
    learner = agent()
    obs, action, noise, reward, nxt, mask, g = batch(learner)
    captured = {}
    def capture(*args, **kwargs):
        captured["g"] = kwargs["physics"][0]
        return {}
    monkeypatch.setattr(learner, "_update_actor_and_alpha", capture)
    learner.update_fresh_state_actor(obs, action, noise, reward, nxt, mask, reward_gradient=g,
                                    signal="reward", control=control,
                                    control_generator=torch.Generator().manual_seed(4))
    got = captured["g"]
    if control == "random":
        torch.testing.assert_close(got.norm(dim=-1), g.norm(dim=-1))
        assert not torch.allclose(got, g)
    else:
        torch.testing.assert_close(got, -g if control == "negative" else torch.zeros_like(g))


class AffineEnv:
    """Small deterministic dynamics for exercising the real online loop without IPC."""
    N, obs_dim = 4, 4

    def reset(self, seed):
        self.obs = np.random.default_rng(seed).normal(0, .05, (self.N, self.obs_dim))
        self.steps = 0
        return self.obs.copy()

    def step(self, actions, capture=False):
        self.obs[:, :3] += .1*np.asarray(actions)
        self.steps += 1
        distance = np.linalg.norm(self.obs[:, :3], axis=1)
        out = dict(obs=self.obs.copy(), reward=-distance**2, distance=distance,
                   success=distance < .05, done=self.steps >= 8)
        if capture:
            tangent = np.zeros((self.N, self.obs_dim, 3))
            tangent[:, :3, :] = .1*np.eye(3)
            out.update(tangent=tangent, reward_gradient=-.2*self.obs[:, :3].copy())
        return out


def test_online_fresh_coverage_and_actor_clock_do_not_depend_on_sidecar_capacity(tmp_path, monkeypatch):
    from uipc_manip import iaql_benchmark as bench
    monkeypatch.setattr(bench, "AGENT_OPTIONS", dict(physics_actor_mode="mix", physics_actor_rho=.5,
                                                    physics_actor_sigma=.5, batch_size=8))
    monkeypatch.setattr(bench, "AGENT_DEVICE", "cpu")
    results = []
    for capacity in (1, 1024):
        args = SimpleNamespace(actor_batch="fresh", eval_episodes=4, online_steps=128, warmup_transitions=64,
            online_weights=[0.], seed=0, actor_weight=1., actor_rho=.5, tangent_rows=capacity,
            replay_rows=0, updates_per_step=1., shuffle_labels=False, reward_mode="dense",
            physics_signal="bellman", physics_control="paired", eval_every=0, eval_steps=2,
            out=tmp_path/str(capacity))
        result = bench.online(AffineEnv(), args)[0]
        assert result["actor_updates"] == 16
        assert result["critic_updates"] == 68
        assert result["sidecar_rows"] == 0  # actor-only mechanics never enter TD replay
        assert result["replay_rows"] == 128
        assert result["stats"]["physics_valid_fraction"] == 1.
        assert result["stats"]["physics_effective_rho"] == pytest.approx(.5)
        results.append(result)
    assert results[0]["evaluation"] == results[1]["evaluation"]


def test_terminal_evaluation_cannot_stop_before_the_reward(tmp_path):
    from uipc_manip.iaql_benchmark import main
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path), "--reward-mode", "terminal", "--eval-steps", "50"])
