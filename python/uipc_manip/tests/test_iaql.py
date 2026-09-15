"""Numerical Bellman and learner contracts; these tests need no simulator/GPU."""
import copy

import numpy as np
import pytest
import torch

from uipc_manip.iaql import derivative_loss, soft_targets
from uipc_manip.models import PrivilegedCritic, StateActor, gaussian_logprob, squash
from uipc_manip.obs import ObsSpec
from uipc_manip.sac import SACAgent, SACConfig


def test_soft_target_matches_full_path_derivative_and_finite_difference():
    torch.manual_seed(4)
    actor = StateActor(4, 2, 16, -3, 1).double()
    critic = PrivilegedCritic(4, 2, 16).double()
    base = torch.randn(2, 4, dtype=torch.double)
    tangent = torch.randn(2, 4, 2, dtype=torch.double)*.2
    u = torch.tensor([[.1, -.2], [.3, .1]], dtype=torch.double)
    ns = base + torch.einsum("bsa,ba->bs", tangent, u)
    reward = -u.square().sum(-1, keepdim=True)
    mask = torch.tensor([[1.], [0.]], dtype=torch.double)
    noise = torch.randn(2, 2, dtype=torch.double)
    def full(u):
        state = base + torch.einsum("bsa,ba->bs", tangent, u)
        mu, ls = actor.head(state)
        _, ap, lp = squash(mu, mu+ls.exp()*noise, gaussian_logprob(noise, ls))
        q1, q2 = critic(state, ap)
        return -u.square().sum(-1, keepdim=True)+.93*mask*(torch.minimum(q1, q2)-.2*lp)
    y, g = soft_targets(actor, critic, ns, reward, mask, tangent, -2*u, .2, .93, 100, noise)
    variable = u.clone().requires_grad_(True)
    true_g = torch.autograd.grad(full(variable).sum(), variable)[0]
    torch.testing.assert_close(y, full(u))
    torch.testing.assert_close(g, true_g)
    for axis in range(2):
        delta = torch.zeros_like(u)
        delta[:, axis] = 1e-5
        fd = ((full(u+delta)-full(u-delta))/(2e-5)).squeeze(-1)
        torch.testing.assert_close(g[:, axis], fd, rtol=1e-5, atol=1e-7)
    assert not y.requires_grad and not g.requires_grad
    assert all(p.grad is None and p.requires_grad for m in (actor, critic) for p in m.parameters())
    clipped_y, clipped_g = soft_targets(actor, critic, ns, reward, mask, tangent, -2*u, .2, .93, .001, noise)
    assert (clipped_y.abs() <= .001).all()
    assert torch.equal(clipped_g, torch.zeros_like(clipped_g))


def test_derivative_loss_mixed_parameter_gradient_and_invalid_zero_labels():
    a = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    w = torch.tensor([.2, -.4], dtype=torch.double, requires_grad=True)
    labels = torch.tensor([[0., 0.], [float("nan"), float("nan")], [1., -1.]], dtype=torch.double)
    valid = torch.tensor([True, False, True])
    q = (a*w).sum(-1, keepdim=True)
    loss = derivative_loss((q, 2*q), a, labels, valid)
    expected = sum(((factor*w-labels[i])**2).sum() for factor in [1, 2] for i in [0, 2])/6
    actual_grad = torch.autograd.grad(loss, w, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, w)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    invalid_loss = derivative_loss((q, 2*q), a, labels, torch.zeros(3, dtype=torch.bool))
    assert float(invalid_loss.detach()) == 0
    with pytest.raises(ValueError, match="nonfinite"):
        derivative_loss((q, 2*q), a, labels, torch.ones(3, dtype=torch.bool))


def make_agent(weight=0, activation="relu"):
    return SACAgent(ObsSpec(3), 2, SACConfig(actor_type="state", critic_input="privileged", privileged_dim=4,
        hidden_dim=16, batch_size=8, adjoint_weight=weight, actor_update_freq=2, state_activation=activation), "cpu")


@pytest.mark.parametrize("activation", ["relu", "silu"])
def test_state_learner_refresh_and_checkpoint(tmp_path, activation):
    torch.manual_seed(3)
    agent = make_agent(.1, activation)
    state = torch.randn(8, 4)
    action = torch.rand(8, 2)*2-1
    tangent = torch.randn(8, 4, 2)*.1
    nxt = state+torch.einsum("bsa,ba->bs", tangent, action)
    for _ in range(4):
        stats = agent.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1),
            tangent=tangent, reward_gradient=torch.zeros(8, 2), valid=torch.ones(8))
        assert np.isfinite(stats["critic_loss"]) and "adjoint_loss" in stats
    assert "actor_loss" in stats
    path = agent.save(tmp_path/"model.pt", 4)
    other = make_agent(.1, activation)
    other.load(path)
    np.testing.assert_array_equal(agent.act(state.numpy(), True), other.act(state.numpy(), True))
    with pytest.raises(ValueError, match="mechanics"):
        agent.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1))


def test_zero_weight_matches_existing_scalar_update_exactly():
    torch.manual_seed(6)
    agent = make_agent(0)
    other = copy.deepcopy(agent)
    state, action = torch.randn(8, 4), torch.randn(8, 2)
    r, mask = torch.randn(8, 1), torch.ones(8, 1)
    torch.manual_seed(77)
    first = agent.update_state_batch(state, action, r, state, mask)
    torch.manual_seed(77)
    second = other._update_critic(state, action, r, state, mask, state, state)
    other._finish_update(second, state, state)
    assert first["critic_loss"] == second["critic_loss"]
    for a, b in zip(agent.critic.parameters(), other.critic.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sidecar_batch_marks_rows_without_mechanics_invalid():
    from uipc_manip.iaql_benchmark import sidecar_batch
    rows = [dict(tangent=np.ones((4, 3), np.float32), reward_gradient=np.ones(3)), dict(obs=None)]
    kw = sidecar_batch(rows, 4)
    assert kw["valid"].tolist() == [1.0, 0.0]
    assert kw["tangent"].shape == (2, 4, 3) and float(kw["tangent"][1].abs().sum()) == 0
    assert kw["reward_gradient"].shape == (2, 3) and float(kw["reward_gradient"][1].abs().sum()) == 0


def test_refit_sweeps_weights_and_shuffles_only_training_labels(tmp_path):
    import types
    from uipc_manip.iaql_benchmark import refit
    rng = np.random.default_rng(0)
    n = 64
    np.savez_compressed(tmp_path/"fixed_dataset.npz", obs=rng.normal(size=(n, 4)).astype(np.float32),
                        actions=rng.uniform(-1, 1, (n, 3)).astype(np.float32), targets=rng.normal(size=(n, 1)).astype(np.float32),
                        gradients=rng.normal(size=(n, 3)).astype(np.float32), episodes=np.arange(n)//16)
    results = refit(types.SimpleNamespace(out=tmp_path, seed=0, fit_updates=2, refit_weights=[0.0, 0.1]))
    assert [(r["weight"], r["labels"]) for r in results] == [(None, "train_mean_slope"), (0.0, "paired"), (0.1, "paired"), (0.1, "shuffled")]
    assert all(np.isfinite(h["slope_mse"]) for r in results for h in r["heads"])
    assert (tmp_path/"refit.json").exists()


def test_sidecar_batch_shuffle_permutes_only_valid_rows():
    from uipc_manip.iaql_benchmark import sidecar_batch
    rows = [dict(tangent=np.full((2, 3), k, np.float32), reward_gradient=np.full(3, k)) for k in (1., 2., 3.)] + [dict()]
    kw = sidecar_batch(rows, 2, shuffle=np.random.default_rng(0))
    assert kw["valid"].tolist() == [1., 1., 1., 0.]
    assert sorted(float(t[0, 0]) for t in kw["tangent"][:3]) == [1., 2., 3.]
    assert all(float(kw["tangent"][i, 0, 0]) == float(kw["reward_gradient"][i, 0]) for i in range(3))
    assert float(kw["tangent"][3].abs().sum()) == 0


def test_state_batch_feeds_the_actor_term_from_the_same_label():
    torch.manual_seed(9)
    agent = SACAgent(ObsSpec(3), 2, SACConfig(actor_type="state", critic_input="privileged", privileged_dim=4,
        hidden_dim=16, batch_size=8, adjoint_weight=0.0, physics_actor_weight=0.5, actor_update_freq=1,
        physics_actor_action_distance=10.0, state_activation="silu"), "cpu")
    state, action = torch.randn(8, 4), torch.rand(8, 2)*2-1
    tangent = torch.randn(8, 4, 2)*.1
    nxt = state+torch.einsum("bsa,ba->bs", tangent, action)
    stats = agent.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1),
                                     tangent=tangent, reward_gradient=torch.zeros(8, 2), valid=torch.ones(8))
    assert "physics_loss" in stats and stats["physics_rows"] == 8 and stats["physics_actor_fraction"] == 1.0
    assert "adjoint_loss" not in stats and agent.physics_beta is not None
    agent.cfg.physics_actor_action_distance = 0.0
    stats = agent.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1),
                                     tangent=tangent, reward_gradient=torch.zeros(8, 2), valid=torch.ones(8))
    assert stats["physics_actor_fraction"] == 0.0 and "physics_loss" not in stats


def test_detailed_targets_split_and_trust():
    from uipc_manip.iaql import soft_targets_detailed
    torch.manual_seed(11)
    actor = StateActor(4, 2, 16, -3, 1).double()
    critic = PrivilegedCritic(4, 2, 16).double()
    ns = torch.randn(3, 4, dtype=torch.double)
    tangent = torch.randn(3, 4, 2, dtype=torch.double)*.3
    reward, mask = torch.randn(3, 1, dtype=torch.double), torch.ones(3, 1, dtype=torch.double)
    g_r = torch.randn(3, 2, dtype=torch.double)
    noise = torch.randn(3, 2, dtype=torch.double)
    y, g = soft_targets(actor, critic, ns, reward, mask, tangent, g_r, .2, .9, 100, noise)
    y2, g2, info = soft_targets_detailed(actor, critic, ns, reward, mask, tangent, g_r, .2, .9, 100, noise)
    torch.testing.assert_close(y, y2)
    torch.testing.assert_close(g, g2)
    torch.testing.assert_close(info["reward_norm"], g_r.norm(dim=-1))
    assert (info["trust"] == 1).all() and (info["disagreement"] >= 0).all()
    # the continuation is the label minus the reward part
    torch.testing.assert_close(info["continuation_norm"], (g - g_r).norm(dim=-1))
    # identical heads: no disagreement, full trust at any kappa
    critic.Q2.load_state_dict(critic.Q1.state_dict())
    _, g_same, info_same = soft_targets_detailed(actor, critic, ns, reward, mask, tangent, g_r, .2, .9, 100, noise, continuation_trust_kappa=5.0)
    torch.testing.assert_close(info_same["disagreement"], torch.zeros(3, dtype=torch.double), atol=1e-10, rtol=0)
    torch.testing.assert_close(info_same["trust"], torch.ones(3, dtype=torch.double))
    # distinct heads with a large kappa: the continuation is discounted, the reward part stays
    torch.manual_seed(12)
    critic2 = PrivilegedCritic(4, 2, 16).double()
    _, g_full, _ = soft_targets_detailed(actor, critic2, ns, reward, mask, tangent, g_r, .2, .9, 100, noise)
    _, g_trust, info_t = soft_targets_detailed(actor, critic2, ns, reward, mask, tangent, g_r, .2, .9, 100, noise, continuation_trust_kappa=50.0)
    assert (info_t["trust"] < 1).any()
    torch.testing.assert_close(g_trust, g_r + info_t["trust"][:, None]*(g_full - g_r), atol=1e-12, rtol=0)


def test_mix_mode_replaces_the_fraction_rho_of_the_critic_gradient():
    torch.manual_seed(13)
    cfg = dict(actor_type="state", critic_input="privileged", privileged_dim=4, hidden_dim=16, batch_size=8,
               actor_update_freq=1, state_activation="silu", physics_actor_weight=1.0, physics_actor_mode="mix",
               physics_actor_rho=1.0, physics_actor_action_distance=100.0, actor_log_std_min=-20, actor_log_std_max=-19)
    agent = SACAgent(ObsSpec(3), 2, SACConfig(**cfg), "cpu")
    state, action = torch.randn(8, 4), torch.rand(8, 2)*2-1
    tangent = torch.randn(8, 4, 2)*.1
    nxt = state+torch.einsum("bsa,ba->bs", tangent, action)
    stats = agent.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1),
                                     tangent=tangent, reward_gradient=torch.zeros(8, 2), valid=torch.ones(8))
    assert "physics_mix_cosine" in stats and stats["physics_rows"] == 8 and agent.physics_beta is None
    assert "label_critic_cosine" in stats and "label_continuation_ratio" in stats
    # with rho = 0 the update is the plain SAC update
    torch.manual_seed(13)
    plain = SACAgent(ObsSpec(3), 2, SACConfig(**{**cfg, "physics_actor_rho": 0.0}), "cpu")
    torch.manual_seed(13)
    ref = SACAgent(ObsSpec(3), 2, SACConfig(**{**cfg, "physics_actor_weight": 0.0}), "cpu")
    torch.manual_seed(5); plain.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1),
                                                   tangent=tangent, reward_gradient=torch.zeros(8, 2), valid=torch.ones(8))
    torch.manual_seed(5); ref.update_state_batch(state, action, torch.zeros(8, 1), nxt, torch.ones(8, 1))
    for a, b in zip(plain.actor.parameters(), ref.actor.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
