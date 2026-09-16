"""Bellman/likelihood semantics and conservative legacy replay links."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip.offline_rl import ImplicitQLearner, advantage_weights, exact_successors, expectile_loss, segment_ids
from uipc_manip.obs import ObsSpec
from uipc_manip.sac import SACAgent
from uipc_manip.tests.test_sac_agent import ToyEnv, _small_cfg


def test_links_do_not_join_interleaved_streams_or_terminal_resets():
    obs = np.array([[0], [10], [1], [11], [2], [12]], dtype=np.float32)
    nxt = np.array([[1], [11], [2], [12], [3], [13]], dtype=np.float32)
    mask = np.ones((6, 1))
    mask[3] = 0
    links = exact_successors(obs, nxt, mask)
    np.testing.assert_array_equal(links, [2, 3, 4, -1, -1, -1])
    ids = segment_ids(links)
    assert ids[0] == ids[2] == ids[4]
    assert ids[1] == ids[3] and ids[1] != ids[5] and ids[1] != ids[0]


def test_links_reject_duplicate_observations_and_multiple_predecessors():
    obs = np.array([[0], [1], [1], [2], [3]], dtype=np.float32)
    nxt = np.array([[1], [2], [2], [3], [4]], dtype=np.float32)
    np.testing.assert_array_equal(exact_successors(obs, nxt, np.ones((5, 1))), [-1, -1, -1, 4, -1])
    with pytest.raises(ValueError):
        segment_ids(np.array([1, 0]))


def test_expectile_and_weight_direction_and_overflow():
    diff = torch.tensor([-2.0, 1.0], requires_grad=True)
    loss = expectile_loss(diff, 0.8)
    assert float(loss.detach()) == pytest.approx(0.8)
    loss.backward()
    torch.testing.assert_close(diff.grad, torch.tensor([-0.4, 0.8]))
    weights = advantage_weights(torch.tensor([-1000.0, 0.0, 1000.0], requires_grad=True), 3, 100)
    torch.testing.assert_close(weights, torch.tensor([0.0, 1.0, 100.0]))
    assert not weights.requires_grad


def test_iql_bootstraps_terminal_mask_and_never_queries_candidate_actions(monkeypatch):
    torch.manual_seed(3)
    spec = ObsSpec(10)
    agent = SACAgent(spec, 3, _small_cfg(), "cpu")
    learner = ImplicitQLearner(agent)
    env = ToyEnv(spec)
    flat = torch.as_tensor(np.stack([env.reset() for _ in range(4)]))
    action = torch.full((4, 3), 0.2)
    reward = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    mask = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    obs = agent._unpack(flat)
    with torch.no_grad():
        target = reward + mask * agent.cfg.discount * learner.value(obs)
        q1, q2 = agent.critic(obs, action)
        expected = ((q1 - target).square().mean() + (q2 - target).square().mean()).item()
    def refuse_candidate(*args, **kwargs):
        raise AssertionError("IQL must not sample a policy action for its Bellman target")
    monkeypatch.setattr(agent.actor, "forward", refuse_candidate)
    original = agent.critic_target.forward
    def check_recorded(obs, candidate, **kwargs):
        torch.testing.assert_close(candidate, action)
        return original(obs, candidate, **kwargs)
    monkeypatch.setattr(agent.critic_target, "forward", check_recorded)
    before = [x.detach().clone() for x in agent.actor.parameters()]
    stats = learner.update((flat, action, reward, flat, mask))
    assert stats["critic_loss"] == pytest.approx(expected)
    assert all(np.isfinite(x) for x in stats.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert all(x.grad is None for x in agent.critic_target.parameters())
    assert learner.updates == 1


def test_bc_control_leaves_value_and_critic_unchanged():
    spec = ObsSpec(10)
    agent = SACAgent(spec, 3, _small_cfg(), "cpu")
    learner = ImplicitQLearner(agent)
    env = ToyEnv(spec)
    flat = torch.as_tensor(np.stack([env.reset() for _ in range(4)]))
    before = [x.detach().clone() for x in agent.critic.parameters()]
    stats = learner.update((flat, torch.zeros(4, 3), torch.ones(4, 1), flat, torch.ones(4, 1)), behavior_cloning=True)
    assert stats["weight_mean"] == 1 and stats["weight_ess"] == 4
    assert all(torch.equal(a, b) for a, b in zip(before, agent.critic.parameters()))
    assert all(x.grad is None for x in learner.value.parameters())
