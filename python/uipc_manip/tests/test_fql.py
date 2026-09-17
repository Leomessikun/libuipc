"""FQL objectives, full-state resume, and offline transition boundaries."""
import json

import numpy as np
import pytest
import torch

from uipc_manip.expert_baseline import EpisodeTape
from uipc_manip.fql import FQLAgent, FQLConfig, bellman_target
from uipc_manip.obs import ObsSpec
from uipc_manip.tests.test_sac_agent import ToyEnv, _small_cfg
from uipc_manip.train_fql import load_transitions


def small_agent():
    torch.set_num_threads(1)
    return FQLAgent(ObsSpec(10), 3, FQLConfig(encoder=_small_cfg().encoder,
                    hidden_dim=32, flow_steps=3, alpha=10), "cpu")


def batch(agent):
    env = ToyEnv(agent.spec)
    flat = torch.as_tensor(np.stack([env.reset() for _ in range(4)]))
    return flat, torch.full((4, 3), .2), torch.ones(4, 1), flat.clone(), torch.ones(4, 1)


def test_fql_ordinary_bellman_target_and_terminal_bootstrap():
    r = torch.tensor([[1.], [2.]])
    mask = torch.tensor([[0.], [1.]])
    q1, q2 = torch.tensor([[10.], [4.]]), torch.tensor([[6.], [2.]])
    torch.testing.assert_close(bellman_target(r, mask, q1, q2, discount=.5), torch.tensor([[1.], [3.5]]))
    torch.testing.assert_close(bellman_target(r, mask, q1, q2, discount=.5, aggregation="min"), torch.tensor([[1.], [3.]]))


def test_flow_euler_integration_is_detached_and_clipped(monkeypatch):
    agent = small_agent()
    monkeypatch.setattr(agent.behavior, "vector", lambda encoded, x, t: torch.ones_like(x) * 2)
    result = agent.flow_actions(torch.ones(2, 1, requires_grad=True), torch.full((2, 3), -.5, requires_grad=True))
    torch.testing.assert_close(result, torch.ones(2, 3))
    assert not result.requires_grad


def test_fql_updates_all_three_components_and_resumes_exactly(tmp_path):
    torch.manual_seed(91)
    agent = small_agent()
    data = batch(agent)
    before = {name: [p.detach().clone() for p in module.parameters()]
              for name, module in (("actor", agent.actor), ("behavior", agent.behavior), ("critic", agent.critic))}
    metrics = agent.update(data)
    assert all(torch.isfinite(x) for x in metrics.values())
    for name, old in before.items():
        assert any(not torch.equal(a, b) for a, b in zip(old, getattr(agent, name).parameters()))
    assert all(p.grad is None and not p.requires_grad for p in agent.critic_target.parameters())
    assert all(p.requires_grad for p in agent.critic.parameters())
    path = tmp_path / "fql.pt"
    agent.save(path, {"env": "test"})
    expected = agent.act(data[0].numpy(), deterministic=True)
    metrics1 = agent.update(data)
    restored, metadata = FQLAgent.load(path, "cpu", resume=True)
    assert metadata == {"env": "test"}
    np.testing.assert_array_equal(expected, restored.act(data[0].numpy(), deterministic=True))
    metrics2 = restored.update(data)
    for key in metrics1:
        torch.testing.assert_close(metrics1[key], metrics2[key], rtol=0, atol=0)
    for key, value in agent.modules.state_dict().items():
        torch.testing.assert_close(value, restored.modules.state_dict()[key], rtol=0, atol=0)
    assert restored.updates == 2


def write_dataset(path, *, incomplete=False):
    path.mkdir()
    records = []
    for index, body in enumerate([1, 2]):
        tape = EpisodeTape(("upperarm_ratio",), True)
        for step in range(3):
            info = dict(upperarm_ratio=.1, sim_error=False, tracking_error=0., time_limit=step == 2)
            tape.step(np.zeros(1), np.zeros(2), np.full(3, step), "test", float(step), info,
                      next_obs=np.full(3, step + 100) if not (incomplete and step == 2) else None)
        record = dict(path=f"{index}.npz", human=body, garment="test", kept=False, source_sha256=f"source{body}")
        tape.save(path / record["path"], record)
        records.append(record)
    (path / "manifest.json").write_text(json.dumps(dict(completed=True, transition_schema="explicit_successors_v1",
                                                        obs_dim=3, action_dim=2)))
    (path / "episode_metrics.json").write_text(json.dumps(records))


def test_loader_retains_failures_and_timeout_successors_without_terminal_mask(tmp_path):
    source = tmp_path / "dataset"
    write_dataset(source)
    data, _, inventory = load_transitions(source, {2})
    assert len(data["train"][0]) == len(data["validation"][0]) == 3
    np.testing.assert_array_equal(data["train"][3][-1], np.full(3, 102))
    np.testing.assert_array_equal(data["train"][4], np.ones((3, 1)))
    assert not any(r["successful"] for r in inventory)


def test_tape_refuses_partial_successor_sequences(tmp_path):
    with pytest.raises(ValueError, match="successor observation"):
        write_dataset(tmp_path / "dataset", incomplete=True)


def test_loader_refuses_source_leakage(tmp_path):
    source = tmp_path / "dataset"
    write_dataset(source)
    path = source / "episode_metrics.json"
    records = json.loads(path.read_text())
    records[1]["source_sha256"] = records[0]["source_sha256"]
    path.write_text(json.dumps(records))
    with pytest.raises(ValueError, match="both sides"):
        load_transitions(source, {2})


def test_loader_drops_sim_error_reset_transition(tmp_path):
    source = tmp_path / "dataset"
    write_dataset(source)
    path = source / "0.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    metrics = [json.loads(str(row)) for row in arrays["step_metrics"]]
    metrics[-1]["sim_error"] = True
    arrays["step_metrics"] = np.array([json.dumps(row) for row in metrics])
    np.savez_compressed(path, **arrays)
    data, _, _ = load_transitions(source, {2})
    assert len(data["train"][0]) == 2
