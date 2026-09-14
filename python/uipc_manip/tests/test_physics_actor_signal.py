"""CPU checks of the physics actor signal's plumbing: the replay stores and returns the direction
last, snapshots keep it, the actor-loss helpers follow the fine-tune's definitions, and the
adjoint solution maps to six action components as the probe's chain does."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip import physics_actor_signal as signal  # noqa: E402
from uipc_manip.replay import FlatReplayBuffer, ReplaySet  # noqa: E402
from uipc_manip.sac import physics_actor_loss, physics_direction  # noqa: E402


def test_replay_stores_the_direction_last_and_keeps_it_in_snapshots(tmp_path):
    buf = FlatReplayBuffer(4, 6, 8, 3, "cpu", physics=True)
    g = np.arange(6, dtype=np.float32)
    buf.add(np.zeros(4), np.zeros(6), 1.0, np.ones(4), False, physics=g, physics_valid=1.0)
    buf.add(np.zeros(4), np.zeros(6), 1.0, np.ones(4), False)  # no direction: zeros that do not count
    buf.add(np.zeros(4), np.zeros(6), 1.0, np.ones(4), False, physics=-g, physics_valid=0.0)
    batch = buf.sample(3)
    assert len(batch) == 7
    physics, valid = batch[5], batch[6]
    assert physics.shape == (3, 6) and valid.shape == (3, 1)
    rows = {tuple(np.round(p.numpy(), 3)): float(v) for p, v in zip(physics, valid)}
    assert rows.get(tuple(g)) in (1.0, None) and rows.get(tuple(np.zeros(6))) in (0.0, None)
    buf.save(tmp_path / "snap")
    again = FlatReplayBuffer(4, 6, 8, 3, "cpu", physics=True)
    again.load(tmp_path / "snap")
    assert np.allclose(again._physics[:3], buf._physics[:3]) and np.allclose(again._physics_valid[:3], buf._physics_valid[:3])
    plain = FlatReplayBuffer(4, 6, 8, 3, "cpu")
    plain.add(np.zeros(4), np.zeros(6), 1.0, np.ones(4), False)
    assert len(plain.sample(1)) == 5
    plain.save(tmp_path / "plain")
    with_physics = FlatReplayBuffer(4, 6, 8, 3, "cpu", physics=True)
    with_physics.load(tmp_path / "plain")  # an older snapshot resumes with rows that carry no direction
    assert float(with_physics._physics_valid[:1].sum()) == 0.0


def test_replay_set_passes_the_direction_through():
    rs = ReplaySet(["a", "b"], 4, 6, 8, 2, "cpu", physics=True)
    for _ in range(3):
        rs.add("a", np.zeros(4), np.zeros(6), 0.0, np.zeros(4), False, physics=np.ones(6), physics_valid=1.0)
    batch = rs.sample(2)
    assert len(batch) == 8 and batch[-1] == 0  # ..., physics, physics_valid, buffer index
    assert torch.allclose(batch[5], torch.ones(2, 6)) and torch.allclose(batch[6], torch.ones(2, 1))


def test_actor_loss_helpers():
    g = torch.tensor([[3.0, 0.0, 4.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([[1.0], [0.0]])
    d = physics_direction(g, valid)
    assert torch.allclose(d[0], torch.tensor([0.6, 0.0, 0.8, 0.0, 0.0, 0.0])) and torch.all(d[1] == 0)
    mu = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    loss = physics_actor_loss(mu, d, 2.0)
    assert loss.item() == pytest.approx(-2.0 * 0.6 / 2)
    loss.backward()
    assert torch.allclose(mu.grad[0], -2.0 * d[0] / 2) and torch.all(mu.grad[1] == 0)


def test_action_gradient_matches_the_chain_conventions():
    layout = {"n": 5, "anchor_idx": np.array([2]), "strength": 10.0, "mass": np.full(5, 0.5)}
    lam = np.zeros((5, 3))
    lam[2] = [1.0, 0.0, 0.0]
    offsets = np.array([[0.0, 0.02, 0.0]])
    g = signal.action_gradient(lam.reshape(-1), layout, offsets, 0.01, 0.1, True)
    assert np.allclose(g[:3], [5.0 * 0.01, 0.0, 0.0])
    assert np.allclose(g[3:], [0.0, 0.0, -0.1 * 0.1])
