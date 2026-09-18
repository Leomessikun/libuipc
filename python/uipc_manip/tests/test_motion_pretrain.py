"""Material correspondence, causal inputs, and isolated RL encoder transfer."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from uipc_manip.motion_pretrain import (TrackDecoder, initialize_actor_encoder,
                                       prediction_loss, track_windows)
from uipc_manip.obs import ObsSpec
from uipc_manip.tests.test_fql import small_agent, batch


def material_tape():
    spec = ObsSpec(10)
    initial = np.array([[.02, 0, 0], [.04, 0, 0]], dtype=np.float32)
    positions = np.stack([initial + [0, .005 * t, 0] for t in range(5)])
    observations = np.stack([spec.pack(positions[t][::-1], np.zeros(2, bool),
                                      [.2, 0, 0], np.zeros(3), True) for t in range(4)])
    actions = np.arange(8, dtype=np.float32).reshape(4, 2) / 10
    return spec, positions, observations, actions


def test_tracks_use_material_ids_and_preserve_last_successor():
    spec, positions, observations, actions = material_tape()
    rows = track_windows(observations, actions, positions, spec, horizon=2, queries=3)
    assert rows["start"].tolist() == [0, 1, 2]
    assert rows["mask"].sum() == 6  # two real vertices, not three observations
    for i, t in enumerate(rows["start"]):
        ids = rows["vertex_ids"][i]
        np.testing.assert_allclose(rows["query"][i], positions[t, ids])
        expected = (positions[t + 1:t + 3, ids] - positions[t, ids]).transpose(1, 0, 2)
        np.testing.assert_allclose(rows["delta"][i], expected, atol=1e-8)
        np.testing.assert_array_equal(rows["actions"][i], actions[t:t + 2])


def test_future_geometry_does_not_choose_queries_or_change_current_input():
    spec, positions, observations, actions = material_tape()
    a = track_windows(observations, actions, positions, spec, horizon=4, queries=2)
    changed = positions.copy()
    changed[1:] += 1
    b = track_windows(observations, actions, changed, spec, horizon=4, queries=2)
    for key in ("obs", "query", "vertex_ids", "actions"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["delta"], b["delta"])


def test_missing_terminal_geometry_rejected():
    spec, positions, observations, actions = material_tape()
    with pytest.raises(ValueError, match="T\\+1"):
        track_windows(observations, actions, positions[:-1], spec)


def test_geometry_control_cannot_copy_query_coordinates_or_future_actions():
    torch.manual_seed(1)
    decoder = TrackDecoder(8, 2, 2, 3)
    z, action, query = torch.randn(4, 8), torch.randn(4, 2, 2), torch.randn(4, 3, 3)
    a = decoder(z, action, query, objective="geometry")
    b = decoder(z, action + 100, query + 100, objective="geometry")
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    c = decoder(z, action, query, objective="motion")
    d = decoder(z, action + 1, query, objective="motion")
    assert not torch.equal(c, d)


@pytest.mark.parametrize("objective", ["motion", "geometry"])
def test_padding_targets_do_not_affect_loss(objective):
    pred = torch.randn(2, 3, 2, 3, requires_grad=True)
    query, delta = torch.randn(2, 3, 3), torch.randn(2, 3, 2, 3)
    mask = torch.tensor([[1., 1., 0.], [1., 1., 0.]])
    a = prediction_loss(pred, query, delta, mask, objective)
    query[:, 2] += 1000
    delta[:, 2] += 1000
    b = prediction_loss(pred, query, delta, mask, objective)
    torch.testing.assert_close(a, b)
    b.backward()
    assert torch.isfinite(pred.grad).all()


def test_pretraining_reaches_actor_readout_and_transfer_leaves_other_weights(tmp_path):
    agent = small_agent()
    obs = batch(agent)[0]
    decoder = TrackDecoder(agent.actor.encoder.feature_dim + 7, 3, 2, 3)
    before = {k: v.clone() for k, v in agent.modules.state_dict().items()}
    opt = torch.optim.Adam(list(agent.actor.encoder.parameters()) + list(decoder.parameters()), lr=.01)
    pred = decoder(agent.actor._frame_latent(agent.unpack(obs)), torch.randn(4, 2, 3),
                   torch.randn(4, 3, 3), objective="motion")
    loss = prediction_loss(pred, torch.zeros(4, 3, 3), torch.ones_like(pred) * .01,
                           torch.ones(4, 3), "motion")
    loss.backward()
    opt.step()
    assert any(not torch.equal(v, before[k]) for k, v in agent.modules.state_dict().items()
               if k.startswith("actor.encoder."))
    assert all(torch.equal(v, before[k]) for k, v in agent.modules.state_dict().items()
               if not k.startswith("actor.encoder."))
    path = tmp_path / "encoder.pt"
    torch.save(dict(format="dressing_motion_encoder_v1", encoder_config=asdict(agent.cfg.encoder),
                    point_budget=agent.spec.point_budget, action_dim=agent.action_dim,
                    encoder=agent.actor.encoder.state_dict(), objective="motion", steps=1, seed=1,
                    validation_bodies=[2], training_bodies=[1], env={}, seconds=1.), path)
    target = small_agent()
    old = {k: v.clone() for k, v in target.modules.state_dict().items()}
    initialize_actor_encoder(target, path)
    for k, v in target.modules.state_dict().items():
        torch.testing.assert_close(v, agent.modules.state_dict()[k] if k.startswith("actor.encoder.") else old[k])
    assert not target.optimizer.state and target.updates == 0
    # Ordinary FQL checkpoint remains deployable without the disposable decoder.
    target.save(tmp_path / "rl.pt")
    from uipc_manip.fql import FQLAgent
    restored, _ = FQLAgent.load(tmp_path / "rl.pt", "cpu")
    np.testing.assert_array_equal(target.act(obs.numpy(), deterministic=True), restored.act(obs.numpy(), deterministic=True))
