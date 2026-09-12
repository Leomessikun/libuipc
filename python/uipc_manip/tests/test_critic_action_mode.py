"""Where the action enters the point-cloud critic (CPU).

The reference concatenates the action to every point before the encoder; this port concatenated it
to the encoded vector instead until 2026-09-13, which is the baseline the reference reports as much
worse (``agent_docs/performance/2026-09-12-critic-architecture-defect.md``).
"""

import pytest
import torch

from uipc_manip.models import CategoricalCritic, Critic, EncoderConfig
from uipc_manip.obs import EXTRA_DIM, FEATURE_DIM, ObsSpec
from uipc_manip.sac import SACConfig

ACTION_DIM, POINTS, BATCH = 6, 24, 3


def _cfg():
    # A small encoder: the question is where the action enters, not how big the network is.
    return EncoderConfig(sa_mlp=[[8, 8], [8, 8], [8, 8]], linear_mlp=[8], output_dim=5,
                         sa_neighbors=[2, 2], sa_ratio=[1.0, 1.0])


def _obs():
    torch.manual_seed(0)
    return (torch.randn(BATCH, POINTS, 3), torch.randn(BATCH, POINTS, FEATURE_DIM),
            torch.ones(BATCH, POINTS, dtype=torch.bool), torch.randn(BATCH, EXTRA_DIM))


def test_dense_is_the_default():
    assert SACConfig().critic_action_mode == "dense"
    assert Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg()).action_mode == "dense"


def test_dense_widens_the_encoder_by_the_action_and_latent_does_not():
    spec = _cfg()
    dense = Critic(ObsSpec(POINTS), ACTION_DIM, 16, spec, action_mode="dense")
    latent = Critic(ObsSpec(POINTS), ACTION_DIM, 16, spec, action_mode="latent")
    # The first set-abstraction layer sees three coordinates plus the point features, and under
    # 'dense' the action is among them.
    dense_in = dense.encoder.sa1.mlp[0].in_features
    latent_in = latent.encoder.sa1.mlp[0].in_features
    assert dense_in - latent_in == ACTION_DIM
    # And the head no longer takes the action separately.
    assert latent.Q1.trunk[0].in_features - dense.Q1.trunk[0].in_features == ACTION_DIM


def test_the_dense_encoding_depends_on_the_action_and_the_latent_one_does_not():
    obs = _obs()
    a, b = torch.zeros(BATCH, ACTION_DIM), torch.ones(BATCH, ACTION_DIM)
    for mode, expect_same in (("dense", False), ("latent", True)):
        critic = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), action_mode=mode).eval()
        with torch.no_grad():
            pos, feat, valid, _ = obs
            if mode == "dense":
                from uipc_manip.models import _broadcast_action

                za = critic.encoder(pos, _broadcast_action(feat, a), valid)
                zb = critic.encoder(pos, _broadcast_action(feat, b), valid)
            else:
                za = zb = critic.encoder(pos, feat, valid)
        assert torch.allclose(za, zb) is expect_same


def test_both_modes_produce_finite_twin_q_values_of_the_right_shape():
    obs, action = _obs(), torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
    for mode in ("dense", "latent"):
        q1, q2 = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), action_mode=mode)(obs, action)
        assert q1.shape == (BATCH, 1) and q2.shape == (BATCH, 1)
        assert torch.isfinite(q1).all() and torch.isfinite(q2).all()


def test_the_action_reaches_the_q_value_under_both_modes():
    # Whichever way it enters, the gradient of Q with respect to the action must not vanish.
    obs = _obs()
    for mode in ("dense", "latent"):
        action = torch.randn(BATCH, ACTION_DIM, requires_grad=True)
        q1, _ = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), action_mode=mode)(obs, action)
        q1.sum().backward()
        assert action.grad is not None and action.grad.abs().sum() > 0.0


def test_the_categorical_critic_follows_the_same_rule():
    obs, action = _obs(), torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
    for mode in ("dense", "latent"):
        critic = CategoricalCritic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), 11, -5.0, 5.0, action_mode=mode)
        q1, q2, log_p1, log_p2 = critic(obs, action)
        assert q1.shape == (BATCH, 1) and log_p1.shape == (BATCH, 11)
        assert torch.isfinite(q1).all() and torch.isfinite(log_p1).all()


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="action_mode"):
        Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), action_mode="sideways")
