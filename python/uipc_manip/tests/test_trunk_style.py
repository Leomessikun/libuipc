"""The shape of a head's body: plain against pre-normalised residual (CPU)."""

import pytest
import torch

from uipc_manip.models import (
    CategoricalCritic,
    Critic,
    EncoderConfig,
    PrivilegedCritic,
    QHead,
    ResidualBlock,
    trunk_layers,
)
from uipc_manip.obs import EXTRA_DIM, FEATURE_DIM, ObsSpec
from uipc_manip.sac import SACConfig

ACTION_DIM, POINTS, BATCH = 6, 16, 4


def _cfg():
    return EncoderConfig(sa_mlp=[[8, 8], [8, 8], [8, 8]], linear_mlp=[8], output_dim=5,
                         sa_neighbors=[2, 2], sa_ratio=[1.0, 1.0])


def _obs():
    torch.manual_seed(0)
    return (torch.randn(BATCH, POINTS, 3), torch.randn(BATCH, POINTS, FEATURE_DIM),
            torch.ones(BATCH, POINTS, dtype=torch.bool), torch.randn(BATCH, EXTRA_DIM))


def test_low_level_plain_default_preserves_legacy_configs():
    assert SACConfig().trunk_style == "plain"
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in QHead(8, 16).modules())


def test_a_residual_block_is_the_identity_when_its_output_branch_is_zeroed():
    block = ResidualBlock(8)
    torch.nn.init.zeros_(block.fc2.weight)
    torch.nn.init.zeros_(block.fc2.bias)
    x = torch.randn(3, 8)
    assert torch.allclose(block(x), x)


def test_the_residual_trunk_normalises_and_the_plain_one_does_not():
    plain = trunk_layers(8, 16, 2, "plain")
    residual = trunk_layers(8, 16, 2, "residual", blocks=2)
    assert not any(isinstance(m, torch.nn.LayerNorm) for m in plain.modules())
    # One norm inside each block, plus the one before the output projection.
    assert sum(isinstance(m, torch.nn.LayerNorm) for m in residual.modules()) == 3
    for trunk in (plain, residual):
        assert trunk(torch.randn(BATCH, 8)).shape == (BATCH, 2)


def test_blocks_control_the_depth():
    counts = [sum(isinstance(m, ResidualBlock) for m in trunk_layers(8, 16, 2, "residual", blocks=n).modules())
              for n in (1, 2, 4)]
    assert counts == [1, 2, 4]


def test_every_head_owner_accepts_the_residual_trunk():
    obs, action = _obs(), torch.randn(BATCH, ACTION_DIM).clamp(-1, 1)
    spec = ObsSpec(POINTS)
    critic = Critic(spec, ACTION_DIM, 16, _cfg(), trunk_style="residual")
    q1, q2 = critic(obs, action)
    assert q1.shape == (BATCH, 1) and torch.isfinite(q1).all() and torch.isfinite(q2).all()
    cat = CategoricalCritic(spec, ACTION_DIM, 16, _cfg(), 11, -5.0, 5.0, trunk_style="residual")
    cq1, _, log_p1, _ = cat(obs, action)
    assert cq1.shape == (BATCH, 1) and log_p1.shape == (BATCH, 11) and torch.isfinite(log_p1).all()
    priv = PrivilegedCritic(35, ACTION_DIM, 16, trunk_style="residual")
    pq1, _ = priv(torch.randn(BATCH, 35), action)
    assert pq1.shape == (BATCH, 1) and torch.isfinite(pq1).all()


def test_the_residual_critic_still_passes_a_gradient_to_the_action():
    obs = _obs()
    action = torch.randn(BATCH, ACTION_DIM, requires_grad=True)
    q1, _ = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), trunk_style="residual")(obs, action)
    q1.sum().backward()
    assert action.grad is not None and action.grad.abs().sum() > 0.0


def test_an_unknown_style_is_refused():
    with pytest.raises(ValueError, match="trunk style"):
        trunk_layers(8, 16, 2, "curly")


def test_the_protocol_records_a_non_default_trunk_so_old_checkpoints_still_load():
    from uipc_manip.sac import SACAgent

    spec = ObsSpec(POINTS)
    base = SACConfig.from_dict({**SACConfig().to_dict(), "encoder": _cfg().to_dict(), "hidden_dim": 16})
    plain = SACAgent(spec, ACTION_DIM, base, "cpu").protocol()
    residual = SACAgent(spec, ACTION_DIM, SACConfig.from_dict({**base.to_dict(), "trunk_style": "residual"}),
                        "cpu").protocol()
    assert "trunk_style" not in plain
    assert residual["trunk_style"] == "residual" and residual["trunk_blocks"] == 2
