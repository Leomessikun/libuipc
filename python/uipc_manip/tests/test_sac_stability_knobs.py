"""The three temperature and gradient knobs behind the late collapse (CPU).

Every arm of the critic ablation peaked and then fell while ``q1_mean`` kept rising and the policy
entropy kept falling: the six-month baseline reached entropy -7.76 with Q up 29.5. The temperature
target is what drives that fall, so it has to be settable; the floor and the clipping were already
settable but never set (``agent_docs/performance/2026-09-12-critic-architecture-defect.md``).
"""

import numpy as np
import pytest
import torch

from uipc_manip.models import EncoderConfig
from uipc_manip.obs import ObsSpec
from uipc_manip.sac import SACAgent, SACConfig

ACTION_DIM = 6


def _agent(**kw):
    cfg = SACConfig(
        encoder=EncoderConfig(sa_mlp=[[8, 8], [8, 8], [8, 8]], linear_mlp=[8], output_dim=5,
                              sa_neighbors=[2, 2], sa_ratio=[1.0, 1.0]),
        hidden_dim=16, batch_size=2, **kw,
    )
    return SACAgent(ObsSpec(16), ACTION_DIM, cfg, "cpu")


def test_the_default_target_entropy_is_the_reference_minus_action_dim():
    assert _agent().target_entropy == pytest.approx(-float(ACTION_DIM))


@pytest.mark.parametrize("scale, expected", [(0.5, -3.0), (0.25, -1.5), (0.0, -0.0)])
def test_a_smaller_scale_leaves_more_entropy_open(scale, expected):
    assert _agent(target_entropy_scale=scale).target_entropy == pytest.approx(expected)


def test_the_scale_travels_with_the_checkpoint(tmp_path):
    saved = _agent(target_entropy_scale=0.25)
    path = saved.save(tmp_path / "c.pt", 1, {})
    assert SACConfig.from_dict(SACAgent.read_checkpoint(path)["sac_config"]).target_entropy_scale == pytest.approx(0.25)


def test_a_checkpoint_written_before_the_field_reads_as_the_reference(tmp_path):
    payload = SACAgent.read_checkpoint(_agent().save(tmp_path / "c.pt", 1, {}))
    payload["sac_config"].pop("target_entropy_scale")
    assert SACConfig.from_dict(payload["sac_config"]).target_entropy_scale == pytest.approx(1.0)


def _obs(agent, spec, n=4):
    torch.manual_seed(0)
    return spec.unpack_torch(torch.randn(n, spec.dim))


@pytest.mark.parametrize("floor, bounded", [(0.0, False), (0.05, True)])
def test_the_floor_is_what_stops_the_temperature_from_decaying_away(floor, bounded):
    # The collapse ran alpha from 0.034 down to 0.024 while entropy fell to -0.13. With the target
    # far below the policy's entropy the temperature step only ever pushes alpha down, so without a
    # floor it decays without bound; the floor is the only thing that holds exploration open.
    spec = ObsSpec(16)
    # A target far below the policy's entropy, and a temperature step large enough to get there in
    # sixty updates rather than the run's hundred thousand.
    agent = _agent(min_alpha=floor, init_temperature=0.2, target_entropy_scale=8.0, alpha_lr=0.5)
    obs = _obs(agent, spec)
    for _ in range(60):
        agent._update_actor_and_alpha(obs)
    alpha = float(agent.alpha.detach())
    assert alpha == pytest.approx(0.05) if bounded else alpha < 0.01


def test_clipping_is_off_by_default_and_bounds_the_step_when_set():
    assert _agent().cfg.grad_clip_max_norm == 0.0
    agent = _agent(grad_clip_max_norm=1.0)
    for p in agent.actor.parameters():
        p.grad = torch.full_like(p, 100.0)
    assert agent._clip(agent.actor) > 1.0        # the norm reported is the one before clipping
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in agent.actor.parameters()))
    assert float(total) == pytest.approx(1.0, rel=1e-4)


def test_inheriting_weights_leaves_the_optimizer_moments_empty(tmp_path):
    """What --init-from relies on: the same weights, none of the collapsed run's momentum."""
    source = _agent(target_entropy_scale=1.0)
    for p in source.actor.parameters():          # move off the initialisation so equality means something
        with torch.no_grad():
            p.add_(0.1)
    path = source.save(tmp_path / "source.pt", 7, {})

    child = _agent(target_entropy_scale=0.25)    # a different entropy target must not block the transfer
    assert not child.actor_optimizer.state_dict()["state"]
    child.load(path, load_optimizers=False)
    for a, b in zip(source.actor.parameters(), child.actor.parameters(), strict=True):
        assert torch.equal(a, b)
    for a, b in zip(source.critic.parameters(), child.critic.parameters(), strict=True):
        assert torch.equal(a, b)
    assert not child.actor_optimizer.state_dict()["state"]
    assert not child.critic_optimizer.state_dict()["state"]
    # The inherited weights do not drag the source's entropy target along with them.
    assert child.target_entropy == pytest.approx(-1.5) and source.target_entropy == pytest.approx(-6.0)
