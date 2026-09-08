"""CPU tests for the SAC agent on a synthetic point-cloud goal-reaching problem."""

import numpy as np
import pytest

from uipc_manip.obs import ObsSpec

torch = pytest.importorskip("torch")

from uipc_manip.models import EncoderConfig  # noqa: E402
from uipc_manip.replay import FlatReplayBuffer  # noqa: E402
from uipc_manip.sac import (  # noqa: E402
    SACAgent,
    SACConfig,
    gradient_update_budget,
    wang_equivalent_alpha_lr,
    wang_equivalent_discount,
    wang_equivalent_reward_scale,
)


class ToyEnv:
    """A rigid blob of points attached to the tool must reach a goal."""

    def __init__(self, spec: ObsSpec, seed: int = 0) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.max_translation = 0.02
        self.blob = self.rng.normal(scale=0.01, size=(6, 3))
        self.marker = np.array([True, True, False, False, False, False])

    def reset(self):
        self.tool = np.zeros(3)
        self.goal = self.rng.uniform(-0.05, 0.05, size=3)
        return self.obs()

    def obs(self):
        return self.spec.pack(self.blob, self.marker, self.goal - self.tool, self.tool, True)

    def step(self, action):
        prev = np.linalg.norm(self.tool + self.blob[self.marker].mean(0) - self.goal)
        self.tool = self.tool + np.clip(action, -1, 1) * self.max_translation
        dist = np.linalg.norm(self.tool + self.blob[self.marker].mean(0) - self.goal)
        return self.obs(), float(10 * (prev - dist)), dist


def _small_cfg() -> SACConfig:
    cfg = SACConfig(hidden_dim=32, batch_size=16, actor_update_freq=2)
    cfg.encoder = EncoderConfig(sa_mlp=[[16, 16], [16, 16], [16, 32]], linear_mlp=[16], output_dim=8, sa_neighbors=[6, 6])
    return cfg


def test_wang_helpers():
    assert wang_equivalent_discount(150) == pytest.approx(0.99)
    assert wang_equivalent_discount(900) == pytest.approx(0.998333, abs=1e-6)
    assert wang_equivalent_alpha_lr(150) == pytest.approx(1e-4)
    assert wang_equivalent_reward_scale(0.99) == pytest.approx(1.0)
    assert gradient_update_budget(transitions_added=35, replay_size=70, batch_size=64, updates_started=False, updates_per_step=0) == (5, True)
    assert gradient_update_budget(transitions_added=4, replay_size=10, batch_size=64, updates_started=False, updates_per_step=0) == (0, False)
    assert gradient_update_budget(transitions_added=4, replay_size=1000, batch_size=64, updates_started=True, updates_per_step=2) == (2, True)


def test_sac_update_and_checkpoint(tmp_path):
    torch.manual_seed(0)
    spec = ObsSpec(10)
    env = ToyEnv(spec)
    agent = SACAgent(spec, 3, _small_cfg(), "cpu")
    replay = FlatReplayBuffer(spec.dim, 3, 512, 16, "cpu")
    obs = env.reset()
    for _ in range(200):
        action = np.random.uniform(-1, 1, size=3)
        next_obs, reward, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, False)
        obs = next_obs
    stats = None
    for _ in range(40):
        stats = agent.update(replay)
    assert stats is not None and np.isfinite(stats["critic_loss"]) and np.isfinite(stats.get("actor_loss", 0.0))
    assert "alpha" in stats and stats["alpha"] > 0.0
    batch = np.stack([obs, obs])
    before = agent.act(batch, deterministic=True)
    assert before.shape == (2, 3) and np.all(np.abs(before) <= 1.0)
    path = agent.save(tmp_path / "ckpt.pt", step=7, metadata={"task": "toy"})
    assert path.exists() and path.with_suffix(".json").exists()
    replay.save(tmp_path / "replay", metadata={"step": 7})
    restored = SACAgent(spec, 3, _small_cfg(), "cpu")
    payload = restored.load(path)
    assert payload["step"] == 7 and restored.updates == agent.updates
    np.testing.assert_allclose(restored.act(batch, deterministic=True), before, atol=1e-6)
    other = FlatReplayBuffer(spec.dim, 3, 512, 16, "cpu")
    assert other.load(tmp_path / "replay")["step"] == 7 and other.size == replay.size
    mismatched = SACAgent(ObsSpec(12), 3, _small_cfg(), "cpu")
    with pytest.raises(ValueError):
        mismatched.load(path)


def test_actor_update_uses_the_critic_action_gradient():
    """The actor must learn from ``dQ/da``, not from the entropy term alone.

    The critic concatenates the action after its encoder, so detaching that
    encoder during the actor update saves work without cutting the path to the
    action. Detaching the action itself instead would leave ``Q(s, pi(s))``
    constant in ``pi`` and silently reduce the actor to entropy maximisation.
    """
    torch.manual_seed(0)
    spec = ObsSpec(10)
    agent = SACAgent(spec, 3, _small_cfg(), "cpu")
    rng = np.random.default_rng(0)
    blob = rng.normal(scale=0.01, size=(6, 3))
    marker = np.array([True, True, False, False, False, False])
    obs_flat = torch.as_tensor(
        np.stack([spec.pack(blob, marker, rng.normal(size=3) * 0.05, np.zeros(3), True) for _ in range(8)])
    )
    obs = agent._unpack(obs_flat)
    _, pi, _, _ = agent.actor(obs)
    q1, _ = agent.critic(obs, pi, detach_encoder=True)
    grad = torch.autograd.grad(q1.sum(), pi, retain_graph=True)[0]
    assert grad.abs().sum() > 0.0, "critic gives the actor no action gradient"

    before = [p.detach().clone() for p in agent.actor.trunk.parameters()]
    agent.updates = agent.cfg.actor_update_freq - 1
    agent._update_actor_and_alpha(obs)
    after = list(agent.actor.trunk.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))
