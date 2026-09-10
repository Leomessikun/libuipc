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


def _small_cfg(actor_type: str = "flat", algo: str = "sac", encoder: str = "pointnet2") -> SACConfig:
    cfg = SACConfig(hidden_dim=32, batch_size=16, actor_update_freq=2, actor_type=actor_type, algo=algo, num_bins=21, min_v=-5.0, max_v=5.0)
    cfg.encoder = EncoderConfig(
        kind=encoder,
        sa_mlp=[[16, 16], [16, 16], [16, 32]],
        fp_mlp=[[16, 16], [16, 8], [8, 8]],
        linear_mlp=[16],
        output_dim=8,
        sa_neighbors=[6, 6],
        transformer_dim=16,
        transformer_heads=2,
        transformer_layers=1,
    )
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


@pytest.mark.parametrize(
    ("actor_type", "algo", "encoder"),
    [("wang-flow", "sac", "pointnet2"), ("wang-flow", "flashsac", "pointnet2"), ("flat", "flashsac", "transformer"), ("wang-flow", "sac", "transformer")],
)
def test_agent_variants_update_and_roundtrip(tmp_path, actor_type, algo, encoder):
    torch.manual_seed(0)
    spec = ObsSpec(10)
    env = ToyEnv(spec)
    cfg = _small_cfg(actor_type, algo, encoder)
    agent = SACAgent(spec, 3, cfg, "cpu")
    replay = FlatReplayBuffer(spec.dim, 3, 256, 16, "cpu")
    obs = env.reset()
    for _ in range(64):
        action = np.random.uniform(-1, 1, size=3)
        next_obs, reward, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, False)
        obs = next_obs
    stats = None
    for _ in range(8):
        stats = agent.update(replay)
    assert np.isfinite(stats["critic_loss"]) and np.isfinite(stats["q1_mean"])
    if algo == "flashsac":
        assert cfg.min_v <= stats["q1_mean"] <= cfg.max_v
    before = agent.act(np.stack([obs, obs]), deterministic=True)
    path = agent.save(tmp_path / f"{actor_type}_{algo}_{encoder}.pt", step=1)
    restored = SACAgent(spec, 3, _small_cfg(actor_type, algo, encoder), "cpu")
    restored.load(path)
    np.testing.assert_allclose(restored.act(np.stack([obs, obs]), deterministic=True), before, atol=1e-6)
    other = SACAgent(spec, 3, _small_cfg("flat" if actor_type != "flat" else "wang-flow", algo, encoder), "cpu")
    with pytest.raises(ValueError):
        other.load(path)



def test_privileged_critic_trains_roundtrips_and_refuses_the_other_form(tmp_path):
    torch.manual_seed(0)
    spec = ObsSpec(10)
    env = ToyEnv(spec)
    cfg = _small_cfg("wang-flow")
    cfg.critic_input, cfg.privileged_dim = "privileged", 6
    agent = SACAgent(spec, 3, cfg, "cpu")
    assert agent.critic.encoder is None
    replay = FlatReplayBuffer(spec.dim, 3, 256, 16, "cpu", priv_dim=6)
    obs = env.reset()
    state = np.concatenate([env.tool, env.goal])
    for _ in range(64):
        action = np.random.uniform(-1, 1, size=3)
        next_obs, reward, _ = env.step(action)
        next_state = np.concatenate([env.tool, env.goal])
        replay.add(obs, action, reward, next_obs, False, priv=state, next_priv=next_state)
        obs, state = next_obs, next_state
    target = [p.detach().clone() for p in agent.critic_target.parameters()]
    for _ in range(8):
        stats = agent.update(replay)
    assert np.isfinite(stats["critic_loss"]) and np.isfinite(stats["actor_loss"])
    assert any(not torch.equal(b, a) for b, a in zip(target, agent.critic_target.parameters(), strict=True))
    # The actor still learns from dQ/da, now through the state critic.
    _, pi, _, _ = agent.actor(agent._unpack(torch.as_tensor(np.stack([obs, obs]))))
    q1, _ = agent.critic(torch.as_tensor(np.stack([state, state]), dtype=torch.float32), pi)
    assert torch.autograd.grad(q1.sum(), pi)[0].abs().sum() > 0.0
    batch = np.stack([obs, obs])
    path = agent.save(tmp_path / "privileged.pt", step=3)
    replay.save(tmp_path / "replay", metadata={"step": 3})
    restored = SACAgent(spec, 3, cfg, "cpu")
    restored.load(path)
    np.testing.assert_allclose(restored.act(batch, deterministic=True), agent.act(batch, deterministic=True), atol=1e-6)
    other = FlatReplayBuffer(spec.dim, 3, 256, 16, "cpu", priv_dim=6)
    other.load(tmp_path / "replay")
    np.testing.assert_array_equal(other._next_priv[: other.size], replay._next_priv[: replay.size])
    # Neither form resumes from the other's checkpoint or replay, and nothing trains on a state it lacks.
    with pytest.raises(ValueError):
        SACAgent(spec, 3, _small_cfg("wang-flow"), "cpu").load(path)
    with pytest.raises(ValueError, match="privileged"):
        FlatReplayBuffer(spec.dim, 3, 256, 16, "cpu").load(tmp_path / "replay")
    plain = FlatReplayBuffer(spec.dim, 3, 256, 16, "cpu")
    for _ in range(20):
        plain.add(obs, np.zeros(3), 0.0, obs, False)
    with pytest.raises(ValueError, match="privileged"):
        agent.update(plain)
    with pytest.raises(ValueError, match="privileged"):
        replay.add(obs, np.zeros(3), 0.0, obs, False)
    for bad in ({"privileged_dim": 0}, {"algo": "flashsac"}):
        wrong = _small_cfg("wang-flow", bad.get("algo", "sac"))
        wrong.critic_input, wrong.privileged_dim = "privileged", bad.get("privileged_dim", 6)
        with pytest.raises(ValueError, match="privileged"):
            SACAgent(spec, 3, wrong, "cpu")


def test_bf16_encoders_hand_fp32_features_to_fp32_heads():
    torch.manual_seed(0)
    spec = ObsSpec(10)
    env = ToyEnv(spec)
    cfg = _small_cfg("wang-flow")
    half = SACConfig.from_dict({**cfg.to_dict(), "encoder_precision": "bf16"})
    ref, agent = SACAgent(spec, 3, cfg, "cpu"), SACAgent(spec, 3, half, "cpu")
    for name in ("actor", "critic", "critic_target"):
        getattr(agent, name).load_state_dict(getattr(ref, name).state_dict())
    obs = agent._unpack(torch.as_tensor(np.stack([env.reset() for _ in range(4)])))
    action = torch.zeros(4, 3)
    assert agent.critic.encoder(*obs[:3]).dtype == torch.float32
    q, q_ref = agent.critic(obs, action)[0], ref.critic(obs, action)[0]
    assert q.dtype == torch.float32 and torch.allclose(q, q_ref, rtol=0.05, atol=0.05) and not torch.equal(q, q_ref)
    replay = FlatReplayBuffer(spec.dim, 3, 256, 16, "cpu")
    o = env.reset()
    for _ in range(32):
        a = np.random.uniform(-1, 1, size=3)
        n, r, _ = env.step(a)
        replay.add(o, a, r, n, False)
        o = n
    for _ in range(4):
        stats = agent.update(replay)
    assert np.isfinite(stats["critic_loss"]) and np.isfinite(stats["actor_loss"])
    with pytest.raises(ValueError, match="encoder_precision"):
        SACAgent(spec, 3, SACConfig.from_dict({**cfg.to_dict(), "encoder_precision": "fp16"}), "cpu")

def test_garment_curriculum_follows_wang_schedule():
    from uipc_manip.curriculum import WANG_GARMENT_ORDER, curriculum_order, garment_curriculum_stage

    # Wang: curriculum_step = step // curriculum_update_freq + 1, capped at the garment count.
    assert [garment_curriculum_stage(s, interval=100, garment_count=3) for s in (0, 99, 100, 199, 200, 10_000)] == [1, 1, 2, 2, 3, 3]
    assert garment_curriculum_stage(0, interval=0, garment_count=3) == 3
    present = ["tshirt_392", "tshirt_26", "tshirt_392", "tshirt_26"]
    assert curriculum_order(WANG_GARMENT_ORDER, present) == ["tshirt_26", "tshirt_392"]
    assert curriculum_order(None, present) == ["tshirt_392", "tshirt_26"]
    assert curriculum_order(["jacket", "tshirt_26"], present) == ["tshirt_26", "tshirt_392"]


@pytest.mark.parametrize("actor_type", ["flat", "wang-flow"])
def test_action_log_prob_matches_the_sampled_log_pi(actor_type):
    spec = ObsSpec(16)
    cfg = SACConfig(hidden_dim=32, actor_type=actor_type, encoder=EncoderConfig(kind="pointnet2", sa_neighbors=[4, 4]))
    agent = SACAgent(spec, 3, cfg, "cpu")
    env = ToyEnv(spec, seed=3)
    obs = np.stack([env.reset() for _ in range(5)])
    batch = agent._unpack(torch.as_tensor(obs))
    torch.manual_seed(0)
    _, pi, log_pi, _ = agent.actor(batch)
    recomputed = agent.actor.action_log_prob(batch, pi.detach())
    assert torch.allclose(recomputed, log_pi, atol=1e-3, rtol=1e-3)
    # The clamp keeps saturated targets finite.
    assert torch.isfinite(agent.actor.action_log_prob(batch, torch.ones_like(pi))).all()


def test_episode_collector_applies_the_paper_filter(tmp_path):
    from uipc_manip.collect_rollouts import EpisodeCollector

    collector = EpisodeCollector(2, tmp_path, min_upperarm_ratio=0.7, target_kept=1)
    obs = np.zeros(8, dtype=np.float32)
    act = np.zeros(2, dtype=np.float32)
    # Slot 0 dresses the arm without turning early; slot 1 dresses it but cut the elbow.
    for t in range(3):
        last = t == 2
        r0 = collector.step(0, obs, act, 1.0, {"upperarm_ratio": 0.8 if last else 0.2, "garment": "tshirt_26", "time_limit": last})
        r1 = collector.step(1, obs, act, 1.0, {"upperarm_ratio": 0.9 if last else 0.3, "garment": "tshirt_392", "early_turn": t == 1, "time_limit": last})
    assert r0["kept"] and r0["paper_filter_success"] and r0["length"] == 3 and r0["path"] is not None
    assert not r1["kept"] and r1["early_turn"] and r1["path"] is None
    data = np.load(tmp_path / r0["path"])
    assert data["obs"].shape == (3, 8) and data["actions"].shape == (3, 2)
    # The target was reached, so a further success is recorded but not stored.
    r2 = collector.step(0, obs, act, 0.0, {"upperarm_ratio": 0.95, "garment": "tshirt_26", "time_limit": True})
    assert r2["paper_filter_success"] and not r2["kept"]
    summary = collector.summary()
    assert summary["attempted_episodes"] == 3 and summary["kept_episodes"] == 1 and summary["kept_per_garment"] == {"tshirt_26": 1, "tshirt_392": 0}


def test_distillation_smoke_produces_a_loadable_student(tmp_path):
    import json

    from uipc_manip import distill

    spec = ObsSpec(16)
    env = ToyEnv(spec, seed=5)
    rng = np.random.default_rng(0)
    rollouts = tmp_path / "rollouts"
    (rollouts / "episodes").mkdir(parents=True)
    records = []
    for k in range(4):
        obs = np.stack([env.reset() for _ in range(6)])
        act = np.clip(rng.normal(scale=0.3, size=(6, 3)), -1, 1).astype(np.float32)
        path = f"episodes/episode_{k:05d}.npz"
        np.savez_compressed(rollouts / path, obs=obs, actions=act, rewards=np.zeros(6, dtype=np.float32))
        records.append({"path": path, "kept": True, "final_upperarm_ratio": 0.9, "early_turn": False, "sim_error": False})
    (rollouts / "episode_metrics.json").write_text(json.dumps(records))
    (rollouts / "manifest.json").write_text(json.dumps({"obs_dim": spec.dim, "action_dim": 3, "point_budget": 16}))
    distill.main([
        "--source-dirs", str(rollouts), "--work-dir", str(tmp_path), "--run-name", "student",
        "--encoder", "pointnet2", "--hidden-dim", "32", "--steps", "6", "--batch-size", "4",
        "--eval-every", "3", "--save-every", "0", "--device", "cpu", "--loss", "nll_mse",
    ])
    ckpt = tmp_path / "student" / "checkpoints" / "actor_final.pt"
    assert ckpt.exists() and (tmp_path / "student" / "checkpoints" / "actor_best.pt").exists()
    cfg = SACConfig.from_dict(SACAgent.read_checkpoint(ckpt)["sac_config"])
    student = SACAgent(spec, 3, cfg, "cpu")
    payload = student.load(ckpt, load_optimizers=False)
    assert payload["metadata"]["stage"] == "fmvp_distillation"
    assert student.act(np.stack([env.reset()]), deterministic=True).shape == (1, 3)
