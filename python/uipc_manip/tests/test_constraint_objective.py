"""Stage 0: the absorbing grasp constraint as a cost the learner can actually see.

The dressing reward contains no term for the criterion that decides success -- every
per-decision anchor tracking maximum within 2 cm. These tests pin the three pieces that
make the constraint learnable: the observation slot that makes the absorbing state
representable, the reconstruction of the per-transition cost from that slot, and the dual
step that prices it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from collections import deque

from uipc_manip.obs import EXTRA_DIM, POINT_DIM, ObsSpec, constraint_flag, episode_fraction
from uipc_manip.sac import SACAgent, SACConfig


def flat_observation(spec: ObsSpec, violated: bool, elapsed_fraction: float = 0.) -> np.ndarray:
    points = np.zeros((2, 3), dtype=np.float32)
    flags = np.zeros((2, 4), dtype=np.float32)
    flags[:, 0] = 1.0
    return spec.pack_labeled(points, flags, np.zeros(3), np.zeros(3), attached=True,
                             violated=violated, elapsed_fraction=elapsed_fraction)


def test_the_slot_is_opt_in_and_one_float_wide():
    plain, flagged = ObsSpec(768), ObsSpec(768, constraint_flag=True)
    assert plain.dim == 768 * POINT_DIM + EXTRA_DIM == 5383
    assert flagged.dim == plain.dim + 1
    assert plain.extra_dim == EXTRA_DIM and flagged.extra_dim == EXTRA_DIM + 1


def test_a_violation_cannot_be_recorded_without_a_slot_to_record_it_in():
    spec = ObsSpec(8)
    with pytest.raises(ValueError, match="no constraint slot"):
        flat_observation(spec, violated=True)
    # and reading one back is refused rather than answered with a silent zero
    with pytest.raises(ValueError, match="does not carry a constraint flag"):
        constraint_flag(flat_observation(spec, violated=False), spec)


def test_the_flag_round_trips_through_the_observation():
    spec = ObsSpec(8, constraint_flag=True)
    assert float(constraint_flag(flat_observation(spec, violated=False), spec)) == 0.0
    assert float(constraint_flag(flat_observation(spec, violated=True), spec)) == 1.0
    batch = np.stack([flat_observation(spec, False), flat_observation(spec, True)])
    assert constraint_flag(batch, spec).tolist() == [0.0, 1.0]
    assert constraint_flag(torch.from_numpy(batch), spec).tolist() == [0.0, 1.0]


def agent(**cfg) -> SACAgent:
    spec = ObsSpec(8, constraint_flag=True, episode_clock=cfg.get("constraint_episode_horizon", 0) > 0)
    return SACAgent(spec, 3, SACConfig(hidden_dim=16, batch_size=4, **cfg), "cpu")


def test_the_cost_is_the_rising_edge_of_the_flag():
    a = agent(constraint_lambda_lr=0.1)
    spec = a.spec
    before = torch.from_numpy(np.stack([
        flat_observation(spec, False),   # still valid, stays valid
        flat_observation(spec, False),   # the decision that breaks it
        flat_observation(spec, True),    # already broken
    ]))
    after = torch.from_numpy(np.stack([
        flat_observation(spec, False),
        flat_observation(spec, True),
        flat_observation(spec, True),
    ]))
    assert a._transition_cost(before, after).tolist() == [0.0, 1.0, 0.0]
    # The constraint is paid once: a transition inside an already-failed episode is free,
    # so an episode can contribute at most 1 to the expected cost.
    assert float(a._transition_cost(after, before).sum()) == 0.0


def test_an_unflagged_spec_reports_no_cost_at_all():
    a = SACAgent(ObsSpec(8), 3, SACConfig(hidden_dim=16, batch_size=4), "cpu")
    assert a._transition_cost(torch.zeros(2, a.spec.dim), torch.zeros(2, a.spec.dim)) is None


def test_a_constrained_objective_requires_the_flag():
    with pytest.raises(ValueError, match="needs the observation's constraint flag"):
        SACAgent(ObsSpec(8), 3, SACConfig(hidden_dim=16, constraint_lambda_lr=0.1), "cpu")


def test_the_multiplier_rises_on_violations_and_decays_without_them():
    a = agent(constraint_lambda_lr=0.5, constraint_lambda_init=1.0, constraint_budget=0.0)
    reward = torch.ones(4)
    cost = torch.tensor([1.0, 0.0, 0.0, 0.0])

    penalised, stats = a.constrained_reward(reward, cost)
    # priced at the multiplier in force for this update, not at collection time
    assert penalised.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert stats["batch_constraint_cost"] == 0.25
    assert stats["constraint_lambda"] == pytest.approx(1.0 + 0.5 * 0.25)

    a.constraint_lambda = 1.0
    _, stats = a.constrained_reward(reward, torch.zeros(4))
    assert stats["constraint_lambda"] == pytest.approx(1.0)  # at budget 0 with no violations: no change

    a.constraint_lambda = 0.1
    a.cfg.constraint_budget = 0.5
    for _ in range(10):
        _, stats = a.constrained_reward(reward, torch.zeros(4))
    assert stats["constraint_lambda"] == 0.0  # a satisfied constraint stops being priced


def test_the_multiplier_is_capped():
    a = agent(constraint_lambda_lr=10.0, constraint_lambda_max=2.0)
    for _ in range(5):
        _, stats = a.constrained_reward(torch.ones(4), torch.ones(4))
    assert stats["constraint_lambda"] == 2.0


def test_the_penalty_scales_with_the_multiplier():
    a = agent(constraint_lambda_lr=0.0, constraint_lambda_init=3.0)
    penalised, _ = a.constrained_reward(torch.zeros(2), torch.tensor([1.0, 0.0]))
    assert penalised.tolist() == [-3.0, 0.0]


@pytest.mark.parametrize("violation_decision", [1, 2])
@pytest.mark.parametrize("reset_on_done", [True, False])
def test_step_records_cost_on_the_breaking_decision_including_terminal(monkeypatch, violation_decision, reset_on_done):
    from uipc_manip import dressing_env as module

    # Exercise the real step/terminal-observation wiring without a GPU solver.
    env = module.GenesisIPCDressingEnv.__new__(module.GenesisIPCDressingEnv)
    env.cfg = module.DressingConfig(horizon=2, action_repeat=2, constraint_objective=True, anchor_tether_m=None)
    env.num_envs = 1
    env.spec = ObsSpec(8, constraint_flag=True)
    env._anchor = np.zeros((1, 3))
    env._offsets = np.zeros((1, 1, 3))
    env._violated = np.zeros(1, dtype=bool)
    env._episode_step = 0
    env._decision_times = deque()
    env._test_tick = 0
    cell = SimpleNamespace(arm_points=np.ones((1, 3)), opening_idx=[0], finger=np.zeros(3),
                           shoulder=np.ones(3), elbow=np.ones(3), garment="test", human=0, name="test")
    env.cells = [cell]
    positions = [np.zeros((1, 3))]
    env.positions = lambda: positions
    env._update_targets = env._check_world = lambda: None
    env._sim_step = lambda: setattr(env, "_test_tick", env._test_tick + 1)
    # Only the first substep exceeds tolerance; the final position has recovered.
    env._tracking_error = lambda i, p: 0.03 if env._test_tick == 2 * violation_decision - 1 else 0.001
    env._actual_tcp = lambda i, p: np.zeros(3)
    env._progress = lambda p: [SimpleNamespace(reward=0., upperarm_ratio=0., forearm_ratio=0.,
                                              task_reward=0., on_forearm=False, on_upperarm=False, collision=0.)]
    env._privileged_state = lambda p, pr: np.zeros((1, 1))
    env.observation = lambda p=None: np.stack([flat_observation(env.spec, bool(env._violated[0]))])
    def reset():
        env._violated[:] = False
        env._episode_step = env._test_tick = 0
        return env.observation()
    env.reset = reset
    monkeypatch.setattr(module, "opening_threaded", lambda *a: (False, None))
    monkeypatch.setattr(module, "early_turn", lambda *a: False)
    obs = env.observation()
    for decision in (1, 2):
        next_obs, _, done, infos = env.step(np.zeros((1, 6)), reset_on_done=reset_on_done)
        after = infos[0].get("terminal_obs", next_obs[0])
        reconstructed = float(constraint_flag(after, env.spec) - constraint_flag(obs[0], env.spec))
        assert reconstructed == infos[0]["constraint_cost"] == float(decision == violation_decision)
        obs = next_obs
    assert bool(done[0])
    assert float(constraint_flag(infos[0]["terminal_obs"], env.spec)) == 1.
    assert float(constraint_flag(next_obs[0], env.spec)) == float(not reset_on_done)


def test_checkpoint_preserves_the_learned_multiplier(tmp_path):
    a = agent(constraint_lambda_lr=0.1)
    a.constraint_lambda = 7.25
    path = a.save(tmp_path / "agent.pt", step=10)
    restored = agent(constraint_lambda_lr=0.1)
    restored.load(path)
    assert restored.constraint_lambda == 7.25


def test_legacy_constrained_checkpoint_cannot_silently_reset_the_multiplier(tmp_path):
    a = agent(constraint_lambda_lr=0.1)
    path = a.save(tmp_path / "agent.pt", step=10)
    payload = a.read_checkpoint(path)
    del payload["constraint_lambda"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="requires the saved constraint_lambda"):
        a.load(path)
    a.load(path, load_optimizers=False)  # Actor-only evaluation remains possible.


def test_update_applies_a_fixed_nonzero_penalty(monkeypatch):
    a = agent(constraint_lambda_lr=0.0, constraint_lambda_init=3.0)
    cost = torch.tensor([1., 0., 0., 0.])
    sample = (None, torch.zeros(4, 3), torch.zeros(4, 1), None, None, None, None, None, None, None, cost)
    monkeypatch.setattr(a, "_sample_single", lambda replay: sample)
    received = []
    monkeypatch.setattr(a, "_update_critic", lambda obs, action, reward, *args: received.append(reward.clone()) or {})
    monkeypatch.setattr(a, "_finish_update", lambda stats, *args, **kwargs: stats)
    a._update(None)
    assert received[0].flatten().tolist() == [-3., 0., 0., 0.]


def test_episode_penalty_is_independent_of_violation_time():
    a = agent(constraint_episode_horizon=300, discount=0.995, constraint_lambda_init=2.)
    steps = np.array([0, 40, 299])
    before = torch.from_numpy(np.stack([flat_observation(a.spec, False, t / 300) for t in steps]))
    after = torch.from_numpy(np.stack([flat_observation(a.spec, True, (t + 1) / 300) for t in steps]))
    costs = a._transition_cost(before, after)
    penalized, _ = a.constrained_reward(torch.zeros(3), costs)
    np.testing.assert_allclose(0.995 ** steps * penalized.numpy(), [-2., -2., -2.], rtol=2e-6)
    assert episode_fraction(before, a.spec).tolist() == pytest.approx((steps / 300).tolist())
    assert constraint_flag(after, a.spec).tolist() == [1., 1., 1.]
    assert a._transition_cost(after, after).sum().item() == 0.


def test_episode_dual_uses_complete_episode_rate_not_replayed_costs(tmp_path):
    a = agent(constraint_episode_horizon=300, constraint_lambda_lr=2., constraint_budget=0.1)
    stats = a.update_constraint_dual([1., 0.])
    assert stats["episode_violation_rate"] == 0.5
    assert a.constraint_lambda == pytest.approx(0.8)
    for _ in range(10):
        a.constrained_reward(torch.zeros(4), torch.full((4,), 4.))
    assert a.constraint_lambda == pytest.approx(0.8)
    a.update_constraint_dual([0., 0.])
    assert a.constraint_lambda == pytest.approx(0.6)
    path = a.save(tmp_path / "episode.pt", 600)
    other = agent(constraint_episode_horizon=300, constraint_lambda_lr=2., constraint_budget=0.1)
    other.load(path)
    assert other.constraint_lambda == pytest.approx(0.6)
    assert other.constraint_episodes == 4
    from uipc_manip.physics_gradient_actor import load_agent
    analysis_agent = load_agent(path, 8, 3, "cpu")
    assert analysis_agent.spec.episode_clock and analysis_agent.spec.constraint_flag
    assert analysis_agent.constraint_lambda == pytest.approx(0.6)


def test_episode_mode_requires_a_visible_clock():
    with pytest.raises(ValueError, match="episode clock"):
        SACAgent(ObsSpec(8, constraint_flag=True), 3,
                 SACConfig(hidden_dim=16, constraint_episode_horizon=300), "cpu")


def test_episode_collector_excludes_heldout_cost_and_terminates_bootstrap(monkeypatch, tmp_path):
    from uipc_manip import train_sac, replay as replay_module, sac
    spec = ObsSpec(8, constraint_flag=True, episode_clock=True)
    recorded, dual_batches = [], []

    class Env:
        num_envs, action_dim, obs_dim = 2, 3, spec.dim
        descriptions = [dict(garment="test", human=i, config={}, build_seconds=0., settle_displacement_m=0.) for i in range(2)]

        def reset(self, seeds=None):
            self.t = 0
            return np.stack([flat_observation(spec, False), flat_observation(spec, False)])

        def step(self, actions):
            self.t += 1
            done = self.t == 2
            # The held-out slot violates; the training slot never does.
            obs = np.stack([flat_observation(spec, True, self.t / 2), flat_observation(spec, False, self.t / 2)])
            infos = [dict(constraint_cost=float(i == 0 and self.t == 1), constraint_violated=i == 0) for i in range(2)]
            if done:
                for i in range(2):
                    infos[i]["terminal_obs"] = obs[i].copy()
                obs = self.reset()
            return obs, np.zeros(2), np.full(2, done), infos

        def close(self):
            pass

    class Agent:
        def __init__(self, spec, action_dim, cfg, device):
            self.cfg, self.updates = cfg, 0
        def act(self, obs, deterministic):
            return np.zeros((2, 3))
        def train(self, training):
            pass
        def update_constraint_dual(self, costs):
            dual_batches.append(list(costs))
            return {}
        def save(self, path, step, metadata):
            return path

    class Replay(replay_module.FlatReplayBuffer):
        def add(self, obs, action, reward, next_obs, done, **kw):
            recorded.append((float(episode_fraction(obs, spec)), float(episode_fraction(next_obs, spec)), done))
            super().add(obs, action, reward, next_obs, done, **kw)

    monkeypatch.setattr(train_sac, "make_env", lambda args: Env())
    monkeypatch.setattr(train_sac, "built_cell_plan", lambda *a: ([("test", 0), ("test", 1)], [0]))
    monkeypatch.setattr(train_sac, "evaluate", lambda *a, **kw: dict(success_rate=0., mean_final_distance=1., mean_return=0.))
    monkeypatch.setattr(sac, "SACAgent", Agent)
    monkeypatch.setattr(replay_module, "FlatReplayBuffer", Replay)
    train_sac.main(["--task", "dressing", "--constraint-objective", "--constraint-episode",
                    "--horizon", "2", "--point-budget", "8", "--total-transitions", "4",
                    "--eval-freq", "2", "--checkpoint-interval", "2", "--replay-capacity", "16",
                    "--device", "cpu", "--work-dir", str(tmp_path)])
    assert dual_batches == [[0.], [0.]]
    assert recorded == [(0., 0.5, False), (0.5, 1., True)] * 2


def test_branch_restore_restores_violation_history(monkeypatch):
    from uipc_manip import physics_gradient_probe as probe
    h = SimpleNamespace(**{k: np.zeros(1) for k in ("stage", "_steps", "_align_steps", "_best_upper")})
    env = SimpleNamespace(_world=SimpleNamespace(dump=lambda: True, frame=lambda: 10,
                                                recover=lambda frame: True, retrieve=lambda: None),
                          _anchor=np.zeros((1, 3)), _offsets=[np.zeros((1, 3))],
                          _last_progress=[], _privileged=np.zeros((1, 1)), _violated=np.array([True]),
                          positions=lambda: [np.zeros((1, 3))], rngs=[np.random.default_rng(1)],
                          _update_targets=lambda: None, _decision_times=deque(),
                          cfg=SimpleNamespace(constraint_objective=True))
    monkeypatch.setattr(probe, "heuristic", lambda e: h)
    monkeypatch.setattr(probe, "measure", lambda e: {})
    snap = probe.take_snapshot(env, "prefix", 10)
    env._violated[:] = False
    probe.restore(env, snap)
    assert env._violated.tolist() == [True]
    del snap["constraint_violated"]
    with pytest.raises(ValueError, match="violation history"):
        probe.restore(env, snap)
