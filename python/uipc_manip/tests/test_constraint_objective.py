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

from uipc_manip.obs import EXTRA_DIM, POINT_DIM, ObsSpec, constraint_flag
from uipc_manip.sac import SACAgent, SACConfig


def flat_observation(spec: ObsSpec, violated: bool) -> np.ndarray:
    points = np.zeros((2, 3), dtype=np.float32)
    flags = np.zeros((2, 4), dtype=np.float32)
    flags[:, 0] = 1.0
    return spec.pack_labeled(points, flags, np.zeros(3), np.zeros(3), attached=True, violated=violated)


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
    spec = ObsSpec(8, constraint_flag=True)
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
