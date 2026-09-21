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
