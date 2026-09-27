"""Keep grasp/task failure separate from validated full body motion."""
from copy import deepcopy

import numpy as np
import pytest

from probe_arm_motion import classify_failures, endpoint_reached, initial_state_difference
from run_motion_pilot import motion_smoke_gate


def test_common_endpoint_accepts_wrapped_interior_with_cuff_past_fingers():
    state = dict(sleeve_sections_wrapped=True, sleeve_armhole_upper_fraction=.72,
                 sleeve_wrapped=False, upperarm_ratio=.55)
    assert endpoint_reached(state, "interior_armhole", .7)
    assert not endpoint_reached(state, "legacy_ratio", .7)
    state["sleeve_sections_wrapped"] = False
    state["upperarm_ratio"] = .99
    assert not endpoint_reached(state, "interior_armhole", .7)


def test_first_grasp_failure_survives_recovery_and_later_solver_failure():
    physics, grasp = classify_failures(dict(grasp_valid=False), 4)
    assert physics is None and grasp["kind"] == "invalid_grasp" and grasp["step"] == 5
    physics, same_grasp = classify_failures(dict(grasp_valid=True), 5, grasp)
    assert physics is None and same_grasp == grasp
    physics, same_grasp = classify_failures(dict(sim_error=True, error="body tracking invalid"), 6, grasp)
    assert physics["kind"] == "invalid_physics" and same_grasp == grasp


@pytest.mark.parametrize("key", ["positions", "human_vertices", "tcp"])
def test_controller_comparison_rejects_different_initial_geometry(key):
    reference = dict(positions=np.zeros((2, 3)), human_vertices=np.zeros((4, 3)), tcp=np.zeros(3))
    state = deepcopy(reference)
    assert max(initial_state_difference(reference, state).values()) == 0
    state[key][...] += .001
    with pytest.raises(ValueError, match="reference initial state"):
        initial_state_difference(reference, state)


def fixture(tmp_path, *, frames=41, excursion=.01):
    human = np.zeros((frames, 2, 3), dtype=np.float32)
    human[:, :, 0] = np.linspace(0, excursion, frames)[:, None]
    path = tmp_path / "hold.npz"
    np.savez(path, human_vertices=human)
    result = dict(steps=frames - 1, body_motion_valid=True, failure=None,
                  body_tracking_max_m=.0008, body_tracking_tolerance_m=.002,
                  grasp_failure=None)
    return result, path


def test_full_tracking_smoke_can_pass_while_keeping_failed_grasp(tmp_path):
    result, path = fixture(tmp_path)
    result["grasp_failure"] = result["failure"] = dict(kind="invalid_grasp", step=5)
    gate = motion_smoke_gate(result, path)
    assert gate["passed"] and gate["grasp_failure"]["step"] == 5
    assert result["failure"]["kind"] == "invalid_grasp"


@pytest.mark.parametrize("condition", ["partial", "tracking", "solver", "no_motion", "nonfinite"])
def test_incomplete_or_invalid_body_motion_cannot_pass(tmp_path, condition):
    result, path = fixture(tmp_path, frames=6 if condition == "partial" else 41,
                           excursion=0 if condition == "no_motion" else .01)
    result = deepcopy(result)
    if condition == "tracking":
        result["body_tracking_max_m"] = .003
    elif condition == "solver":
        result["failure"] = dict(kind="invalid_physics")
    elif condition == "nonfinite":
        np.savez(path, human_vertices=np.full((41, 2, 3), np.nan))
    assert not motion_smoke_gate(result, path)["passed"]
