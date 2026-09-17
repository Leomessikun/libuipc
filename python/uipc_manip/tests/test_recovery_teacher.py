import numpy as np
import pytest

from uipc_manip.recovery_teacher import cap_commands, route_summary, verified_improvement


def rows(values, valid=True):
    return [dict(slot=i, sustained_coverage=x, final_upperarm_ratio=x,
                 whole_episode_grasp_valid=valid, sim_error=False) for i, x in enumerate(values)]


def test_verification_requires_improvement_over_both_controls_on_every_slot():
    assert verified_improvement(rows([.9, .85]), rows([.3, .4]), rows([.5, .6]))
    assert not verified_improvement(rows([.9, .85]), rows([.3, .4]), rows([.5, .84]))
    assert not verified_improvement(rows([.9, .65]), rows([.3, .4]), rows([.5, .6]))
    assert not verified_improvement(rows([.9, .85], valid=False), rows([.3, .4]), rows([.5, .6]))


def test_high_final_coverage_does_not_override_a_failed_continuation():
    candidate = rows([.4, .9])
    candidate[0]['final_upperarm_ratio'] = .99
    assert not verified_improvement(candidate, rows([.1, .2]), rows([.1, .2]))
    assert route_summary(candidate)['score'] == .4


def test_verification_rejects_nonfinite_and_unmatched_comparisons():
    assert not verified_improvement(rows([np.nan]), rows([.1]), rows([.2]))
    assert not verified_improvement(rows([.9]), rows([np.nan]), rows([.2]))
    with pytest.raises(ValueError, match='matched'):
        verified_improvement(rows([.9]), [], rows([.2]))
    other = rows([.1, .2])[::-1]
    with pytest.raises(ValueError, match='slot'):
        verified_improvement(rows([.9, .9]), other, rows([.2, .2]))


def test_route_caps_apply_same_translation_rotation_limits_without_active_x_rotation():
    action = np.array([[1., -1., .5, 1., -.5, .8], [0., 0., 0., 0., 0., 0.]])
    got = cap_commands(action, .7, .3)
    assert np.max(np.linalg.norm(got[:,:3], axis=1)) <= .700001
    assert np.max(np.linalg.norm(got[:,3:], axis=1)) <= .300001
    assert np.all(got[:,3] == 0) and np.array_equal(got[1], action[1])
    assert action[0,3] == 1  # The caller's policy output is not mutated.
