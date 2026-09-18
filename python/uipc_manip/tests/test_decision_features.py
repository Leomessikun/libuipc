"""Decision features: the three information sets and the small ridge used to compare them."""
from __future__ import annotations

import numpy as np
import pytest

from uipc_manip import decision_features as df
from uipc_manip.obs import EXTRA_DIM, FLAG_DEFORMABLE, FLAG_MARKER, POINT_DIM

BUDGET = 16


def pack(cloth, arm, goal_rel=(0.1, 0.0, 0.0), tool=(0.5, 0.2, 1.0), attached=True):
    """Build one packed observation the way the dressing environment does."""
    block = np.zeros((BUDGET, POINT_DIM), dtype=np.float32)
    points = list(arm) + list(cloth)
    for i, p in enumerate(points):
        block[i, :3] = p
        block[i, 3 + (FLAG_MARKER if i < len(arm) else FLAG_DEFORMABLE)] = 1.0
    extra = np.zeros(EXTRA_DIM, dtype=np.float32)
    extra[0:3], extra[3:6], extra[6] = tool, goal_rel, float(attached)
    return np.concatenate([block.reshape(-1), extra])


def test_unpack_rejects_an_observation_of_the_wrong_length():
    with pytest.raises(ValueError):
        df.unpack_observation(np.zeros(10), BUDGET)


def test_observation_features_separate_cloth_from_arm_points():
    obs = pack(cloth=[(0.0, 0.0, 0.02), (0.0, 0.0, 0.04)], arm=[(0.1, 0.0, 0.0)])
    f = dict(zip(df.OBSERVATION_FEATURE_NAMES, df.observation_features(obs, BUDGET), strict=True))
    assert f["cloth_fraction"] == pytest.approx(2 / BUDGET)
    assert f["arm_fraction"] == pytest.approx(1 / BUDGET)
    assert f["cloth_centroid_z"] == pytest.approx(0.03)
    assert f["arm_centroid_x"] == pytest.approx(0.1)
    assert f["goal_distance"] == pytest.approx(0.1)
    assert f["attached"] == pytest.approx(1.0)


def test_observation_features_measure_the_gap_between_cloth_and_arm():
    near = df.observation_features(pack(cloth=[(0.0, 0.0, 0.01)], arm=[(0.0, 0.0, 0.0)]), BUDGET)
    far = df.observation_features(pack(cloth=[(0.0, 0.0, 0.50)], arm=[(0.0, 0.0, 0.0)]), BUDGET)
    names = list(df.OBSERVATION_FEATURE_NAMES)
    assert near[names.index("cloth_arm_min_gap")] == pytest.approx(0.01)
    assert near[names.index("cloth_arm_close_fraction")] == pytest.approx(1.0)
    assert far[names.index("cloth_arm_close_fraction")] == pytest.approx(0.0)


def test_observation_features_survive_an_empty_segment():
    f = df.observation_features(pack(cloth=[], arm=[(0.1, 0.0, 0.0)]), BUDGET)
    assert np.isfinite(f).all()
    assert f[0] == 0.0


def test_history_features_report_the_garment_response_per_commanded_metre():
    # The cloth centroid moves 1 cm while 2 cm of translation is commanded: a response of 0.5.
    first = pack(cloth=[(0.0, 0.0, 0.00)], arm=[(0.1, 0.0, 0.0)])
    second = pack(cloth=[(0.0, 0.0, 0.01)], arm=[(0.1, 0.0, 0.0)])
    actions = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    f = df.history_features(np.stack([first, second]), actions, BUDGET, max_translation=0.02)
    per_frame = len(df.OBSERVATION_FEATURE_NAMES)
    assert f.size == 2 * per_frame + 6 + 3
    assert f[-3] == pytest.approx(0.01)   # metres moved
    assert f[-2] == pytest.approx(0.5)    # moved per commanded metre
    assert f[-1] == pytest.approx(0.02)   # metres commanded


def test_history_features_reject_a_mismatched_window():
    obs = np.stack([pack(cloth=[(0.0, 0.0, 0.0)], arm=[(0.1, 0.0, 0.0)])] * 2)
    with pytest.raises(ValueError):
        df.history_features(obs, np.zeros((2, 6)), BUDGET, 0.02)


def test_privileged_features_are_the_state_itself():
    priv = np.arange(35, dtype=np.float32)
    assert np.array_equal(df.privileged_features(priv), priv.astype(np.float64))


def test_standardize_leaves_constant_columns_alone():
    train = np.array([[1.0, 5.0], [3.0, 5.0]])
    a, b = df.standardize(train, np.array([[2.0, 5.0]]))
    assert np.allclose(a[:, 0], [-1.0, 1.0])
    assert np.allclose(a[:, 1], 0.0)
    assert np.allclose(b, [[0.0, 0.0]])


def test_ridge_recovers_a_linear_rule_and_shrinks_toward_the_mean():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((40, 3))
    y = x @ np.array([2.0, -1.0, 0.0]) + 0.5
    weights = df.ridge_fit(x, y, alpha=1e-6)
    assert np.allclose(weights[:3], [2.0, -1.0, 0.0], atol=1e-3)
    assert weights[-1] == pytest.approx(0.5, abs=1e-3)
    assert np.allclose(df.ridge_predict(weights, x), y, atol=1e-3)
    heavy = df.ridge_fit(x, y, alpha=1e6)
    assert abs(heavy[0]) < 0.01 and heavy[-1] == pytest.approx(y.mean(), abs=1e-3)
