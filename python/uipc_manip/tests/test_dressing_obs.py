"""Observation budget regressions independent of the simulator."""

import numpy as np
import pytest

from uipc_manip.dressing_obs import sample_segmented_cloud


def test_dense_arm_cannot_remove_cloth_from_observation():
    arm = np.arange(900, dtype=np.float32).reshape(-1, 3)
    cloth = np.arange(300, dtype=np.float32).reshape(-1, 3) + 1000
    a, c = sample_segmented_cloud(arm, cloth, 64, np.random.default_rng(7))
    assert len(a) + len(c) == 64
    assert len(a) > 0 and len(c) > 0
    # The entire spatial extent remains eligible, rather than only a prefix.
    assert a[:, 0].max() > arm[len(a), 0]
    assert c[:, 0].max() > cloth[len(c), 0]
    assert len(np.unique(a, axis=0)) == len(a)
    assert len(np.unique(c, axis=0)) == len(c)
    a2, c2 = sample_segmented_cloud(arm, cloth, 64, np.random.default_rng(7))
    np.testing.assert_array_equal(a, a2)
    np.testing.assert_array_equal(c, c2)


@pytest.mark.parametrize("arm_count,cloth_count,budget", [(0, 20, 5), (20, 0, 5), (1, 20, 5), (20, 1, 5), (20, 20, 1)])
def test_cloud_budget_handles_empty_and_small_segments(arm_count, cloth_count, budget):
    a, c = sample_segmented_cloud(
        np.zeros((arm_count, 3)), np.ones((cloth_count, 3)), budget, np.random.default_rng(2)
    )
    assert len(a) + len(c) == budget
    if cloth_count:
        assert len(c) > 0
    if arm_count and budget > 1:
        assert len(a) > 0


def test_cloud_that_fits_preserves_values_and_rng_state():
    arm, cloth = np.zeros((3, 3)), np.ones((4, 3))
    rng = np.random.default_rng(4)
    before = rng.bit_generator.state
    a, c = sample_segmented_cloud(arm, cloth, 10, rng)
    assert a is arm and c is cloth
    assert rng.bit_generator.state == before
