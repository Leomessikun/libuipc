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


def _synthetic_scene(rng, n_arm, n_cloth):
    finger = np.array([0.0, 0.0, 1.0])
    shoulder = np.array([0.5, 0.1, 1.0])
    arm = finger + rng.uniform(-0.05, 0.05, size=(n_arm, 3)) + np.linspace(0, 1, n_arm)[:, None] * (shoulder - finger)
    cloth = finger + rng.normal(scale=0.08, size=(n_cloth, 3))
    return arm.astype(np.float32), cloth.astype(np.float32), finger, shoulder


@pytest.mark.parametrize("augment", [False, True])
def test_batched_visible_points_matches_the_per_environment_builder(augment):
    from uipc_manip.dressing_obs import BatchedDressingObservationBuilder, DressingObsConfig, DressingObservationBuilder

    cfg = DressingObsConfig(image_wh=32, voxel_size_m=0.03)
    single = DressingObservationBuilder(cfg, "cpu")
    batched = BatchedDressingObservationBuilder(cfg, "cpu")
    scenes = [_synthetic_scene(np.random.default_rng(10 + i), 60 + 7 * i, 200 - 13 * i) for i in range(5)]
    seeds = [100 + i for i in range(5)]
    expected = [
        single.visible_points(a, c, f, s, np.random.default_rng(seed), augment)
        for (a, c, f, s), seed in zip(scenes, seeds, strict=True)
    ]
    rngs = [np.random.default_rng(seed) for seed in seeds]
    got = batched.visible_points(
        [x[0] for x in scenes], [x[1] for x in scenes], [x[2] for x in scenes], [x[3] for x in scenes], rngs, augment
    )
    for (arm_e, cloth_e), (arm_g, cloth_g) in zip(expected, got, strict=True):
        assert arm_e.shape == arm_g.shape and cloth_e.shape == cloth_g.shape
        assert np.allclose(arm_e, arm_g, atol=1e-5) and np.allclose(cloth_e, cloth_g, atol=1e-5)
    # Both builders consumed the same random draws, so the generators agree afterwards.
    for (a, c, f, s), rng, seed in zip(scenes, rngs, seeds, strict=True):
        reference = np.random.default_rng(seed)
        single.visible_points(a, c, f, s, reference, augment)
        assert rng.bit_generator.state == reference.bit_generator.state
    assert any(len(a) for a, _ in got) and any(len(c) for _, c in got)


def test_nearest_body_distance_matches_the_pairwise_scan():
    from uipc_manip.dressing_reward import nearest_body_distance

    rng = np.random.default_rng(3)
    body = rng.normal(size=(2000, 3))
    cuff = rng.normal(size=(40, 3)) + 2.0
    brute = float(np.min(np.linalg.norm(cuff[:, None, :] - body[None, :, :], axis=2)))
    assert nearest_body_distance(cuff, body) == pytest.approx(brute, abs=1e-12)
    # A different body array gets its own tree.
    other = body * 0.5
    assert nearest_body_distance(cuff, other) == pytest.approx(
        float(np.min(np.linalg.norm(cuff[:, None, :] - other[None, :, :], axis=2))), abs=1e-12
    )
