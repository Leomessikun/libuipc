"""The frame and feature mapping between our observation and Wang's (CPU, numpy only).

The network half of `wang_bridge` needs the `curl` conda environment; these cover the half that
decides whether the policy sees what it was trained on. A wrong mapping does not raise -- it
produces a policy that looks like it fails to transfer -- so each convention is pinned here.
"""

import numpy as np
import pytest

from uipc_manip.wang_bridge import (
    ARM,
    CLOTH,
    GRIPPER,
    to_reference_cloud,
    up_axis_rotation,
    voxel_downsample,
)
from uipc_manip.obs import FLAG_DEFORMABLE, FLAG_GOAL, FLAG_MARKER, FLAG_TOOL


def test_our_up_becomes_their_up():
    # dress_env.py:643 has gravity = [0, 1, 0]; ours is (0, 0, -9.8).
    assert (np.array([0.0, 0.0, 1.0]) @ up_axis_rotation(0.0).T) == pytest.approx([0.0, 1.0, 0.0])


def test_the_rotation_is_a_rotation_at_every_yaw():
    for yaw in (0.0, 37.0, 90.0, 180.0, -125.0):
        r = up_axis_rotation(yaw)
        assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_yaw_turns_about_their_vertical_and_leaves_it_alone():
    for yaw in (0.0, 45.0, 90.0, 210.0):
        assert (np.array([0.0, 0.0, 1.0]) @ up_axis_rotation(yaw).T) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)
    # A quarter turn sends our +x into their -z half-space rather than leaving it fixed.
    # A quarter turn about our z takes +x to +y, which the map then sends to their -z.
    turned = np.array([1.0, 0.0, 0.0]) @ up_axis_rotation(90.0).T
    assert turned == pytest.approx([0.0, 0.0, -1.0], abs=1e-12)


def _observation(n_arm=40, n_cloth=60, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.normal(0, 0.2, (n_arm + n_cloth, 3)), np.zeros((1, 3))], axis=0)
    flags = np.zeros((len(pos), 4), dtype=np.float32)
    flags[:n_arm, FLAG_MARKER] = 1.0
    flags[n_arm:n_arm + n_cloth, FLAG_DEFORMABLE] = 1.0
    flags[-1, FLAG_TOOL] = 1.0
    return pos, flags


def test_the_three_point_types_map_across():
    pos, flags = _observation()
    _, x = to_reference_cloud(pos, flags, voxel=0.0)
    assert x[:40, ARM].all() and x[40:100, CLOTH].all() and x[100, GRIPPER] == 1.0
    assert x.sum() == len(x)                      # exactly one type per point
    assert x.shape[1] == 3                        # their pc_feature_dim


def test_the_goal_point_is_dropped_because_they_have_none():
    pos, flags = _observation()
    flags[5, FLAG_MARKER], flags[5, FLAG_GOAL] = 0.0, 1.0
    _, x = to_reference_cloud(pos, flags, voxel=0.0)
    assert len(x) == len(pos) - 1


def test_the_gripper_survives_a_voxel_that_would_merge_it():
    # One point per occupied voxel; the gripper sits at the origin among dense cloth and must stay.
    pos = np.concatenate([np.zeros((30, 3)) + 1e-4, np.zeros((1, 3))], axis=0)
    flags = np.zeros((31, 4), dtype=np.float32)
    flags[:30, FLAG_DEFORMABLE] = 1.0
    flags[30, FLAG_TOOL] = 1.0
    _, x = to_reference_cloud(pos, flags, voxel=0.0625)
    assert x[:, GRIPPER].sum() == 1.0


def test_an_observation_without_a_tool_point_is_an_error_not_a_guess():
    pos, flags = _observation()
    flags[-1, FLAG_TOOL] = 0.0
    with pytest.raises(ValueError, match="gripper point"):
        to_reference_cloud(pos, flags, voxel=0.0)


def test_voxel_downsampling_thins_to_one_point_per_cell():
    # Voxel centres, not corners: points jittered across a cell boundary belong to two cells.
    grid = np.stack(np.meshgrid(*[(np.arange(4) + 0.5) * 0.0625] * 3, indexing="ij"), -1).reshape(-1, 3)
    dense = np.repeat(grid, 5, axis=0) + np.random.default_rng(0).normal(0, 1e-4, (len(grid) * 5, 3))
    assert len(voxel_downsample(dense, 0.0625)) == len(grid)
    assert len(voxel_downsample(dense, 0.0)) == len(dense)        # off means keep everything


def test_mismatched_positions_and_features_are_refused():
    pos, flags = _observation()
    with pytest.raises(ValueError, match="feature rows"):
        to_reference_cloud(pos, flags[:-1], voxel=0.0)
