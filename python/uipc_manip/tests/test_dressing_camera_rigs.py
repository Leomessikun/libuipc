"""Camera-rig geometry and visibility, independent of the simulator."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uipc_manip.dressing_obs import (
    RIG_MODES,
    SEG_ARM,
    SEG_BODY,
    SEG_CLOTH,
    BatchedDressingObservationBuilder,
    DressingObsConfig,
    DressingObservationBuilder,
    RigCamera,
    RigInputs,
    body_occluders,
    rig_cameras,
    rig_visible_mask,
)


def _t(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def _arm_scene(yaw=0.0):
    """A body facing +y with its right arm stretched along +x at shoulder height, turned by ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    shoulder = rz @ np.array([0.2, 0.0, 1.4])
    elbow = rz @ np.array([0.5, 0.0, 1.4])
    finger = rz @ np.array([0.8, 0.0, 1.4])
    lateral = rz @ np.array([0.4, 0.0, 0.0])
    return finger, elbow, shoulder, lateral


def _camera(**kw):
    base = dict(position=_t([[0.0, 0.0, 0.0]]), target=_t([[1.0, 0.0, 0.0]]), up=_t([[0.0, 0.0, 1.0]]),
                hfov_deg=60.0, vfov_deg=40.0, width=64, height=48, near_m=0.1, far_m=2.0, jitter_m=0.0)
    base.update(kw)
    return RigCamera(**base)


TOL = {"splat_m": 0.0, "self_tolerance_m": 0.02, "cross_tolerance_m": 0.005}


def test_stretch_head_stands_front_right_at_mast_height_and_turns_with_the_body():
    cfg = DressingObsConfig(mode="stretch3_head")
    for yaw in (0.0, 0.7, -2.0):
        finger, elbow, shoulder, lateral = _arm_scene(yaw)
        (cam,) = rig_cameras(cfg, _t([finger]), _t([elbow]), _t([shoulder]), _t([lateral]), _t([0.0]), _t([finger]), _t([np.eye(3)]))
        mid = 0.5 * (finger + shoulder)
        c, s = math.cos(yaw), math.sin(yaw)
        toward = np.array([c - s, s + c, 0.0]) / math.sqrt(2.0)  # the turned (forward + right) / sqrt(2)
        expected = mid + cfg.head_distance_m * toward
        expected[2] = cfg.head_height_m
        np.testing.assert_allclose(cam.position[0].numpy(), expected, atol=1e-5)
        np.testing.assert_allclose(cam.target[0].numpy(), mid, atol=1e-6)
        assert (cam.hfov_deg, cam.vfov_deg) == (58.0, 87.0)


def test_wrist_camera_rides_the_tool():
    cfg = DressingObsConfig(mode="stretch3_head_wrist")
    finger, elbow, shoulder, lateral = _arm_scene()
    tool = finger + np.array([0.05, 0.0, 0.05])
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for r in (np.eye(3), quarter):
        _, wrist = rig_cameras(cfg, _t([finger]), _t([elbow]), _t([shoulder]), _t([lateral]), _t([0.0]), _t([tool]), _t([r]))
        f = r @ np.array([-1.0, 0.0, 0.0])  # along the forearm, finger to elbow
        n = r @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(wrist.position[0].numpy(), tool - cfg.wrist_back_m * f + cfg.wrist_up_m * n, atol=1e-5)
        np.testing.assert_allclose(wrist.target[0].numpy(), tool + cfg.wrist_look_m * f, atol=1e-5)
        np.testing.assert_allclose(wrist.up[0].numpy(), n, atol=1e-6)
        assert (wrist.near_m, wrist.far_m) == tuple(cfg.wrist_range_m)


def test_field_of_view_and_range_limits():
    half_h, half_v = math.radians(30.0), math.radians(20.0)
    pts = np.array([
        [1.0, math.tan(half_h) * 0.98, 0.0],
        [1.0, math.tan(half_h) * 1.02, 0.0],
        [1.0, 0.0, math.tan(half_v) * 0.98],
        [1.0, 0.0, math.tan(half_v) * 1.02],
        [0.05, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    seg = torch.full((1, len(pts)), SEG_CLOTH)
    visible = rig_visible_mask(_t([pts]), seg, torch.ones(1, len(pts), dtype=torch.bool), _camera(), **TOL)[0].tolist()
    assert visible == [True, False, True, False, False, False, False]


def test_sleeve_hides_skin_live_but_not_in_the_static_capture():
    # Skin on the plane x = 1 and a garment patch 1 cm in front of its upper half, seen from the origin.
    grid = np.stack(np.meshgrid(np.linspace(-0.1, 0.1, 21), np.linspace(-0.1, 0.1, 21)), -1).reshape(-1, 2)
    arm = np.column_stack([np.full(len(grid), 1.0), grid])
    cloth = arm[arm[:, 1] > 0.0] - np.array([0.01, 0.0, 0.0])
    pts = np.concatenate([arm, cloth])
    seg = torch.as_tensor([SEG_ARM] * len(arm) + [SEG_CLOTH] * len(cloth))[None]
    valid = torch.ones_like(seg, dtype=torch.bool)
    cam = _camera(hfov_deg=30.0, vfov_deg=30.0, width=40, height=40)
    tol = {**TOL, "splat_m": 0.01}
    live = rig_visible_mask(_t([pts]), seg, valid, cam, **tol)[0].numpy()
    bare = rig_visible_mask(_t([pts]), seg, valid & (seg != SEG_CLOTH), cam, **tol)[0].numpy()
    under, clear = arm[:, 1] > 0.02, arm[:, 1] < -0.02
    assert not live[: len(arm)][under].any() and live[: len(arm)][clear].all()
    assert bare[: len(arm)].all()
    assert live[len(arm):].all()


def test_splats_keep_a_sparse_surface_opaque_up_close():
    # A garment sampled every 1.5 cm, 10 cm from the camera, in front of skin 1 cm behind it.
    grid = np.stack(np.meshgrid(np.linspace(-0.03, 0.03, 5), np.linspace(-0.03, 0.03, 5)), -1).reshape(-1, 2)
    cloth = np.column_stack([np.full(len(grid), 0.10), grid])
    skin = np.column_stack([np.full(len(grid), 0.11), grid + 0.0075])
    pts = np.concatenate([skin, cloth])
    seg = torch.as_tensor([SEG_ARM] * len(skin) + [SEG_CLOTH] * len(cloth))[None]
    valid = torch.ones_like(seg, dtype=torch.bool)
    cam = _camera(hfov_deg=87.0, vfov_deg=58.0, width=48, height=28, near_m=0.07, far_m=0.5)
    inner = (np.abs(grid) < 0.02).all(axis=1)
    bare = rig_visible_mask(_t([pts]), seg, valid, cam, **TOL)[0].numpy()[: len(skin)]
    splat = rig_visible_mask(_t([pts]), seg, valid, cam, **{**TOL, "splat_m": 0.01})[0].numpy()[: len(skin)]
    assert bare[inner].any()
    assert not splat[inner].any()


def test_body_hides_the_garment():
    pts = np.array([[1.0, 0.0, 0.0], [1.0, 0.05, 0.0], [0.9, 0.0, 0.0]])
    seg = torch.as_tensor([SEG_CLOTH, SEG_CLOTH, SEG_BODY])[None]
    visible = rig_visible_mask(_t([pts]), seg, torch.ones_like(seg, dtype=torch.bool), _camera(), **TOL)[0].tolist()
    assert visible == [False, True, True]


def test_body_occluders_drop_the_arm_itself():
    arm = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    body = np.array([[0.0, 0.0, 0.0], [0.1, 0.005, 0.0], [0.5, 0.0, 0.0]])
    np.testing.assert_allclose(body_occluders(body, arm, 0.015), [[0.5, 0.0, 0.0]])


def _rig_scene(b):
    rng = np.random.default_rng(0)
    finger, elbow, shoulder, lateral = _arm_scene()
    t = np.linspace(0.0, 1.0, 200)[:, None]
    arm = (finger + t * (shoulder - finger) + rng.normal(scale=0.03, size=(200, 3))).astype(np.float32)
    cloth = (finger + np.array([0.05, 0.0, 0.0]) + rng.normal(scale=0.06, size=(300, 3))).astype(np.float32)
    torso = (np.array([0.0, 0.0, 1.2]) + rng.normal(scale=[0.15, 0.1, 0.25], size=(400, 3))).astype(np.float32)
    rig = RigInputs(elbows=np.tile(elbow, (b, 1)), lateral=np.tile(lateral, (b, 1)), floors=np.zeros(b),
                    tools=np.tile(finger + np.array([0.1, 0.0, 0.1]), (b, 1)), tool_rotations=np.tile(np.eye(3), (b, 1, 1)),
                    bodies=[torso] * b)
    return [arm] * b, [cloth] * b, [finger] * b, [shoulder] * b, rig


@pytest.mark.parametrize("augment", [False, True])
@pytest.mark.parametrize("mode", RIG_MODES)
def test_every_rig_observes_arm_and_garment(mode, augment):
    arms, cloths, fingers, shoulders, rig = _rig_scene(3)
    builder = BatchedDressingObservationBuilder(DressingObsConfig(mode=mode, voxel_size_m=0.03), "cpu")
    out = builder.visible_points(arms, cloths, fingers, shoulders, [np.random.default_rng(i) for i in range(3)], augment, rig=rig)
    assert len(out) == 3
    for arm, cloth in out:
        assert arm.ndim == 2 and arm.shape[1] == 3 and cloth.shape[1] == 3
        assert len(arm) > 0 and len(cloth) > 0


def test_per_environment_builder_delegates_rig_modes():
    arms, cloths, fingers, shoulders, rig = _rig_scene(1)
    cfg = DressingObsConfig(mode="stretch3_head_wrist", voxel_size_m=0.03)
    single = DressingObservationBuilder(cfg, "cpu").visible_points(arms[0], cloths[0], fingers[0], shoulders[0], np.random.default_rng(0), False, rig=rig)
    batched = BatchedDressingObservationBuilder(cfg, "cpu").visible_points(arms, cloths, fingers, shoulders, [np.random.default_rng(0)], False, rig=rig)[0]
    for x, y in zip(single, batched, strict=True):
        np.testing.assert_allclose(x, y)


def test_rig_modes_need_rig_inputs():
    arms, cloths, fingers, shoulders, _ = _rig_scene(1)
    builder = BatchedDressingObservationBuilder(DressingObsConfig(mode="stretch3_head"), "cpu")
    with pytest.raises(ValueError, match="RigInputs"):
        builder.visible_points(arms, cloths, fingers, shoulders, [np.random.default_rng(0)], False)


def test_rig_fields_leave_the_legacy_modes_untouched():
    arms, cloths, fingers, shoulders, rig = _rig_scene(2)
    base = DressingObsConfig(image_wh=32, voxel_size_m=0.03)
    tweaked = DressingObsConfig(image_wh=32, voxel_size_m=0.03, static_arm=True, head_height_m=2.0, rig_splat_m=0.05, wrist_range_m=(0.0, 9.0))
    for augment in (False, True):
        a = BatchedDressingObservationBuilder(base, "cpu").visible_points(
            arms, cloths, fingers, shoulders, [np.random.default_rng(3), np.random.default_rng(4)], augment)
        b = BatchedDressingObservationBuilder(tweaked, "cpu").visible_points(
            arms, cloths, fingers, shoulders, [np.random.default_rng(3), np.random.default_rng(4)], augment, rig=rig)
        for (arm_a, cloth_a), (arm_b, cloth_b) in zip(a, b, strict=True):
            np.testing.assert_array_equal(arm_a, arm_b)
            np.testing.assert_array_equal(cloth_a, cloth_b)


def test_rig_inputs_from_cells_read_shoulders_floor_and_grip():
    finger, elbow, shoulder, lateral = _arm_scene()
    body = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 1.7]])
    cell = SimpleNamespace(elbow=elbow, shoulder=shoulder, human_points=body,
                           landmarks={"left_shoulder": shoulder - lateral, "right_shoulder": shoulder})
    offsets0 = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01], [0.01, 0.01, 0.0]])
    quarter = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rig = RigInputs.from_cells([cell], [finger], [offsets0 @ quarter.T], [offsets0])
    np.testing.assert_allclose(rig.lateral[0], lateral)
    assert rig.floors[0] == pytest.approx(0.05)
    np.testing.assert_allclose(rig.tool_rotations[0], quarter, atol=1e-9)
    np.testing.assert_allclose(rig.elbows[0], elbow)


def test_rig_inputs_fall_back_to_the_hips_without_a_left_shoulder():
    finger, elbow, shoulder, lateral = _arm_scene(0.7)
    pelvis = np.array([0.0, 0.0, 0.9])
    cell = SimpleNamespace(elbow=elbow, shoulder=shoulder, human_points=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]]),
                           landmarks={"right_shoulder": shoulder, "pelvis": pelvis, "right_hip": pelvis + 0.2 * lateral})
    rig = RigInputs.from_cells([cell], [finger], [np.eye(3)], [np.eye(3)])
    np.testing.assert_allclose(rig.lateral[0], 0.2 * lateral)
