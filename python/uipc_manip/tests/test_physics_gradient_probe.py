"""The physics-gradient probe's bookkeeping on a fake environment: differences per executed
metre under a movement lock, walk repeats and executed travel, the combined directions."""

import numpy as np
import pytest

from uipc_manip import physics_gradient_probe as probe


class LockedEnv:
    """Coverage rises with +x and force falls with +y; any move that would put the anchor past
    +3 mm in x is dropped, the way the environment's no-move collision rule drops a command."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"max_translation": 0.00866, "max_rotation": 0.08726646, "clip_rotation_to_yz": True})()
        self._anchor = np.zeros((1, 3))
        self._offsets = [np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01], [0.01, 0.01, 0.0]])]
        self.yaw = 0.0  # accumulated rotation about z; coverage also rises with it

    def outcome(self):
        a = self._anchor[0]
        return {"upperarm_ratio": 0.2 + 0.5 * a[0] + 0.1 * self.yaw, "opening_axis_m": 0.4 + a[0], "upperarm_axis_m": 0.1 + a[0],
                "net_normal_n": 600.0 - 40000.0 * a[1], "summed_normal_n": 0.0, "contact_energy": 1.0 - a[1],
                "reward": 0.0, "tracking_error": 0.02, "arm_clearance_m": 0.012 - a[0], "anchor": a.tolist()}

    def step(self, action):
        move = np.asarray(action[0][:3], dtype=np.float64) * self.cfg.max_translation
        if self._anchor[0, 0] + move[0] <= 0.003:
            self._anchor = self._anchor + move[None]
        rot = np.asarray(action[0][3:], dtype=np.float64) * self.cfg.max_rotation
        rot[0] = 0.0
        angle = np.linalg.norm(rot)
        if angle > 0:
            k = rot / angle
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
            self._offsets = [self._offsets[0] @ R.T]
            self.yaw += rot[2]
        return None, np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool), [{}]


@pytest.fixture
def env(monkeypatch):
    env = LockedEnv()
    monkeypatch.setattr(probe, "measure", lambda e: e.outcome())
    def restore(e, snap):
        e._anchor = np.array(snap["anchor"])[None]
        e._offsets = [np.array(snap["offsets"])]
        e.yaw = 0.0
        return 0.0
    monkeypatch.setattr(probe, "restore", restore)
    monkeypatch.setattr(probe, "heuristic", lambda e: type("H", (), {"actions": lambda self: np.array([[0, 0, 0, 0, 0, 0.5]])})())
    return env


def _snap(env):
    return {"anchor": [0.0, 0.0, 0.0], "offsets": env._offsets[0].copy(), "measure": env.outcome()}


def test_differences_are_reported_per_commanded_and_per_executed_metre(env):
    snap = _snap(env)
    g = probe.gradients(env, snap, np.zeros(6), [0.001, 0.005], 2)
    pe = g["per_epsilon"]["0.005"]
    # +5 mm in x is refused by the lock, so the commanded-metre difference is one-sided ...
    assert pe["mean_executed_m"]["plus"][0] == 0.0 and pe["mean_executed_m"]["minus"][0] == pytest.approx(0.005)
    assert pe["mean"]["upperarm_ratio"][0] == pytest.approx(0.25)
    # ... and the executed-metre one is the physics.
    assert pe["gradients"][0]["per_executed_metre"]["upperarm_ratio"][0] == pytest.approx(0.5)
    assert pe["gradients"][0]["per_executed_metre"]["net_normal_n"][1] == pytest.approx(-40000.0)
    assert g["locality_cosine"]["upperarm_ratio"][0][1] == pytest.approx(1.0)
    # 1 mm goes through in both directions, so there it is a central difference.
    assert probe.gradients(env, snap, np.zeros(6), [0.001], 1)["per_epsilon"]["0.001"]["mean_executed_m"]["plus"][0] == pytest.approx(0.001)


def test_walks_carry_executed_travel_repeats_and_the_combined_directions(env):
    snap = _snap(env)
    g = probe.gradients(env, snap, np.zeros(6), [0.001, 0.002], 1)
    u = probe.usefulness(env, snap, g, 6, 0.004, 0, repeats=2,
                         walks=("coverage_gradient", "combined_gradient", "release_then_advance", "hold", "random"))
    assert set(u["walks"]) == {"coverage_gradient", "combined_gradient", "release_then_advance", "hold", "random_0", "random_1", "random_2"}
    assert u["release_steps"] == 2 and u["walk_repeats"] == 2
    cov = u["walks"]["coverage_gradient"]
    assert cov["commanded_m"] == pytest.approx(0.024) and cov["executed_m"] == 0.0  # straight into the lock
    assert len(cov["repeats"]) == 2 and cov["repeat_std"]["delta_opening_axis_m"] == 0.0
    rta = u["walks"]["release_then_advance"]
    first, later = rta["rows"][0]["anchor"], rta["rows"][-1]["anchor"]
    assert first[1] > 0 and first[0] == 0.0, "release first: down the force gradient (+y), no x"
    assert later[1] == pytest.approx(first[1] * 2) and later[0] == pytest.approx(0.0), "then advance, refused by the lock in x"
    comb = u["walks"]["combined_gradient"]
    assert comb["rows"][0]["executed_m"] > 0 and comb["rows"][0]["anchor"][0] > 0 and comb["rows"][0]["anchor"][1] > 0
    assert all("executed_m" in r and "commanded_m" in r for r in comb["rows"])


def test_the_parser_selects_states_and_walks():
    args = probe.build_parser().parse_args(["--out", "x", "--states", "stall", "--walks", "combined_gradient", "release_then_advance",
                                            "--walk-repeats", "3"])
    assert args.states == ["stall"] and args.walks == ["combined_gradient", "release_then_advance"] and args.walk_repeats == 3
    assert probe.build_parser().parse_args(["--out", "x"]).walks == list(probe.WALKS)
    with pytest.raises(SystemExit):
        probe.build_parser().parse_args(["--out", "x", "--walks", "teleport"])


def test_rotation_differences_and_walks_use_the_live_axes(env):
    snap = _snap(env)
    g = probe.gradients(env, snap, np.zeros(6), [0.001, 0.002], 1)
    rg = probe.gradients(env, snap, np.zeros(6), [np.deg2rad(1.0), np.deg2rad(2.5)], 1, probe.ROTATION_AXES)
    assert rg["axes"] == [4, 5] and rg["unit"] == "rad"
    pe = rg["per_epsilon"][str(np.deg2rad(2.5))]
    # coverage rises with yaw (axis 5) only; the executed rotation equals the commanded one
    assert pe["mean"]["upperarm_ratio"][1] == pytest.approx(0.1, rel=1e-3) and abs(pe["mean"]["upperarm_ratio"][0]) < 1e-9
    assert pe["mean_executed"]["plus"][1] == pytest.approx(np.deg2rad(2.5), rel=1e-6)
    assert pe["gradients"][0]["per_executed_unit"]["upperarm_ratio"][1] == pytest.approx(0.1, rel=1e-3)
    u = probe.usefulness(env, snap, g, 4, 0.004, 0, walks=("rotation_gradient", "full_gradient", "axis_gradient_expert_rotation", "hold"),
                         rot_grad=rg, rot_magnitude_rad=np.deg2rad(2.5))
    assert u["rotation"]["rotation_direction_from"] == "upperarm_ratio" and u["rotation"]["axes"] == [4, 5]
    r = u["walks"]["rotation_gradient"]
    assert r["executed_m"] == 0.0 and r["executed_rad"] == pytest.approx(4 * np.deg2rad(2.5), rel=1e-5)
    assert r["delta_upperarm"] == pytest.approx(0.1 * 4 * np.deg2rad(2.5), rel=1e-3)
    f = u["walks"]["full_gradient"]
    assert f["executed_rad"] == pytest.approx(4 * np.deg2rad(2.5), rel=1e-5)
    assert f["executed_m"] == 0.0 and f["commanded_m"] == pytest.approx(0.016), "4 mm along +x runs into the lock, the rotation still executes"
    e = u["walks"]["axis_gradient_expert_rotation"]
    assert e["executed_rad"] == pytest.approx(4 * 0.5 * 0.08726646, rel=1e-5), "the expert's own rotation each step"
    assert e["rows"][0]["commanded_m"] == pytest.approx(0.004)
