"""The physics-gradient probe's bookkeeping on a fake environment: differences per executed
metre under a movement lock, walk repeats and executed travel, the combined directions."""

import numpy as np
import pytest

from uipc_manip import physics_gradient_probe as probe


class LockedEnv:
    """Coverage rises with +x and force falls with +y; any move that would put the anchor past
    +3 mm in x is dropped, the way the environment's no-move collision rule drops a command."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"max_translation": 0.00866})()
        self._anchor = np.zeros((1, 3))

    def outcome(self):
        a = self._anchor[0]
        return {"upperarm_ratio": 0.2 + 0.5 * a[0], "opening_axis_m": 0.4 + a[0], "upperarm_axis_m": 0.1 + a[0],
                "net_normal_n": 600.0 - 40000.0 * a[1], "summed_normal_n": 0.0, "contact_energy": 1.0 - a[1],
                "reward": 0.0, "tracking_error": 0.02, "arm_clearance_m": 0.012 - a[0], "anchor": a.tolist()}

    def step(self, action):
        move = np.asarray(action[0][:3], dtype=np.float64) * self.cfg.max_translation
        if self._anchor[0, 0] + move[0] <= 0.003:
            self._anchor = self._anchor + move[None]
        return None, np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool), [{}]


@pytest.fixture
def env(monkeypatch):
    env = LockedEnv()
    monkeypatch.setattr(probe, "measure", lambda e: e.outcome())
    monkeypatch.setattr(probe, "restore", lambda e, snap: (e.__setattr__("_anchor", np.array(snap["anchor"])[None]), 0.0)[1])
    return env


def test_differences_are_reported_per_commanded_and_per_executed_metre(env):
    snap = {"anchor": [0.0, 0.0, 0.0], "measure": env.outcome()}
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
    snap = {"anchor": [0.0, 0.0, 0.0], "measure": env.outcome()}
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
