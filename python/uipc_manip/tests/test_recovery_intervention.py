"""The deployment-side stall detector and the full-episode summary."""
from __future__ import annotations

import numpy as np
import pytest

from uipc_manip import recovery_intervention as ri
from uipc_manip.tests.test_decision_features import BUDGET, pack


def feed(detector, steps, motion_m, commanded_m):
    """Advance the detector with a constant garment motion and commanded translation."""
    position = np.zeros(3)
    for _ in range(steps):
        position = position + np.array([motion_m, 0.0, 0.0])
        detector.update(position, commanded_m)
    return detector


def test_a_responding_garment_does_not_trigger():
    d = feed(ri.StallDetector(window=4, threshold=0.15), 8, motion_m=0.004, commanded_m=0.008)
    assert d.response() == pytest.approx(0.5)
    assert not d.triggered()


def test_a_stalled_garment_triggers():
    d = feed(ri.StallDetector(window=4, threshold=0.15), 8, motion_m=0.0002, commanded_m=0.008)
    assert d.response() == pytest.approx(0.025)
    assert d.triggered()


def test_standing_still_is_not_a_stall():
    d = feed(ri.StallDetector(window=4, threshold=0.15, min_command_m=0.002), 8, motion_m=0.0, commanded_m=0.0)
    assert d.response() is None
    assert not d.triggered()


def test_the_window_must_fill_before_the_detector_speaks():
    d = feed(ri.StallDetector(window=6, threshold=0.15), 3, motion_m=0.0, commanded_m=0.008)
    assert d.response() is None
    assert not d.triggered()


def test_firing_starts_a_cooldown_and_clears_the_window():
    d = feed(ri.StallDetector(window=4, threshold=0.15, cooldown=10), 8, motion_m=0.0001, commanded_m=0.008)
    assert d.triggered()
    d.fired()
    assert d.response() is None
    # Six decisions refill the window but the cooldown still blocks a second trigger.
    feed(d, 6, motion_m=0.0001, commanded_m=0.008)
    assert d.response() is not None
    assert not d.triggered()
    feed(d, 5, motion_m=0.0001, commanded_m=0.008)
    assert d.triggered()


def test_reset_forgets_everything():
    d = feed(ri.StallDetector(window=4, threshold=0.15), 8, motion_m=0.0, commanded_m=0.008)
    d.reset()
    assert d.response() is None and not d.triggered()


def test_the_detector_reads_its_centroid_from_the_deployed_observation():
    obs = pack(cloth=[(0.0, 0.0, 0.02), (0.0, 0.0, 0.04)], arm=[(0.1, 0.0, 0.0)], tool=(0.5, 0.2, 1.0))
    assert np.allclose(ri.observation_centroid(obs, BUDGET), [0.5, 0.2, 1.03])


def test_invalid_detector_settings_are_rejected():
    for kwargs in (dict(window=1), dict(threshold=0.0), dict(cooldown=-1)):
        with pytest.raises(ValueError):
            ri.StallDetector(**kwargs)


def test_episode_summary_applies_the_coverage_and_grasp_rule():
    rows = [dict(upperarm_ratio=c, tracking_error=t) for c, t in
            [(0.3, 0.001)] * 10 + [(0.8, 0.001)] * 12]
    s = ri.episode_summary(rows)
    assert s["success"] and s["sustained_coverage"] == pytest.approx(0.8)
    slipped = [dict(upperarm_ratio=c, tracking_error=t) for c, t in [(0.8, 0.03)] * 12]
    assert not ri.episode_summary(slipped)["success"]
    dropped = [dict(upperarm_ratio=c, tracking_error=0.001) for c in [0.9] * 6 + [0.2] * 6]
    assert not ri.episode_summary(dropped)["success"]
    with pytest.raises(ValueError):
        ri.episode_summary(rows[:5])
