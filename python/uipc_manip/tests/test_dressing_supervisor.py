"""The teacher supervisor: reading the sleeve's place on the arm and the three overrides."""
from __future__ import annotations

import numpy as np
import pytest

from uipc_manip import dressing_supervisor as ds
from uipc_manip.dressing_privileged import PRIVILEGED_DIM


def privileged(*, length=0.30, shoulder=(0.30, 0.25), centre=(0.15, 0.0, 0.0), radius=0.09,
               tracking_cm=0.5, upperarm_ratio=0.0):
    """One privileged vector with the blocks this module reads."""
    row = np.zeros(PRIVILEGED_DIM)
    row[slice(*ds.OFFSETS["arm"])] = [length, shoulder[0], shoulder[1]]
    row[slice(*ds.OFFSETS["opening_center"])] = centre
    row[slice(*ds.OFFSETS["opening_radius"])] = [radius, 0.0]
    row[slice(*ds.OFFSETS["tracking_error"])] = tracking_cm
    row[slice(*ds.OFFSETS["progress"])] = [0.0, upperarm_ratio, 0.0, 0.0, 0.0]
    return row


def test_the_blocks_are_read_at_the_layout_s_own_offsets():
    row = privileged(radius=0.07, tracking_cm=1.5)
    assert ds.read(row, "opening_radius")[0] == pytest.approx(0.07)
    assert ds.read(row, "tracking_error")[0] == pytest.approx(1.5)


def test_sleeve_position_places_the_opening_along_the_arm():
    on_forearm = ds.sleeve_position(privileged(centre=(0.15, 0.0, 0.0)))
    assert on_forearm["segment"] == "forearm"
    assert on_forearm["arc"] == pytest.approx(0.15 / 0.55, abs=1e-6)
    assert on_forearm["lateral"] == pytest.approx(0.0)
    past_elbow = ds.sleeve_position(privileged(centre=(0.30, 0.10, 0.0)))
    assert past_elbow["segment"] == "upperarm"
    assert past_elbow["arc"] == pytest.approx((0.30 + 0.10) / 0.55, abs=1e-6)


def test_containment_is_the_lateral_offset_in_ring_radii():
    state = ds.sleeve_position(privileged(centre=(0.15, 0.0, 0.045), radius=0.09))
    assert state["lateral"] == pytest.approx(0.045)
    assert state["containment"] == pytest.approx(0.5)
    assert np.allclose(ds.sleeve_position(privileged(centre=(0.15, 0.0, 0.045)))["outward"], [0.0, 0.0, 0.045])


def test_a_strained_grasp_halves_whatever_was_commanded():
    supervisor = ds.TeacherSupervisor(grasp_cm=1.2)
    action, reason = supervisor.command(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                        privileged(tracking_cm=1.5))
    assert reason == "grasp"
    assert action[0] == pytest.approx(0.5)


def test_a_drifting_ring_is_commanded_back_toward_the_arm():
    supervisor = ds.TeacherSupervisor(containment=0.6, hold=4)
    action, reason = supervisor.command(np.zeros(6), privileged(centre=(0.15, 0.0, 0.08), radius=0.09))
    assert reason == "centre"
    # The ring sits above the centreline, so the override points back down toward it.
    assert action[2] < 0 and abs(action[0]) < 1e-9
    # The plan holds for the remaining decisions.
    assert supervisor.command(np.ones(6), privileged(centre=(0.15, 0.0, 0.08)))[1] == "plan"


def test_a_stall_is_commanded_along_the_arm_and_needs_a_full_window():
    supervisor = ds.TeacherSupervisor(stall_window=4, stall_arc=0.02, hold=3)
    row = privileged(centre=(0.15, 0.0, 0.0))
    for _ in range(4):
        assert supervisor.command(np.zeros(6), row)[1] == "teacher"
    action, reason = supervisor.command(np.zeros(6), row)
    assert reason == "stall"
    assert action[0] == pytest.approx(1.0)   # along the forearm, toward the elbow


def test_progress_prevents_the_stall_override():
    supervisor = ds.TeacherSupervisor(stall_window=4, stall_arc=0.02, hold=3)
    for step in range(6):
        row = privileged(centre=(0.10 + 0.02 * step, 0.0, 0.0))
        assert supervisor.command(np.zeros(6), row)[1] == "teacher"


def test_a_nearly_dressed_sleeve_is_not_treated_as_stalled():
    supervisor = ds.TeacherSupervisor(stall_window=3, hold=2)
    row = privileged(centre=(0.30, 0.24, 0.0), upperarm_ratio=0.99)
    for _ in range(5):
        assert supervisor.command(np.zeros(6), row)[1] == "teacher"


def test_reset_clears_the_window_the_plan_and_the_reasons():
    supervisor = ds.TeacherSupervisor(stall_window=3, hold=3)
    row = privileged()
    for _ in range(5):
        supervisor.command(np.zeros(6), row)
    supervisor.reset()
    assert not supervisor.reasons and not supervisor._plan and not supervisor._arc


def test_invalid_settings_are_rejected():
    for kwargs in (dict(stall_window=1), dict(hold=0), dict(containment=0.0)):
        with pytest.raises(ValueError):
            ds.TeacherSupervisor(**kwargs)
