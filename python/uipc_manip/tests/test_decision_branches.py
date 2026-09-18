"""Counterfactual branch primitives: directions, macro commands and outcome statistics."""
from __future__ import annotations

import numpy as np
import pytest

from uipc_manip import decision_branches as db
from uipc_manip.collect_decision_branches import state_id

FINGER = np.array([0.0, 0.0, 0.0])
ELBOW = np.array([0.30, 0.0, 0.0])
SHOULDER = np.array([0.30, 0.25, 0.0])


def test_centerline_projection_picks_the_nearer_segment():
    point, segment = db.centerline_projection(np.array([0.15, 0.05, 0.0]), FINGER, ELBOW, SHOULDER)
    assert segment == "forearm"
    assert np.allclose(point, [0.15, 0.0, 0.0])
    point, segment = db.centerline_projection(np.array([0.34, 0.20, 0.0]), FINGER, ELBOW, SHOULDER)
    assert segment == "upperarm"
    assert np.allclose(point, [0.30, 0.20, 0.0])


def test_slot_directions_are_orthonormal_where_the_geometry_says_so():
    d = db.slot_directions(np.array([0.15, 0.0, 0.06]), FINGER, ELBOW, SHOULDER)
    assert np.allclose(d["forearm"], [1.0, 0.0, 0.0])
    assert np.allclose(d["upperarm"], [0.0, 1.0, 0.0])
    # The tool sits above the forearm, so "outward" leaves the arm's centreline vertically.
    assert np.allclose(d["outward"], [0.0, 0.0, 1.0])
    assert all(abs(np.linalg.norm(v) - 1.0) < 1e-9 for v in d.values())


def test_a_tool_on_the_centreline_falls_back_to_a_finite_outward_direction():
    d = db.slot_directions(ELBOW, FINGER, ELBOW, SHOULDER)
    assert np.isfinite(d["outward"]).all()
    assert abs(np.linalg.norm(d["outward"]) - 1.0) < 1e-9


def test_macro_parts_split_the_window_in_order():
    macro = dict(name="retreat_outward", kind="direction", parts=(("upperarm", -1.0), ("outward", 1.0)))
    assert [db.macro_part(macro, t, 8) for t in range(8)] == [("upperarm", -1.0)] * 4 + [("outward", 1.0)] * 4
    with pytest.raises(ValueError):
        db.macro_part(macro, 8, 8)
    with pytest.raises(ValueError):
        db.macro_part(macro, 0, 1)


def test_direction_macros_command_the_same_magnitude_as_the_scaled_policy():
    d = db.slot_directions(np.array([0.15, 0.0, 0.06]), FINGER, ELBOW, SHOULDER)
    policy = np.array([0.9, -0.2, 0.1, 0.0, 0.3, 0.0])
    forward = db.macro_action(db.MACROS[2], 0, 8, d, policy, 0.008, 0.00866)
    scaled = db.macro_action(db.MACROS[1], 0, 8, d, policy, 0.008, 0.00866)
    assert np.isclose(np.linalg.norm(forward[:3]), np.linalg.norm(scaled[:3]))
    assert np.allclose(db.unit(forward[:3]), d["upperarm"])
    # A direction macro commands no rotation; the scaled control keeps the policy's.
    assert np.allclose(forward[3:], 0.0)
    assert np.allclose(scaled[3:], policy[3:])


def test_the_policy_macro_passes_the_action_through_unchanged():
    d = db.slot_directions(np.array([0.15, 0.0, 0.06]), FINGER, ELBOW, SHOULDER)
    policy = np.array([0.9, -0.2, 0.1, 0.0, 0.3, 0.0])
    assert np.array_equal(db.macro_action(db.MACROS[0], 0, 8, d, policy, 0.008, 0.00866), policy)


def test_a_macro_step_beyond_the_controller_limit_is_rejected():
    d = db.slot_directions(np.array([0.15, 0.0, 0.06]), FINGER, ELBOW, SHOULDER)
    with pytest.raises(ValueError):
        db.macro_action(db.MACROS[2], 0, 8, d, np.zeros(6), 0.02, 0.00866)


def trace(coverages, tracking=0.001):
    return [dict(upperarm_ratio=c, forearm_ratio=1.0, tracking_error=tracking, grasp_valid=True,
                 collision_rejected_substeps=0, tether_rejected_substeps=0,
                 commanded_translation_m=0.008, accepted_anchor_translation_m=0.007) for c in coverages]


def test_branch_summary_reports_the_sustained_tail_not_the_peak():
    s = db.branch_summary(trace([0.9] * 4 + [0.5] * 12))
    assert s["max_coverage"] == pytest.approx(0.9)
    assert s["sustained_coverage"] == pytest.approx(0.5)
    assert s["final_coverage"] == pytest.approx(0.5)
    assert s["whole_branch_grasp_valid"]
    assert db.branch_summary(trace([0.8] * 12, tracking=0.03))["whole_branch_grasp_valid"] is False
    with pytest.raises(ValueError):
        db.branch_summary(trace([0.5] * 11))


def test_consequence_compares_the_best_macro_with_the_policy_and_with_the_repeat_spread():
    returns = {"policy": np.array([0.20, 0.22]), "retreat": np.array([0.60, 0.58]), "lift": np.array([0.10, 0.30])}
    c = db.consequence(returns)
    assert c["best_macro"] == "retreat"
    assert c["consequence"] == pytest.approx(0.59 - 0.21)
    assert c["spread"] == pytest.approx((0.02 + 0.02 + 0.20) / 3)
    assert c["decisive"]
    # A gain inside the repeat spread is not evidence.
    noisy = {"policy": np.array([0.20, 0.60]), "retreat": np.array([0.30, 0.70])}
    assert not db.consequence(noisy)["decisive"]
    with pytest.raises(ValueError):
        db.consequence({"retreat": np.array([0.3, 0.4])})


def test_ranking_agreement_separates_a_stable_order_from_a_coin_flip():
    stable = {"a": np.array([0.9, 0.85, 0.88]), "b": np.array([0.5, 0.45, 0.52])}
    r = db.ranking_agreement(stable)
    assert r["top1_agreement"] == pytest.approx(1.0)
    assert r["pairwise_agreement"] == pytest.approx(1.0)
    flipped = {"a": np.array([0.9, 0.2]), "b": np.array([0.3, 0.8])}
    f = db.ranking_agreement(flipped)
    assert f["top1_agreement"] == pytest.approx(0.5)
    assert f["pairwise_agreement"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        db.ranking_agreement({"a": np.array([0.1])})
    with pytest.raises(ValueError):
        db.ranking_agreement({"a": np.array([0.1, 0.2]), "b": np.array([0.3])})


def test_returns_by_macro_orders_repeats_and_state_ids_are_unique_per_slot():
    records = [dict(macro="retreat", repeat=1, sustained_coverage=0.4),
               dict(macro="retreat", repeat=0, sustained_coverage=0.6),
               dict(macro="policy", repeat=0, sustained_coverage=0.1)]
    grouped = db.returns_by_macro(records)
    assert np.allclose(grouped["retreat"], [0.6, 0.4])
    assert np.allclose(grouped["policy"], [0.1])
    assert state_id(("tshirt_26", 14046), 3301, 140) != state_id(("tshirt_26", 14046), 3302, 140)
