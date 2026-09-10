"""CPU tests of the multi-cell dressing protocol: slot plan, held-out split, summary, score."""

import math

import numpy as np
import pytest

from uipc_manip.cellplan import (
    cell_label,
    checkpoint_score,
    coverage_problems,
    plan_slots,
    select_heldout_bodies,
    summarize,
    training_rows,
)

GARMENTS = ["hospital_gown", "tshirt_26", "tshirt_4", "tshirt_68"]
CELLS = [(g, b) for b in range(8) for g in GARMENTS]  # 4 garments x 8 bodies = 32 cells


def _records(slots, succeeds):
    rows = []
    for slot in range(len(slots)):
        ok = bool(succeeds(slot))
        rows.append({
            "slot": slot, "success": ok, "paper_filter": ok, "return": 50.0 if ok else 5.0,
            "final_upperarm_ratio": 0.8 if ok else 0.05, "final_forearm_ratio": 0.9 if ok else 0.4,
        })
    return rows


def test_heldout_bodies_are_the_greatest_complete_rows():
    assert select_heldout_bodies(CELLS, GARMENTS, 1) == [7]
    assert select_heldout_bodies(CELLS, GARMENTS, 2) == [6, 7]
    assert select_heldout_bodies(CELLS, GARMENTS, 0) == []
    sparse = [c for c in CELLS if c != ("tshirt_68", 7)]
    assert select_heldout_bodies(sparse, GARMENTS, 1) == [6]  # a body missing a garment is skipped
    with pytest.raises(ValueError, match="carry every garment"):
        select_heldout_bodies(sparse, GARMENTS, 8)


def test_plan_pins_heldout_cells_first_and_stratifies_training_by_garment():
    slots, heldout = plan_slots(CELLS, 32, [7])
    assert len(slots) == 32 and len(set(slots)) == 32
    assert heldout == [0, 1, 2, 3]
    assert all(b == 7 for _, b in slots[:4]) and all(b != 7 for _, b in slots[4:])
    assert {g: sum(1 for x, _ in slots[4:] if x == g) for g in GARMENTS} == dict.fromkeys(GARMENTS, 7)

    slots, heldout = plan_slots(CELLS, 32, [6, 7])
    assert len(heldout) == 8 and len(set(slots)) == 32

    # Fewer slots than cells: every garment keeps an equal share, no duplicate while the pool lasts.
    slots, heldout = plan_slots(CELLS, 16, [7])
    assert len(heldout) == 4 and len(set(slots)) == 16
    assert {g: sum(1 for x, _ in slots[4:] if x == g) for g in GARMENTS} == dict.fromkeys(GARMENTS, 3)

    # A single-body regional teacher cycles its garments over every slot.
    slots, heldout = plan_slots([("tshirt_392", 0), ("tshirt_26", 0)], 5, [])
    assert heldout == [] and slots == [("tshirt_26", 0), ("tshirt_392", 0)] * 2 + [("tshirt_26", 0)]

    with pytest.raises(ValueError, match="no training cells"):
        plan_slots(CELLS[:4], 4, [0])
    with pytest.raises(ValueError, match="held-out grid needs 8 slots"):
        plan_slots(CELLS, 4, [6, 7])


def test_coverage_problems_name_each_gap():
    assert coverage_problems(CELLS, GARMENTS, range(8), 32, 32) == []
    assert coverage_problems(CELLS, GARMENTS, range(8), 32, None) == []
    # Duplicate slots are allowed: the single-body path cycles two cells over sixteen slots.
    assert coverage_problems([("tshirt_26", 0), ("tshirt_392", 0)], ["tshirt_26", "tshirt_392"], [0], 16, 16) == []
    problems = coverage_problems(CELLS, [*GARMENTS, "tshirt_392"], [*range(8), 9], 16, 8)
    assert len(problems) == 4
    assert "tshirt_392" in problems[0] and "[9]" in problems[1]
    assert "32 cells do not fit in 16 slots" in problems[2] and "8 evaluation episodes" in problems[3]


def test_summary_reports_heldout_training_and_per_cell_keys():
    slots, heldout = plan_slots(CELLS, 32, [7])
    records = _records(slots, lambda s: s >= 4 and s % 3 != 0)  # held-out slots 0-3 all fail
    s = summarize(records, slots, heldout)
    assert s["episodes"] == 32 and s["cell_count"] == 32
    assert s["heldout_episode_count"] == 4 and s["heldout_cell_count"] == 4
    assert s["heldout_success_rate"] == 0.0 and s["heldout_zero_cell_count"] == 4
    assert s["training_episode_count"] == 28 and s["training_cell_count"] == 28
    for prefix in ("", "heldout_", "training_"):
        for key in ("success_rate", "worst_cell_success_rate", "zero_cell_count", "paper_filter_rate", "mean_return",
                    "mean_final_upperarm_ratio", "mean_final_forearm_ratio"):
            assert prefix + key in s, prefix + key
    assert s["success_rate_tshirt_26|human=7"] == 0.0
    assert s["mean_final_upperarm_ratio_tshirt_26|human=7"] == pytest.approx(0.05)
    assert s["episodes_tshirt_26"] == 8 and "success_rate_tshirt_26" in s and "mean_final_upperarm_ratio_tshirt_26" in s
    # Without a held-out split there is no heldout_ block and every slot is a training slot.
    s = summarize(records, slots, [])
    assert "heldout_episode_count" not in s and s["training_episode_count"] == 32


def test_cell_label_without_body_is_the_garment():
    assert cell_label("tshirt_26", 3) == "tshirt_26|human=3"
    assert cell_label("tshirt_26", None) == "tshirt_26"
    s = summarize([{"slot": 0, "success": True, "paper_filter": True, "return": 1.0}], [("garment_0", None)], [], metric_keys=())
    assert s["success_rate_garment_0"] == 1.0 and s["episodes_garment_0"] == 1
    assert "mean_final_upperarm_ratio_garment_0" not in s  # the records carry no ratio


def test_score_ranks_heldout_generalization_ahead_of_memorisation():
    slots, heldout = plan_slots(CELLS, 32, [7])
    memoriser = summarize(_records(slots, lambda s: s >= 4), slots, heldout)  # every training cell, no unseen one
    generaliser = summarize(_records(slots, lambda s: s < 4 or s % 2 == 0), slots, heldout)  # unseen cells, half the rest
    assert memoriser["training_success_rate"] > generaliser["training_success_rate"]
    assert checkpoint_score(generaliser) > checkpoint_score(memoriser)
    assert len(checkpoint_score(generaliser)) == 7


def test_score_leads_with_the_continuous_ratio_then_success_then_worst_cell():
    base = {"heldout_episode_count": 8, "heldout_success_rate": 0.0, "heldout_worst_cell_success_rate": 0.0}
    near_miss = {**base, "heldout_mean_final_upperarm_ratio": 0.69}
    far = {**base, "heldout_mean_final_upperarm_ratio": 0.30}
    assert checkpoint_score(near_miss) > checkpoint_score(far)  # the ratio moved; a 0.70 threshold sees nothing
    tied = {**base, "heldout_mean_final_upperarm_ratio": 0.5}
    one_cell = {**tied, "heldout_success_rate": 0.25}
    assert checkpoint_score(one_cell) > checkpoint_score(tied)
    assert checkpoint_score({**one_cell, "heldout_worst_cell_success_rate": 0.25}) > checkpoint_score(one_cell)
    # All-cell keys only break ties; training-cell success cannot outrank the held-out ratio.
    assert checkpoint_score({**far, "success_rate": 1.0, "mean_final_upperarm_ratio": 0.9}) < checkpoint_score(near_miss)


def test_score_without_heldout_uses_all_cell_keys_and_floors_nan():
    single_body = {"success_rate": 0.5, "mean_final_upperarm_ratio": 0.6, "mean_max_upperarm_ratio": 0.8, "mean_return": 3.0}
    assert checkpoint_score(single_body)[:2] == (0.6, 0.5)
    assert checkpoint_score({"success_rate": 0, "mean_final_distance": 1, "mean_return": 0})[1] == 0.0
    broken = checkpoint_score({"mean_final_upperarm_ratio": math.nan, "success_rate": 0.0, "mean_return": math.nan})
    assert not any(math.isnan(v) for v in broken)
    assert checkpoint_score(single_body) > broken  # one NaN round cannot freeze the best checkpoint


def test_training_rows_exclude_heldout_slots_and_ungated_garments():
    slots, heldout = plan_slots(CELLS, 32, [7])
    mask = np.ones(32, dtype=bool)
    mask[heldout] = False
    rank = np.array([GARMENTS.index(g) for g, _ in slots])
    rows = training_rows(mask, rank, 2, np.zeros(32, dtype=bool))
    assert not rows[:4].any()
    assert {g for (g, _), keep in zip(slots, rows) if keep} == {"hospital_gown", "tshirt_26"}
    assert int(rows.sum()) == 14
    errors = np.zeros(32, dtype=bool)
    errors[4] = True
    assert not training_rows(mask, rank, 4, errors)[4]
