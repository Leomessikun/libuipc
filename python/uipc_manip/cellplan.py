"""Multi-cell dressing protocol: slot planning, held-out split, per-cell summary, score.

One policy for many garments and many bodies needs four things the single-cell
loop does not: a cell bound to every slot, whole bodies reserved from training,
an evaluation that reports per cell rather than pooled, and a checkpoint score
that ranks generalisation ahead of training-cell success. This module is the
simulator-free half of that, so it can be tested on the CPU.

Ported from the Newton dressing trainer's `_assign_static_config_ids`,
`select_complete_heldout_human` and `eval_metrics.policy_checkpoint_score`; see
`agent_docs/performance/2026-09-10-one-policy-protocol.md` for the mapping.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- slot planning
def select_heldout_bodies(cells, garments, count: int) -> list[int]:
    """Deterministically reserve whole bodies that carry every requested garment.

    Port of ``training.config.select_complete_heldout_human`` generalised to N
    bodies: the reference reserves a complete pose row so a held-out score is
    never confounded with garment composition, and takes the greatest such id so
    the choice is reproducible.
    """
    requested = {str(g) for g in garments}
    by_body: dict[int, set[str]] = {}
    for garment, body in cells:
        by_body.setdefault(int(body), set()).add(str(garment))
    complete = sorted(b for b, gs in by_body.items() if requested <= gs)
    if len(complete) < int(count):
        raise ValueError(
            f"{count} held-out bodies requested but only {len(complete)} carry every garment {sorted(requested)}"
        )
    return complete[len(complete) - int(count):]


def plan_slots(cells, num_envs: int, heldout_bodies) -> tuple[list[tuple[str, int]], list[int]]:
    """Bind one cell to each slot: held-out cells first, then garment-stratified.

    Held-out cells take the leading slots so every evaluation round scores the
    same grid (Newton ``_assign_static_config_ids`` ``pinned_eval_human_ids``).
    Training cells are stratified by garment before being spread over bodies, so
    an uneven cell library cannot starve one garment
    (Newton ``_assign_static_config_ids`` ``by_garment``).
    """
    heldout = {int(b) for b in heldout_bodies}
    pinned = sorted((g, b) for g, b in cells if int(b) in heldout)
    pool = sorted((g, b) for g, b in cells if int(b) not in heldout)
    if len(pinned) > int(num_envs):
        raise ValueError(f"held-out grid needs {len(pinned)} slots, only {num_envs} exist")
    if not pool:
        raise ValueError("the held-out split leaves no training cells")
    count = int(num_envs) - len(pinned)
    by_garment: dict[str, list[tuple[str, int]]] = {}
    for cell in pool:
        by_garment.setdefault(cell[0], []).append(cell)
    garment_names = sorted(by_garment)
    training: list[tuple[str, int]] = []
    for slot in range(count):
        bucket = by_garment[garment_names[slot % len(garment_names)]]
        training.append(bucket[(slot // len(garment_names)) % len(bucket)])
    slots = pinned + training
    heldout_slots = list(range(len(pinned)))
    return slots, heldout_slots


# ------------------------------------------------------------- evaluation summary
def cell_label(garment: str, body: int) -> str:
    return f"{garment}|human={int(body)}"


def summarize(records, slot_cells, heldout_slots, metric_keys=("upperarm_ratio", "forearm_ratio")):
    """Per-cell evaluation summary with held-out and training subsets.

    ``records`` are the port's existing evaluation records with an added
    ``slot`` field. Cell rates, not episode rates, are what
    ``worst_cell_success_rate`` ranks, so one dead cell cannot hide behind
    fifteen good ones.
    """
    heldout = {int(s) for s in heldout_slots}
    labels = [cell_label(g, b) for g, b in slot_cells]
    out: dict[str, float] = {"episodes": float(len(records))}

    def block(prefix: str, rows):
        if not rows:
            return
        by_cell: dict[str, list[dict]] = {}
        for r in rows:
            by_cell.setdefault(labels[int(r["slot"])], []).append(r)
        cell_rates = [float(np.mean([x["success"] for x in v])) for v in by_cell.values()]
        p = f"{prefix}_" if prefix else ""
        out[f"{p}episode_count"] = float(len(rows))
        out[f"{p}cell_count"] = float(len(by_cell))
        out[f"{p}success_rate"] = float(np.mean([r["success"] for r in rows]))
        out[f"{p}worst_cell_success_rate"] = float(min(cell_rates))
        out[f"{p}zero_cell_count"] = float(sum(1 for r in cell_rates if r == 0.0))
        out[f"{p}paper_filter_rate"] = float(np.mean([r["paper_filter"] for r in rows]))
        out[f"{p}mean_return"] = float(np.mean([r["return"] for r in rows]))
        for k in metric_keys:
            out[f"{p}mean_final_{k}"] = float(np.nanmean([r[f"final_{k}"] for r in rows]))

    block("", records)
    block("heldout", [r for r in records if int(r["slot"]) in heldout])
    block("training", [r for r in records if int(r["slot"]) not in heldout])
    for label in sorted(set(labels)):
        rows = [r for r in records if labels[int(r["slot"])] == label]
        if rows:
            out[f"success_rate_{label}"] = float(np.mean([r["success"] for r in rows]))
            out[f"mean_final_upperarm_ratio_{label}"] = float(np.nanmean([r["final_upperarm_ratio"] for r in rows]))
    for garment in sorted({g for g, _ in slot_cells}):
        rows = [r for r in records if slot_cells[int(r["slot"])][0] == garment]
        if rows:
            out[f"success_rate_{garment}"] = float(np.mean([r["success"] for r in rows]))
    return out


# --------------------------------------------------------------- checkpoint score
def checkpoint_score(metrics: dict) -> tuple:
    """Rank held-out generalization first, exactly as ``policy_checkpoint_score``.

    ``success_rate`` here is already FMVP's own criterion: the upper-arm ratio
    the trajectory ENDS with, at least 0.7 (``dressing_env`` reports ``success``
    from the final step's ratio at the shared time limit).
    """
    p = "heldout_" if float(metrics.get("heldout_episode_count", 0.0)) > 0.0 else ""
    return (
        float(metrics.get(f"{p}success_rate", 0.0)),
        float(metrics.get(f"{p}worst_cell_success_rate", 0.0)),
        # No paper_filter here: the reference uses the early-turn test only in
        # the Stage I-B rollout filter, never in policy_checkpoint_score.
        float(metrics.get(f"{p}mean_final_upperarm_ratio", 0.0)),
        float(metrics.get(f"{p}mean_final_forearm_ratio", 0.0)),
        float(metrics.get(f"{p}mean_return", float("-inf"))),
        # All-cell metrics break ties without letting training-cell success
        # outrank generalization to an unseen body.
        float(metrics.get("success_rate", 0.0)),
        float(metrics.get("mean_final_upperarm_ratio", 0.0)),
    )


# --------------------------------------------------------------------- curriculum
def training_rows(slot_mask, slot_rank, stage, sim_error):
    """Which slots may write to replay: training cell, garment admitted, no error."""
    return np.asarray(slot_mask) & (np.asarray(slot_rank) < int(stage)) & ~np.asarray(sim_error)
