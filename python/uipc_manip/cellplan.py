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
def complete_bodies(cells, garments) -> list[int]:
    """Bodies that carry a cell of every requested garment, ascending."""
    requested = {str(g) for g in garments}
    by_body: dict[int, set[str]] = {}
    for garment, body in cells:
        by_body.setdefault(int(body), set()).add(str(garment))
    return sorted(b for b, gs in by_body.items() if requested <= gs)


def select_heldout_bodies(cells, garments, count: int) -> list[int]:
    """Deterministically reserve whole bodies that carry every requested garment.

    Port of ``training.config.select_complete_heldout_human`` generalised to N
    bodies: the reference reserves a complete pose row so a held-out score is
    never confounded with garment composition, and takes the greatest such id so
    the choice is reproducible.
    """
    complete = complete_bodies(cells, garments)
    if len(complete) < int(count):
        raise ValueError(
            f"{count} held-out bodies requested but only {len(complete)} carry every garment {sorted({str(g) for g in garments})}"
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


def coverage_problems(library, garments, bodies, num_envs: int, eval_episodes: int | None = None) -> list[str]:
    """Why ``num_envs`` slots would not train and score every requested cell; empty when they do.

    The Newton launcher refuses these for a formal run: a library cell without a
    slot is never trained or scored, a requested garment or body without a cell
    silently narrows the distribution, and a round of fewer episodes than cells
    cannot score each cell once. Duplicate slots are allowed, so a single-body
    regional teacher can cycle its few garments over every slot.
    """
    cells = {(str(g), int(b)) for g, b in library}
    problems = []
    missing_garments = [str(g) for g in dict.fromkeys(garments) if all(c[0] != str(g) for c in cells)]
    if missing_garments:
        problems.append(f"no cell for garments {missing_garments}")
    missing_bodies = [int(b) for b in dict.fromkeys(bodies) if all(c[1] != int(b) for c in cells)]
    if missing_bodies:
        problems.append(f"no cell for bodies {missing_bodies}")
    if len(cells) > int(num_envs):
        problems.append(
            f"{len(cells)} cells do not fit in {num_envs} slots, so {len(cells) - int(num_envs)} would never be trained or scored"
        )
    if eval_episodes is not None and int(eval_episodes) < len(cells):
        problems.append(f"{eval_episodes} evaluation episodes cannot score {len(cells)} cells once each")
    return problems


# ------------------------------------------------------------- evaluation summary
def cell_label(garment: str, body: int | None) -> str:
    """``garment|human=N``; just the garment when the body is unknown."""
    return str(garment) if body is None else f"{garment}|human={int(body)}"


def summarize(records, slot_cells, heldout_slots, metric_keys=("upperarm_ratio", "forearm_ratio")):
    """Per-cell evaluation summary with held-out and training subsets.

    ``records`` are the port's existing evaluation records with an added
    ``slot`` field. Cell rates, not episode rates, are what
    ``worst_cell_success_rate`` ranks, so one dead cell cannot hide behind
    fifteen good ones.
    """
    heldout = {int(s) for s in heldout_slots}
    labels = [cell_label(g, b) for g, b in slot_cells]
    out: dict[str, float | int] = {"episodes": len(records)}

    def block(prefix: str, rows):
        if not rows:
            return
        by_cell: dict[str, list[dict]] = {}
        for r in rows:
            by_cell.setdefault(labels[int(r["slot"])], []).append(r)
        cell_rates = [float(np.mean([x["success"] for x in v])) for v in by_cell.values()]
        p = f"{prefix}_" if prefix else ""
        out[f"{p}episode_count"] = len(rows)
        out[f"{p}cell_count"] = len(by_cell)
        out[f"{p}success_rate"] = float(np.mean([r["success"] for r in rows]))
        out[f"{p}worst_cell_success_rate"] = float(min(cell_rates))
        out[f"{p}zero_cell_count"] = sum(1 for r in cell_rates if r == 0.0)
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
            if "final_upperarm_ratio" in rows[0]:
                out[f"mean_final_upperarm_ratio_{label}"] = float(np.nanmean([r["final_upperarm_ratio"] for r in rows]))
    for garment in sorted({str(g) for g, _ in slot_cells}):
        rows = [r for r in records if str(slot_cells[int(r["slot"])][0]) == garment]
        if rows:
            out[f"episodes_{garment}"] = len(rows)
            out[f"success_rate_{garment}"] = float(np.mean([r["success"] for r in rows]))
            for k in metric_keys:
                out[f"mean_final_{k}_{garment}"] = float(np.nanmean([r[f"final_{k}"] for r in rows]))
    return out


# --------------------------------------------------------------- checkpoint score
def checkpoint_score(metrics: dict) -> tuple:
    """Lexicographic best-checkpoint key: held-out first, each block led by the dressed ratio.

    When a held-out subset was scored its keys lead, as in the reference's
    ``policy_checkpoint_score``, so a policy that memorises its training cells
    cannot outrank one that dresses an unseen body; the all-cell keys only break
    ties. Inside each block the continuous final upper-arm ratio leads. ``success``
    is FMVP's own criterion, that ratio at least 0.7 where the trajectory ends, but
    with a handful of episodes per cell the thresholded rate is coarse and noisy: a
    run whose best evaluation scored 0.50 had one garment finish at 0.7528 against
    the 0.70 threshold, and the next evaluation's 0.00 was the same garment at
    0.4541. The threshold rate follows, then the worst cell's rate, which exposes
    the port's actual failure mode: one garment at exactly 0.00 behind a 0.50 mean.
    ``paper_filter_rate`` stays out: the reference applies the early-turn test only
    in the Stage I-B rollout filter, never in checkpoint selection. A missing or
    non-finite value scores as the floor, so one NaN round cannot freeze ``best.pt``.

    A round that hit a simulator error ranks below every clean round, whatever its
    ratios. Such a round is scored at each episode's last-seen reading, and in this
    port a decision watchdog trip raises for the whole world, so those readings are
    mid-pull: a sleeve on the upper arm when the round was cut may still slip back by
    the horizon. Relaunched teacher r13 showed why the veto is needed rather than a
    tie-break: its voided round at 205,000 transitions averaged 0.302 that way, above
    the 0.283 of the best honest round, and took ``best.pt`` with `success_rate` 0 and
    every cell scored zero.
    """

    def value(key: str, floor: float = 0.0) -> float:
        try:
            v = float(metrics.get(key, floor))
        except (TypeError, ValueError):
            return floor
        return v if np.isfinite(v) else floor

    p = "heldout_" if value("heldout_episode_count") > 0.0 else ""
    return (
        0.0 if value("sim_errors") > 0.0 else 1.0,
        value(f"{p}mean_final_upperarm_ratio"),
        value(f"{p}success_rate"),
        value(f"{p}worst_cell_success_rate"),
        # The all-cell key the single-body trainer ranked by, as tie-breakers.
        value("mean_final_upperarm_ratio"),
        value("success_rate"),
        value("mean_max_upperarm_ratio"),
        value("mean_return", float("-inf")),
    )


# --------------------------------------------------------------------- curriculum
def training_rows(slot_mask, slot_rank, stage, sim_error):
    """Which slots may write to replay: training cell, garment admitted, no error."""
    return np.asarray(slot_mask) & (np.asarray(slot_rank) < int(stage)) & ~np.asarray(sim_error)
