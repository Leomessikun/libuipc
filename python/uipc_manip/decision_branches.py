"""Counterfactual branches at the decision states a dressing policy actually visits.

At a state the policy reaches, the world is snapshotted and several short recovery
macros are executed in its place, each followed by the *same* base policy, so the
only difference between branches is the intervention itself. Every branch is
repeated with identical commands, because repeated continuations from one restored
state do not reproduce (see the predictability record), so a single outcome cannot
rank two macros.

This collects labelled decisions; it trains nothing and changes no solver setting.
The three quantities it is built to measure are, per state ``s``:

* consequence ``C(s)``  = best macro return minus the policy's own return,
* reliability ``R(s)``  = agreement of the macro ranking across independent repeats,
* and the inputs needed later to ask whether the better macro is identifiable from
  deployment-time observations alone (observation, history window, privileged state).
"""
from __future__ import annotations

import numpy as np

ARM_DIRECTIONS = ("forearm", "upperarm", "outward", "up")
"""Unit directions available to a macro, in the world frame of one slot."""

MACROS = (
    dict(name="policy", kind="policy"),
    dict(name="policy_scaled", kind="scaled"),
    dict(name="forward", kind="direction", parts=(("upperarm", 1.0),)),
    dict(name="retreat", kind="direction", parts=(("upperarm", -1.0),)),
    dict(name="outward", kind="direction", parts=(("outward", 1.0),)),
    dict(name="lift", kind="direction", parts=(("up", 1.0),)),
    dict(name="retreat_outward", kind="direction", parts=(("upperarm", -1.0), ("outward", 1.0))),
)
"""The macro library. ``policy`` is the no-intervention reference and ``policy_scaled``
is its direction at the macros' command magnitude, so a macro that wins only by moving
further is visible as such. A ``direction`` macro splits its window equally between its
parts, in order; a negative weight reverses the direction."""


def unit(v: np.ndarray, fallback=(0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.asarray(fallback, dtype=np.float64)


def centerline_projection(point: np.ndarray, finger: np.ndarray, elbow: np.ndarray,
                          shoulder: np.ndarray) -> tuple[np.ndarray, str]:
    """Closest point on the finger-elbow-shoulder polyline, and which segment carries it."""
    point = np.asarray(point, dtype=np.float64)
    best = None
    for name, (a, b) in (("forearm", (finger, elbow)), ("upperarm", (elbow, shoulder))):
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        d = b - a
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        lam = float(np.clip((point - a) @ (d / n), 0.0, n))
        q = a + lam * d / n
        gap = float(np.linalg.norm(point - q))
        if best is None or gap < best[0]:
            best = (gap, q, name)
    if best is None:
        raise ValueError("Degenerate arm landmarks")
    return best[1], best[2]


def slot_directions(tool: np.ndarray, finger: np.ndarray, elbow: np.ndarray,
                    shoulder: np.ndarray) -> dict[str, np.ndarray]:
    """World-frame unit directions at ``tool``.

    ``upperarm`` points along the arm toward the shoulder, ``forearm`` from the
    fingertip toward the elbow, ``outward`` away from the arm's centreline through the
    tool (the direction that reduces the sleeve's wrap), and ``up`` is the world's.
    """
    foot, _ = centerline_projection(tool, finger, elbow, shoulder)
    outward = np.asarray(tool, dtype=np.float64) - foot
    return {
        "forearm": unit(np.asarray(elbow, dtype=np.float64) - np.asarray(finger, dtype=np.float64), (1.0, 0.0, 0.0)),
        "upperarm": unit(np.asarray(shoulder, dtype=np.float64) - np.asarray(elbow, dtype=np.float64), (1.0, 0.0, 0.0)),
        "outward": unit(outward, (0.0, 0.0, 1.0)),
        "up": np.asarray((0.0, 0.0, 1.0)),
    }


def macro_part(macro: dict, t: int, window: int) -> tuple[str, float]:
    """The (direction, sign) this ``direction`` macro uses at step ``t`` of its window."""
    parts = macro["parts"]
    if window < len(parts):
        raise ValueError("Macro window is shorter than the macro's part count")
    edges = [round((i + 1) * window / len(parts)) for i in range(len(parts))]
    for edge, (name, weight) in zip(edges, parts, strict=True):
        if t < edge:
            if name not in ARM_DIRECTIONS:
                raise ValueError(f"Unknown macro direction {name}")
            return name, float(weight)
    raise ValueError("Step outside the macro window")


def macro_action(macro: dict, t: int, window: int, directions: dict[str, np.ndarray],
                 policy_action: np.ndarray, step_m: float, max_translation: float) -> np.ndarray:
    """One slot's normalized command for macro ``macro`` at step ``t`` of its window."""
    scale = float(step_m) / float(max_translation)
    if scale > 1.0 + 1e-9:
        raise ValueError("Requested macro step exceeds the controller's per-decision limit")
    action = np.asarray(policy_action, dtype=np.float64).copy()
    if macro["kind"] == "policy":
        return action
    if macro["kind"] == "scaled":
        action[:3] = unit(action[:3], (1.0, 0.0, 0.0)) * scale
        return action
    name, weight = macro_part(macro, t, window)
    out = np.zeros_like(action)
    out[:3] = weight * directions[name] * scale
    return out


def branch_summary(trace: list[dict], sustained_decisions: int = 12) -> dict:
    """Outcome of one branch: the metrics a return may be built from, no weighting yet."""
    if len(trace) < sustained_decisions:
        raise ValueError("Branch is shorter than the sustained window")
    upper = np.asarray([row["upperarm_ratio"] for row in trace], dtype=np.float64)
    tracking = np.asarray([row["tracking_error"] for row in trace], dtype=np.float64)
    tail = upper[-sustained_decisions:]
    return dict(sustained_coverage=float(tail.min()), mean_tail_coverage=float(tail.mean()),
                final_coverage=float(upper[-1]), max_coverage=float(upper.max()),
                max_tracking_error=float(tracking.max()),
                whole_branch_grasp_valid=bool(tracking.max() <= 0.02),
                collision_rejections=int(sum(row["collision_rejected_substeps"] for row in trace)),
                tether_rejections=int(sum(row["tether_rejected_substeps"] for row in trace)),
                commanded_translation_m=float(sum(row["commanded_translation_m"] for row in trace)),
                accepted_translation_m=float(sum(row["accepted_anchor_translation_m"] for row in trace)))


def returns_by_macro(records: list[dict], key: str = "sustained_coverage") -> dict[str, np.ndarray]:
    """Group one state's branch returns by macro name, in repeat order."""
    out: dict[str, list[float]] = {}
    for r in sorted(records, key=lambda r: (r["macro"], r["repeat"])):
        out.setdefault(r["macro"], []).append(float(r[key]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def consequence(returns: dict[str, np.ndarray], reference: str = "policy") -> dict:
    """How much better the best macro is than the policy's own continuation.

    ``spread`` is the mean within-macro range over repeats: the scale below which a
    difference between two macros is not evidence.
    """
    if reference not in returns:
        raise ValueError(f"Missing the {reference} reference branch")
    means = {k: float(v.mean()) for k, v in returns.items()}
    spread = float(np.mean([float(np.ptp(v)) for v in returns.values()]))
    best = max((k for k in means if k != reference), key=lambda k: means[k])
    return dict(best_macro=best, best_return=means[best], reference_return=means[reference],
                consequence=means[best] - means[reference], spread=spread,
                decisive=bool(means[best] - means[reference] > spread), means=means)


def ranking_agreement(returns: dict[str, np.ndarray]) -> dict:
    """Do independent repeats rank the macros the same way?

    ``top1`` is the fraction of repeats whose best macro is the best of the repeat
    means; ``pairwise`` is the fraction of macro pairs ordered identically in every
    repeat. Both are 1.0 when a single repeat already settles the ranking.
    """
    names = sorted(returns)
    counts = {len(v) for v in returns.values()}
    if len(counts) != 1:
        raise ValueError("Every macro needs the same number of repeats")
    repeats = counts.pop()
    if repeats < 2:
        raise ValueError("Ranking agreement needs at least two repeats")
    means = {k: float(returns[k].mean()) for k in names}
    overall_best = max(names, key=lambda k: means[k])
    per_repeat_best = [max(names, key=lambda k: returns[k][i]) for i in range(repeats)]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    agree = 0
    for a, b in pairs:
        signs = {int(np.sign(returns[a][i] - returns[b][i])) for i in range(repeats)}
        agree += int(len(signs) == 1 and 0 not in signs)
    return dict(repeats=repeats, top1_agreement=float(np.mean([b == overall_best for b in per_repeat_best])),
                pairwise_agreement=float(agree / len(pairs)) if pairs else 1.0,
                per_repeat_best=per_repeat_best, overall_best=overall_best)
