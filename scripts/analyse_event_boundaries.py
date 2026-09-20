"""Describe task-event variability in existing dressing branches; no simulation.

Macro dependence, mixed-repeat fractions and event-conditioned coverage gaps are
screening statistics. They do not identify an action-space discontinuity or its
one-sided limits. Historical JSON field names remain for compatibility; each
collection includes interpretation metadata. Cost estimates assume independent
Bernoulli sample means and are precision heuristics, not estimator lower bounds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from collections import defaultdict


def load_collections(root: str) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "decision_branches*", "**", "result.json"), recursive=True)):
        with open(path) as handle:
            data = json.load(handle)
        if not data.get("records"):
            continue
        if data.get("repeats", 0) < 3:
            continue
        out.append((path, data))
    return out


def branch_events(record: dict) -> dict[str, bool]:
    """Binary task events read from one branch."""
    trace = record.get("trace") or []
    final = trace[-1] if trace else {}
    max_forearm = max((step.get("forearm_ratio", 0.0) for step in trace), default=0.0)
    return {
        # grasp retained over the whole branch (the criterion used by the audits)
        "grasp_kept": bool(record.get("whole_branch_grasp_valid")),
        # the opening is on the arm at the end of the branch
        "on_arm_end": float(final.get("forearm_ratio", 0.0) or 0.0) > 0.0,
        # the opening was on the arm at any point in the branch
        "on_arm_any": max_forearm > 0.0,
        # the branch retains coverage over its tail
        "sustained": float(record.get("sustained_coverage", 0.0) or 0.0) > 0.0,
    }


def boundary_jump(groups: list[list[tuple[int, float]]]) -> dict:
    """Event-conditioned coverage gap within repeated state/macro cells.

    Historical function/output names are retained for artifact compatibility.
    Different outcomes under repeated interventions do not identify an action-space
    boundary or its one-sided limits. Conditioning on the outcome also does not
    identify its causal effect. These are descriptive gaps, not boundary jumps.
    """
    jumps = []
    for cell in groups:
        true_values = [v for flag, v in cell if flag]
        false_values = [v for flag, v in cell if not flag]
        if not true_values or not false_values:
            continue
        jumps.append(sum(true_values) / len(true_values) - sum(false_values) / len(false_values))
    if not jumps:
        return {"cells": 0, "mean_jump": float("nan"), "mean_abs_jump": float("nan")}
    mean = sum(jumps) / len(jumps)
    variance = sum((j - mean) ** 2 for j in jumps) / max(len(jumps) - 1, 1)
    return {
        "cells": len(jumps),
        "mean_jump": mean,
        "mean_abs_jump": sum(abs(j) for j in jumps) / len(jumps),
        "sd_jump": math.sqrt(variance),
        "se_jump": math.sqrt(variance / len(jumps)),
        "max_abs_jump": max(abs(j) for j in jumps),
    }


def permutation_eta(cells: dict[str, list[int]], trials: int, rng: random.Random) -> tuple[float, float]:
    """Fraction of event variance explained by macro identity, plus permutation p.

    `cells` maps macro -> list of 0/1 outcomes of identical repeats at one state.
    """
    labels = []
    values = []
    for macro, outcomes in cells.items():
        for value in outcomes:
            labels.append(macro)
            values.append(value)
    n = len(values)
    if n < 4:
        return float("nan"), float("nan")
    grand = sum(values) / n
    total = sum((v - grand) ** 2 for v in values)
    if total <= 0.0:
        # the event is decided at this state: no variance to explain
        return float("nan"), float("nan")

    def between(assignment: list[str]) -> float:
        groups: dict[str, list[int]] = defaultdict(list)
        for label, value in zip(assignment, values):
            groups[label].append(value)
        return sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values())

    observed = between(labels) / total
    shuffled = labels[:]
    hits = 0
    for _ in range(trials):
        rng.shuffle(shuffled)
        if between(shuffled) / total >= observed - 1e-12:
            hits += 1
    return observed, (hits + 1) / (trials + 1)


def repeats_for_difference(p_pooled: float, difference: float, sigmas: float = 3.0) -> float:
    """Repeats per candidate so that `difference` clears `sigmas` standard errors."""
    variance = max(p_pooled * (1.0 - p_pooled), 1e-6)
    return 2.0 * variance * (sigmas / difference) ** 2


def analyse(path: str, data: dict, trials: int, rng: random.Random) -> dict:
    records = data["records"]
    window = int(data.get("window", 0))
    follow = int(data.get("follow", 0))
    per_branch_decisions = window + follow
    seconds = float(data.get("seconds", 0.0))
    physical = int(data.get("physical_decisions", 0)) or 1
    seconds_per_decision = seconds / physical

    by_step: dict[int, dict] = {}
    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(int(record["step"]), record["state"], record["macro"])].append(record)

    events = ["grasp_kept", "on_arm_end", "on_arm_any", "sustained"]
    for step in sorted({int(r["step"]) for r in records}):
        summary: dict[str, dict] = {}
        for event in events:
            cells_by_state: dict[str, dict[str, list[int]]] = defaultdict(dict)
            split_cells = 0
            total_cells = 0
            true_count = 0
            outcome_count = 0
            gap_true: list[float] = []
            gap_false: list[float] = []
            split_cell_values: list[list[tuple[int, float]]] = []
            for (rec_step, state, macro), group in grouped.items():
                if rec_step != step:
                    continue
                outcomes = [int(branch_events(r)[event]) for r in group]
                if 0 < sum(outcomes) < len(outcomes):
                    split_cell_values.append([
                        (outcome, float(r.get("sustained_coverage", 0.0) or 0.0))
                        for r, outcome in zip(group, outcomes)
                    ])
                cells_by_state[state][macro] = outcomes
                total_cells += 1
                true_count += sum(outcomes)
                outcome_count += len(outcomes)
                if 0 < sum(outcomes) < len(outcomes):
                    split_cells += 1
                for r, outcome in zip(group, outcomes):
                    value = float(r.get("sustained_coverage", 0.0) or 0.0)
                    (gap_true if outcome else gap_false).append(value)

            undecided_states = 0
            separable_states = 0
            etas: list[float] = []
            pvalues: list[float] = []
            for state, cells in cells_by_state.items():
                flat = [v for outcomes in cells.values() for v in outcomes]
                if 0 < sum(flat) < len(flat):
                    undecided_states += 1
                    eta, pvalue = permutation_eta(cells, trials, rng)
                    if not math.isnan(eta):
                        etas.append(eta)
                        pvalues.append(pvalue)
                        # "separable" = macro identity explains the event at this
                        # state at the 5 % level of the permutation test
                        if pvalue <= 0.05:
                            separable_states += 1
                    # a locatable flip needs one macro unanimously on each side
                    unanimous_true = any(sum(o) == len(o) for o in cells.values())
                    unanimous_false = any(sum(o) == 0 for o in cells.values())
                    if not (unanimous_true and unanimous_false):
                        separable_states -= 0  # bookkeeping only; reported below

            locatable_states = 0
            for state, cells in cells_by_state.items():
                unanimous_true = any(sum(o) == len(o) for o in cells.values())
                unanimous_false = any(sum(o) == 0 for o in cells.values())
                if unanimous_true and unanimous_false:
                    locatable_states += 1

            p_event = true_count / max(outcome_count, 1)
            mean_true = sum(gap_true) / len(gap_true) if gap_true else float("nan")
            mean_false = sum(gap_false) / len(gap_false) if gap_false else float("nan")
            summary[event] = {
                "states": len(cells_by_state),
                "cells": total_cells,
                "event_rate": p_event,
                "within_command_split_cells": split_cells,
                "within_command_split_fraction": split_cells / max(total_cells, 1),
                "undecided_states": undecided_states,
                "locatable_flip_states": locatable_states,
                "macro_separable_states_p05": max(separable_states, 0),
                "mean_permutation_eta": (sum(etas) / len(etas)) if etas else float("nan"),
                "median_permutation_p": (sorted(pvalues)[len(pvalues) // 2] if pvalues else float("nan")),
                "sustained_coverage_event_true": mean_true,
                "sustained_coverage_event_false": mean_false,
                "sustained_coverage_gap": (mean_true - mean_false) if gap_true and gap_false else float("nan"),
                "boundary_jump": boundary_jump(split_cell_values),
                "repeats_for_0.25_difference_at_3sd": repeats_for_difference(p_event, 0.25),
                "repeats_for_0.50_difference_at_3sd": repeats_for_difference(p_event, 0.50),
            }
            cost = summary[event]["repeats_for_0.25_difference_at_3sd"] * per_branch_decisions
            summary[event]["decisions_per_candidate_at_0.25"] = cost
            summary[event]["seconds_per_candidate_at_0.25"] = cost * seconds_per_decision
        summary["_macro_selection"] = macro_selection(records, step)
        by_step[step] = summary

    return {
        "path": path,
        "interpretation": {
            "boundary_jump": "within-cell event-conditioned coverage gap; not a measured discontinuity",
            "within_command_split_fraction": "fraction of cells with mixed repeats; not pairwise flip probability",
            "locatable_flip_states": "opposite unanimous macro labels; no continuous boundary localization",
            "query_cost": "independent Bernoulli two-mean precision heuristic; not an estimator lower bound",
        },
        "checkpoint": data.get("checkpoint"),
        "macros": data.get("macros"),
        "repeats": data.get("repeats"),
        "window": window,
        "follow": follow,
        "decisions_per_branch": per_branch_decisions,
        "seconds_per_decision": seconds_per_decision,
        "records": len(records),
        "by_step": by_step,
    }


def macro_selection(records: list[dict], step: int) -> dict:
    """What is choosing the macro worth at this step, held out across repeats?

    For each state: choose the macro by the mean of all repeats but one, then score
    it on the excluded repeat. Average over exclusions. Compared against always
    running the policy macro, and against the oracle best macro.
    """
    by_state: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if int(record["step"]) != step:
            continue
        by_state[record["state"]][record["macro"]].append(
            float(record.get("sustained_coverage", 0.0) or 0.0)
        )

    policy_scores: list[float] = []
    oracle_scores: list[float] = []
    heldout_scores: list[float] = []
    best_macro_counts: dict[str, int] = defaultdict(int)
    for state, macros in by_state.items():
        if "policy" not in macros:
            continue
        repeats = min(len(v) for v in macros.values())
        if repeats < 2:
            continue
        policy_mean = sum(macros["policy"]) / len(macros["policy"])
        means = {m: sum(v) / len(v) for m, v in macros.items()}
        best = max(means, key=means.get)
        best_macro_counts[best] += 1
        policy_scores.append(policy_mean)
        oracle_scores.append(means[best])
        for held in range(repeats):
            chosen, chosen_value = None, None
            for macro, values in macros.items():
                rest = [v for i, v in enumerate(values[:repeats]) if i != held]
                value = sum(rest) / len(rest)
                if chosen_value is None or value > chosen_value:
                    chosen, chosen_value = macro, value
            heldout_scores.append(macros[chosen][held])

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    return {
        "states": len(policy_scores),
        "policy_macro_mean": mean(policy_scores),
        "oracle_macro_mean": mean(oracle_scores),
        "heldout_selected_mean": mean(heldout_scores),
        "oracle_gain": mean(oracle_scores) - mean(policy_scores),
        "heldout_gain": mean(heldout_scores) - mean(policy_scores),
        "best_macro_counts": dict(best_macro_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/uipc_manip")
    parser.add_argument("--out", required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    collections = load_collections(args.root)
    if not collections:
        raise SystemExit(f"no branch collections with >=3 repeats under {args.root}")

    report = {
        "root": args.root,
        "permutations": args.permutations,
        "seed": args.seed,
        "collections": [analyse(path, data, args.permutations, rng) for path, data in collections],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)

    for collection in report["collections"]:
        print(f"\n=== {collection['path']}  ({collection['records']} branches, "
              f"{collection['seconds_per_decision']:.3f} s/decision)")
        for step, summary in collection["by_step"].items():
            choice = summary.get("_macro_selection", {})
            if choice.get("states"):
                print(
                    f"  step {step:>3} macro choice: policy {choice['policy_macro_mean']:.4f} | "
                    f"held-out {choice['heldout_selected_mean']:.4f} ({choice['heldout_gain']:+.4f}) | "
                    f"oracle {choice['oracle_macro_mean']:.4f} ({choice['oracle_gain']:+.4f}) | "
                    f"{choice['states']} states | best {choice['best_macro_counts']}"
                )
            for event, values in summary.items():
                if event.startswith("_"):
                    continue
                if values["undecided_states"] == 0 and values["event_rate"] in (0.0, 1.0):
                    continue
                print(
                    f"  step {step:>3} {event:<11} rate {values['event_rate']:.3f} | "
                    f"split cells {values['within_command_split_fraction']:.3f} | "
                    f"undecided {values['undecided_states']}/{values['states']} | "
                    f"opposite-label states {values['locatable_flip_states']}/{values['states']} | "
                    f"macro-separable {values['macro_separable_states_p05']} | "
                    f"gap {values['sustained_coverage_gap']:+.4f} | "
                    f"within-cell gap {values['boundary_jump']['mean_jump']:+.4f}"
                    f"+-{values['boundary_jump'].get('se_jump', float('nan')):.4f}"
                    f"({values['boundary_jump']['cells']}) | "
                    f"{values['seconds_per_candidate_at_0.25']:.0f} s/candidate"
                )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
