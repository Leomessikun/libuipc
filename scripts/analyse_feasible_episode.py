"""Audit the predeclared three-arm continuation without running physics.

The collector stores the continuation, not the maximum tracking error of its
approach prefix. Consequently these are branch-valid outcomes, not certified
whole-episode outcomes. Repeats and slots are reported descriptively.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def analyse(path: Path) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    if not data["completed"]:
        raise ValueError("Collection is incomplete; do not report a final comparison")
    horizon = data["window"] + data["follow"]
    expected = len(data["states"]) * len(data["macros"]) * data["repeats"]
    records = data["records"]
    if len(records) != expected:
        raise ValueError(f"Expected {expected} branches, got {len(records)}")
    seen = set()
    groups = defaultdict(list)
    per_state = defaultdict(dict)
    for record in records:
        key = (record["state"], record["macro"], record["repeat"])
        if key in seen:
            raise ValueError(f"Duplicate branch {key}")
        seen.add(key)
        trace = record["trace"]
        if len(trace) != horizon:
            raise ValueError(f"Truncated trace {key}")
        tracking = [float(row["tracking_error"]) for row in trace]
        upper = [float(row["upperarm_ratio"]) for row in trace]
        if not all(math.isfinite(x) for x in tracking + upper):
            raise ValueError(f"Nonfinite trace {key}")
        valid = max(tracking) <= 0.02
        sustained = min(upper[-data["sustained_decisions"]:])
        if valid != record["whole_branch_grasp_valid"] or not math.isclose(
            sustained, record["sustained_coverage"], abs_tol=1e-10
        ):
            raise ValueError(f"Stored summary disagrees with trace {key}")
        first = next((i + 1 for i, value in enumerate(tracking) if value > 0.02), None)
        row = dict(
            state=record["state"], repeat=record["repeat"],
            branch_valid=valid,
            macro_window_valid=max(tracking[:data["window"]]) <= 0.02,
            sustained_coverage=sustained,
            valid_sustained_coverage=sustained if valid else 0.0,
            branch_valid_success=valid and sustained >= 0.7,
            final_coverage=upper[-1], max_coverage=max(upper),
            first_violation_relative_decision=first,
        )
        groups[record["macro"]].append(row)
        per_state[(record["state"], record["repeat"])][record["macro"]] = row
    expected_grid = {
        (state["state"], macro, repeat)
        for state in data["states"] for macro in data["macros"]
        for repeat in range(data["repeats"])
    }
    if seen != expected_grid:
        raise ValueError("Branch identities do not match the declared design")
    arms = {}
    for macro, rows in groups.items():
        arms[macro] = dict(
            n=len(rows),
            branch_valid=sum(r["branch_valid"] for r in rows),
            macro_window_valid=sum(r["macro_window_valid"] for r in rows),
            delayed_violations=sum(r["macro_window_valid"] and not r["branch_valid"] for r in rows),
            branch_valid_success=sum(r["branch_valid_success"] for r in rows),
            mean_sustained_coverage=mean(r["sustained_coverage"] for r in rows),
            mean_valid_sustained_coverage=mean(r["valid_sustained_coverage"] for r in rows),
            mean_final_coverage=mean(r["final_coverage"] for r in rows),
            per_repeat={str(rep): {
                "n": len(sub),
                "branch_valid": sum(r["branch_valid"] for r in sub),
                "mean_sustained_coverage": mean(r["sustained_coverage"] for r in sub),
            } for rep in range(data["repeats"])
                if (sub := [r for r in rows if r["repeat"] == rep])},
            branches=rows,
        )
    contrasts = {}
    for macro in data["macros"]:
        if macro == "policy":
            continue
        delta = [r[macro]["valid_sustained_coverage"] - r["policy"]["valid_sustained_coverage"]
                 for r in per_state.values()]
        contrasts[macro] = dict(mean_valid_coverage_difference=mean(delta),
                                positive=sum(d > 0 for d in delta),
                                zero=sum(d == 0 for d in delta),
                                negative=sum(d < 0 for d in delta))
    branch_decisions = expected * horizon
    return dict(
        source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
        completed=True, window=data["window"], follow=data["follow"],
        snapshot_steps=data["snapshot_steps"], slots=data["slots"], repeats=data["repeats"],
        branch_decisions=branch_decisions,
        retrospective_first_violation_stopping_decisions=sum(
            r["first_violation_relative_decision"] or horizon
            for rows in groups.values() for r in rows
        ),
        total_physical_decisions=data["physical_decisions"],
        approach_decisions=data["physical_decisions"] - branch_decisions,
        seconds=data["seconds"], max_restore_error_m=max(data["restore_errors"]),
        prefix_validity="not recorded; whole-episode validity cannot be certified",
        inference_scope="one garment/body/checkpoint; repeated slots are not independent task draws",
        arms=arms, descriptive_paired_contrasts=contrasts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({m: {k: v for k, v in a.items() if k != "branches"}
                      for m, a in result["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
