"""Report the fixed dynamic-student holdout without launching or altering jobs.

Counts are per checkpoint and task cell. Seed/repeat outcomes are descriptive;
no significance test treats them as additional independent tasks. --watch is a
bounded CPU-only companion to the already running pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import time

from run_dynamic_pipeline import alive, save


def expected_runs(repeats):
    if repeats not in (1, 2):
        raise ValueError("Unknown evaluation repeat count")
    models = ("none", "gicp", "r1_fixed_encoder") + tuple(
        f"{mode}_seed{seed}" for seed in range(3) for mode in ("history", "current"))
    return {f"test_body{body}_{clip}{'_rep' + str(rep) if repeats > 1 else ''}_{model}":
            dict(body=body, clip=clip, repeat=rep, model=model)
            for body in (14054, 14055) for clip in ("pass", "phone")
            for rep in range(repeats) for model in models}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)
    result = {}
    for model, items in sorted(groups.items()):
        counts = Counter(row["status"] for row in items)
        valid = [row for row in items if row["status"] == "valid"]
        successes = sum(row["final_success"] for row in valid)
        result[model] = dict(expected=len(items), statuses=dict(counts), valid=len(valid),
                             successes=successes, success_rate=successes / len(valid) if valid else None,
                             task_failures=dict(Counter(row["failure_kind"] for row in valid
                                                       if not row["final_success"])),
                             elapsed_s=sum(row["elapsed_s"] for row in items if "elapsed_s" in row))
    return result


def paired_history_current(rows):
    lookup = {row["name"]: row for row in rows}
    cells = defaultdict(list)
    expected = Counter()
    for row in rows:
        if not row["model"].startswith("history_seed"):
            continue
        key = f"body{row['body']}_{row['clip']}"
        expected[key] += 1
        other = lookup[row["name"].replace("_history_seed", "_current_seed")]
        if row["status"] == other["status"] == "valid":
            cells[key].append(int(row["final_success"]) - int(other["final_success"]))
    return {key: dict(expected_pairs=n, valid_pairs=len(cells[key]),
                     mean_success_difference=sum(cells[key]) / len(cells[key]) if cells[key] else None,
                     history_only_success=sum(x == 1 for x in cells[key]),
                     current_only_success=sum(x == -1 for x in cells[key]))
            for key, n in sorted(expected.items())}


def report(root):
    config = json.loads((root / "pipeline_config.json").read_text())
    state = json.loads((root / "pipeline_status.json").read_text())
    expected = expected_runs(config["evaluation_repeats"])
    manifest = root / "heldout_evaluation_jobs.json"
    jobs = json.loads(manifest.read_text()) if manifest.exists() else []
    errors = []
    by_name = {job["name"]: job for job in jobs}
    if manifest.exists() and (len(by_name) != len(jobs) or set(by_name) != set(expected)):
        errors.append("Evaluation manifest differs from the fixed body/clip/model/repeat grid")
    rows = []
    for name, design in expected.items():
        row = dict(name=name, **design, status="pending")
        rows.append(row)
        job = by_name.get(name)
        status = state.get("jobs", {}).get(name, {}).get("status", "pending")
        if job is None:
            continue
        marker = Path(job["complete"])
        if marker.resolve() != (root / name / "metrics.json").resolve():
            row.update(status="protocol_error", error="Unexpected result path")
            errors.append(name + ": unexpected result path")
            continue
        if not marker.exists():
            row["status"] = "running" if status == "running" else "pending"
            continue
        # A worker can write its metrics before finishing cleanup. Wait until
        # it exits; a stale running flag must not hide a terminal physics error.
        launch_path = root / f"{name}_launch.json"
        if status == "running" and launch_path.exists():
            launch = json.loads(launch_path.read_text())
            if alive(launch["pid"]):
                row["status"] = "finishing"
                continue
        metrics = json.loads(marker.read_text())
        if len(metrics) != 1:
            raise ValueError(f"Expected one policy result in {marker}")
        metric = metrics[0]
        if not isinstance(metric.get("final_success"), bool):
            raise ValueError(f"Missing final-success outcome in {marker}")
        failure = metric.get("failure") or {}
        invalid = failure.get("kind") == "invalid_physics"
        row.update(status="invalid_physics" if invalid else "valid",
                   final_success=metric["final_success"] if not invalid else None,
                   failure_kind=failure.get("kind", "timeout" if not metric["final_success"] else None),
                   steps=metric["steps"], elapsed_s=metric["elapsed_s"],
                   first_success_step=metric.get("first_success_step"))
        if not invalid and metric["final_success"] and (failure or metric["steps"] != 450):
            row.update(status="protocol_error", error="Success without the complete valid endpoint")
            errors.append(name + ": incomplete/failed episode marked successful")
    counts = Counter(row["status"] for row in rows)
    complete = (state["stage"] == "complete" and counts["valid"] == len(expected) and not errors)
    return dict(pipeline_stage=state["stage"], pipeline_error=state.get("error"),
                expected_episodes=len(expected), expected_task_cells=4, statuses=dict(counts),
                evaluation_complete=complete, protocol_errors=errors,
                policies=summarize(rows), history_current_by_cell=paired_history_current(rows),
                episodes=rows,
                interpretation="Descriptive pilot only. Repeats/training seeds share task cells; "
                               "pending and invalid physics are not policy failures. "
                               "Reported elapsed time covers evaluation workers, not total training cost.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.)
    parser.add_argument("--max-hours", type=float, default=72.)
    args = parser.parse_args()
    if args.interval < 1 or args.max_hours <= 0:
        parser.error("Polling interval and time limit must be positive")
    root = args.root.resolve()
    start = time.monotonic()
    with (root / "evaluation_report.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            value = report(root)
            owner = json.loads((root / "pipeline_launch.json").read_text())
            owner_alive = alive(owner["pid"])
            value["pipeline_owner_alive"] = owner_alive
            if value != previous:
                save(root / "evaluation_report.json", value)
                print(json.dumps({k: value[k] for k in ("pipeline_stage", "statuses", "evaluation_complete",
                                                       "protocol_errors", "pipeline_owner_alive")}), flush=True)
                previous = value
            if not args.watch:
                return
            if value["pipeline_stage"] == "complete":
                if not value["evaluation_complete"]:
                    raise RuntimeError("Queue complete but evaluation grid is incomplete; inspect report")
                return
            if value["protocol_errors"] or value["pipeline_stage"] in (
                    "error", "teacher_gate_failed", "heldout_preflight_failed") or not owner_alive:
                raise RuntimeError("Pipeline stopped or protocol check failed; inspect report before resuming")
            if time.monotonic() - start >= args.max_hours * 3600:
                raise TimeoutError("Report watcher reached its time limit; pipeline is unchanged")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
