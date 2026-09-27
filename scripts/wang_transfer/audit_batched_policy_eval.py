"""Audit an existing eval_policies_batched run without controlling its workers.

Count against the declared policy/body/garment grid, retain missing results and
illegal starts separately, check saved acceptance evidence, and measure actual
initial-state differences. The optional watcher only reads experiment files
and atomically replaces its own report; it never starts or stops GPU work.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def read_rows(path):
    """A live writer may leave only its final JSONL record incomplete."""
    if not path.exists():
        return [], [], False
    text = path.read_text()
    lines = text.splitlines()
    rows, errors, partial = [], [], False
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1 and not text.endswith("\n"):
                partial = True
            else:
                errors.append(f"Invalid result line {i + 1}: {exc}")
    return rows, errors, partial


def episode_evidence(row, root, cache):
    path = resolve(row["log"], root).with_suffix("") / row["path"]
    run_path = path.parent.parent / "run.json"
    stamp = (str(path), path.stat().st_mtime_ns, run_path.stat().st_mtime_ns)
    if stamp not in cache:
        run = json.loads(run_path.read_text())
        with np.load(path, allow_pickle=False) as z:
            actions = z["actions"]
            start = {k: z[k][0].copy() for k in ("positions", "tcp", "obs")}
            start["human_vertices"] = z["human_vertices"].copy()
            valid = z["grasp_valid"]
            wrapped, fraction = z["sleeve_wrapped"], z["sleeve_proximal_upper_fraction"]
            saved = json.loads(str(z["metadata_json"].item()))
            errors = []
            if not all(np.isfinite(a).all() for a in (*start.values(), actions, fraction)):
                errors.append("Non-finite initial state, command or endpoint")
            if len(valid) != len(actions) or len(wrapped) != len(actions) + 1 or len(fraction) != len(wrapped):
                errors.append("Misaligned command/state arrays")
            if saved.get("accepted"):
                s, hold = saved.get("success_state"), int(run["hold"])
                if (not isinstance(s, int) or s < 0 or hold < 1 or len(actions) < s + hold
                        or not valid.all() or saved.get("sim_error")
                        or not bool(wrapped[s:].all())
                        or not bool((fraction[s:] >= float(run["stop_proximal_upper"])).all())
                        or not bool((actions[s:] == 0).all())):
                    errors.append("Accepted result lacks a complete valid zero-command hold")
        cache[stamp] = dict(path=str(path), start=start, run=run, saved=saved, errors=errors)
    evidence = cache[stamp]
    errors = list(evidence["errors"])
    for key in ("body", "seed", "accepted", "transitions", "checkpoint_sha256"):
        if row.get(key) != evidence["saved"].get(key):
            errors.append(f"Ledger/archive disagree on {key}")
    return dict(evidence, errors=errors)


def initial_difference(a, b):
    diff = {}
    for key in ("positions", "tcp", "human_vertices", "obs"):
        x, y = a[key], b[key]
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Invalid initial {key} shape or values")
        delta = x.astype(np.float64) - y.astype(np.float64)
        diff[key] = float(np.abs(delta).max() if key == "obs" else np.linalg.norm(delta, axis=-1).max())
    return diff


def audit(run_dir, root, cache=None, tolerance=1e-5):
    cache = {} if cache is None else cache
    if not (run_dir / "run.json").exists():
        return dict(state="waiting_for_run", run=str(run_dir))
    run = json.loads((run_dir / "run.json").read_text())
    policies, garments, bodies = list(run["policies"]), run["garments"], run["bodies"]
    expected = set(itertools.product(policies, garments, bodies))
    rows, errors, partial = read_rows(run_dir / "results.jsonl")
    checkpoints = {}
    for policy, filename in run["policies"].items():
        path = resolve(filename, root)
        if path.is_file():
            checkpoints[policy] = digest(path)
        else:
            errors.append(f"Missing evaluation checkpoint: {path}")
    grouped = collections.defaultdict(list)
    for row in rows:
        key = (row["policy"], row["garment"], row["body"])
        if key not in expected:
            errors.append(f"Unexpected result: {key}")
        else:
            grouped[key].append(row)
    results, evidence = {}, {}
    for key, values in grouped.items():
        if len(values) != 1:
            errors.append(f"Duplicate results require review: {key}")
            continue
        row = results[key] = values[0]
        if row.get("no_legal_start"):
            continue
        try:
            evidence[key] = episode_evidence(row, root, cache)
            if row.get("checkpoint_sha256") != checkpoints.get(key[0]):
                evidence[key]["errors"].append("Checkpoint file differs from the rollout fingerprint")
            errors.extend(f"{key}: {e}" for e in evidence[key]["errors"])
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{key}: cannot verify archive: {exc}")

    index_path = resolve(run["index"], root)
    index = json.loads(index_path.read_text())
    split_bodies = collections.defaultdict(set)
    for row in index["episodes"]:
        split_bodies[row["split"]].add(row["body"])
    overlaps = {f"{a}/{b}": sorted(split_bodies[a] & split_bodies[b])
                for a, b in itertools.combinations(sorted(split_bodies), 2)}
    if any(overlaps.values()) or set(bodies) - split_bodies[run["split"]]:
        errors.append("Evaluation bodies violate the declared dataset split")

    counts = {}
    for policy in policies:
        counts[policy] = {}
        for garment in garments:
            selected = [results[(policy, garment, b)] for b in bodies if (policy, garment, b) in results]
            attempts = [r for r in selected if not r.get("no_legal_start")]
            counts[policy][garment] = dict(planned=len(bodies), reported=len(selected),
                pending=len(bodies) - len(selected), no_legal_start=len(selected) - len(attempts),
                attempted=len(attempts), reported_accepted=sum(r.get("accepted") is True for r in attempts),
                failures=sum(r.get("accepted") is False for r in attempts),
                grasp_failures=sum("grasp" in str(r.get("sim_error") or "") for r in attempts))

    pairs = []
    settings = ("seed", "placement_offset_mm", "garment", "hang_sha256", "environment_sha256",
                "collector_sha256", "controls_sha256", "bridge_sha256", "package_revision",
                "collision_geometry", "cloth_density", "cloth_strain_rate", "fit_sleeve_ratio",
                "body_fit_filter", "hold", "steps", "stop_proximal_upper", "armhole_endpoint",
                "sections_wrap", "align_armhole_axis", "autonomous_hold")
    for garment, body in itertools.product(garments, bodies):
        keys = [(p, garment, body) for p in policies]
        if not all(k in results for k in keys):
            continue
        pair = dict(garment=garment, body=body, outcomes={p: results[k].get("accepted") for p, k in zip(policies, keys)})
        no_start = [bool(results[k].get("no_legal_start")) for k in keys]
        if any(no_start):
            pair["state"] = "no_legal_start" if all(no_start) else "unmatched_eligibility"
        elif not all(k in evidence and not evidence[k]["errors"] for k in keys):
            pair["state"] = "unverified_artifact"
        else:
            reference = evidence[keys[0]]
            differences, mismatch = {}, {}
            try:
                for key in keys[1:]:
                    other = evidence[key]
                    differences[key[0]] = initial_difference(reference["start"], other["start"])
                    mismatch[key[0]] = [s for s in settings if reference["run"].get(s) != other["run"].get(s)]
                pair.update(initial_difference=differences, settings_mismatch=mismatch)
                close = all(d[k] <= tolerance for d in differences.values() for k in ("positions", "tcp", "human_vertices"))
                pair["state"] = ("settings_mismatch" if any(mismatch.values()) else
                                 "matched_within_tolerance" if close else "initial_state_mismatch")
            except ValueError as exc:
                pair.update(state="invalid_initial_state", error=str(exc))
        pairs.append(pair)
    complete = len(results) == len(expected) and not partial
    return dict(state="complete" if complete and not errors else "complete_with_errors" if complete else "incomplete",
                run=str(run_dir), expected_results=len(expected), reported_results=len(results),
                pending_results=len(expected) - len(results), partial_last_line=partial,
                counts=counts, pair_counts=dict(collections.Counter(p["state"] for p in pairs)), pairs=pairs,
                initial_tolerance_m=tolerance, index_sha256=digest(index_path),
                checkpoint_sha256=checkpoints,
                run_sha256=digest(run_dir / "run.json"), index_summary=index.get("summary"),
                split_overlap=overlaps, errors=errors,
                scope="Student body/pose holdout among bodies with an accepted source recording; five seen garments. "
                      "Saved acceptance flags and holds are checked; geometry/contact are not independently recomputed. "
                      "Initial-state mismatch prevents an identical-start claim. Partial counts are not final rates.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=Path("/home/ge47gax/kun/libuipc"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--watch-seconds", type=float, default=0)
    p.add_argument("--interval", type=float, default=30)
    args = p.parse_args()
    if args.watch_seconds < 0 or not 1 <= args.interval <= 60:
        p.error("Nonnegative watch budget and interval in [1, 60] seconds required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    deadline, cache = time.monotonic() + args.watch_seconds, {}
    previous = None
    while True:
        report = audit(args.run.resolve(), args.source_root.resolve(), cache)
        report["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        temporary = args.out.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.out)
        status = {k: report.get(k) for k in ("state", "reported_results", "expected_results", "pair_counts", "errors")}
        if status != previous:
            print(json.dumps(status), flush=True)
            previous = status
        if time.monotonic() >= deadline or report["state"].startswith("complete"):
            break
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
