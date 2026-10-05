"""Generate a source/target decision report from the actual repair-pilot records.

Repeated development bodies are not independent test tasks. The hindsight bank
coverage is descriptive across separately settled trajectories, not an executed
oracle or a same-state causal bound. No teacher result is an actor result.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

from run_sim2real_audit import digest, write_json


def supervisor_alive(pid):
    """Inspect the actual process identity; a stale status file is insufficient."""
    root = Path("/proc") / str(pid)
    try:
        if root.joinpath("stat").read_text().split(")", 1)[1].split()[0] == "Z":
            return False
        return b"run_feedback_repair_pilot.py" in root.joinpath("cmdline").read_bytes()
    except FileNotFoundError:
        return False


def render(out, destination):
    protocol = json.loads((out / "protocol.json").read_text())
    status = json.loads((out / "status.json").read_text())
    batch_paths = sorted((out / "completed").glob("*.json"))
    batches = [json.loads(p.read_text()) for p in batch_paths]
    rows = [r for b in batches for r in b["rows"]]
    events = [json.loads(l) for l in (out / "events.jsonl").read_text().splitlines()]
    source = [r for r in rows if r["phase"] == "source"]
    target = [r for r in rows if r["phase"] == "target"]
    fit = json.loads((out / "source_fit.json").read_text()) if (out / "source_fit.json").exists() else None
    analysis = json.loads((out / "analysis.json").read_text()) if (out / "analysis.json").exists() else None
    source_complete = len(source) == 84
    terminal = status["stage"] in ("complete", "source_mechanism_rejected", "budget_reached",
                                   "worker_budget_reached", "memory_guard", "worker_error",
                                   "supervisor_error", "needs_inspection", "incomplete_batch", "stopped_by_request")
    # Check every recorded admission/exit, not just the final running status.
    active, peak_concurrency, accounting_errors = set(), 0, []
    for event in events:
        if event["event"] == "worker_start":
            active.add(event["pid"])
            peak_concurrency = max(peak_concurrency, len(active))
        elif event["event"] == "worker_end":
            if event["pid"] not in active:
                accounting_errors.append("Exit without admission: " + str(event["pid"]))
            active.discard(event["pid"])
    ends = [e for e in events if e["event"] == "worker_end"]
    cost = dict(charged_attempts=7 * sum(e["event"] == "worker_start" for e in events),
                finished_attempts=len(rows), worker_seconds=sum(e["worker_seconds"] for e in ends),
                peak_total_gpu_mib=max((e["peak_total_gpu_mib"] for e in ends), default=None),
                peak_sim_workers=peak_concurrency, accounting_errors=accounting_errors,
                recorded_active_workers=sorted(active))
    units = defaultdict(dict)
    for row in source:
        key = (row["condition"], row["body"], row["repeat"])
        name = row["feedback_repair"]["controller"]["name"]
        if name in units[key]:
            raise ValueError(f"Duplicate source unit/controller: {key}, {name}")
        units[key][name] = row
    conditions = ["nominal", "bending_x2", "density_x1p5"]
    names = [c["name"] for c in protocol["source_controllers"]]
    counts, changes, coverage = {}, {}, []
    for condition in conditions:
        cases = [v for k, v in units.items() if k[0] == condition and set(v) == set(names)]
        counts[condition] = {name: dict(successes=sum(bool(v[name]["accepted"]) for v in cases),
                                       attempts=len(cases)) for name in names}
    for name in names[1:]:
        cases = [v for v in units.values() if "base" in v and name in v]
        changes[name] = dict(units=len(cases), wins=sum(v[name]["accepted"] and not v["base"]["accepted"] for v in cases),
                            losses=sum(v["base"]["accepted"] and not v[name]["accepted"] for v in cases))
    for key, values in units.items():
        if set(values) != set(names):
            continue
        coverage.append(dict(condition=key[0], body=key[1], repeat=key[2],
                             base_success=bool(values["base"]["accepted"]),
                             any_alternative_success=any(values[name]["accepted"] for name in names[1:])))
    replicas = defaultdict(dict)
    for row in source:
        name = row["feedback_repair"]["controller"]["name"]
        replicas[(row["condition"], row["body"], name)][row["repeat"]] = bool(row["accepted"])
    repeat_agreement = {}
    for name in names:
        pairs = [v for k, v in replicas.items() if k[2] == name and set(v) == {0, 1}]
        repeat_agreement[name] = dict(pairs=len(pairs), agreeing=sum(v[0] == v[1] for v in pairs),
                                      both_success=sum(v[0] and v[1] for v in pairs))
    target_counts = {c["name"]: dict(successes=sum(r["accepted"] for r in target if r["feedback_repair"]["controller"]["name"] == c["name"]),
                                        attempts=sum(r["feedback_repair"]["controller"]["name"] == c["name"] for r in target))
                     for c in protocol["target_controllers"]}
    record = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), phase=status["stage"],
                  terminal=terminal, source_complete=source_complete, cost=cost, source_counts=counts,
                  source_matched_reset_changes=changes, hindsight_bank_coverage=coverage,
                  source_repeat_agreement=repeat_agreement,
                  target_counts=target_counts, source_gate=fit["source_gate"] if fit else None,
                  source_compilations={k: v["compilation"] for k, v in fit["tables"].items()} if fit else None,
                  integrity_summary={k: v for k, v in analysis.items() if k not in ("outcomes", "reset_checks")} if analysis else None,
                  source_files_sha256={str(p): digest(p) for p in [out / "protocol.json", *batch_paths]},
                  limits=["Two bodies, two repeats, one garment; no publication-scale significance claim.",
                          "Matched reset trajectories diverge before intervention; no identical-state causal claim.",
                          "Hindsight coverage uses separately settled candidates and future outcomes; it is not a deployable oracle.",
                          "Current yaw adapter differs from a strict translation-only robot interface.",
                          "No actor post-training or hardware transfer is established by this pilot."])
    write_json(out / "decision_evidence.json", record)
    lines = ["# Observable feedback-repair pilot: actual outcomes", "",
             "Status: **" + status["stage"] + "**. " + ("Terminal record." if terminal else "Interim record; collection is still active."), "",
             f"Recorded {len(source)}/84 source and {len(target)}/28 possible target attempts. "
             "Target execution depends on the frozen source gate.", "",
             "## Full dressing success", "",
             "All entries include the unchanged physical sleeve criterion, valid grasp and 20-decision hold. "
             "The final design has two bodies x two repeats per source condition; interim denominators "
             "include only completed blocks. Repeats do not create new independent bodies.", "",
             "| Source physics | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for condition in conditions:
        lines.append("| " + condition + " | " + " | ".join(f"{counts[condition][n]['successes']}/{counts[condition][n]['attempts']}" for n in names) + " |")
    lines += ["", "p0 = half-action prefix; p1 = no-rotation prefix; p2 = upward-residual prefix. "
              "s0 = half-action suffix; s1 = no-rotation suffix. Prefix begins at decision 60, "
              "lasts 12 decisions, suffix lasts 80, and FMVP resumes at decision 152.", "",
              "## Changes relative to base in matched reset blocks", "",
              "These are observed win/loss pairs over independent settling, not exact-state interventions.", "",
              "| Correction | New success where base failed | Loss where base succeeded | Blocks |",
              "|---|---:|---:|---:|"]
    for name in names[1:]:
        v = changes[name]
        lines.append(f"| {name} | {v['wins']} | {v['losses']} | {v['units']} |")
    missing_coverage = sum(not v["base_success"] and not v["any_alternative_success"] for v in coverage)
    positive_coverage = sum(not v["base_success"] and v["any_alternative_success"] for v in coverage)
    lines += ["", f"Among {len(coverage)} complete source blocks, {positive_coverage} have an alternative success "
              f"where base fails; {missing_coverage} have no success from either base or the six alternatives. "
              "This hindsight coverage describes the sampled bank and cannot be reported as an oracle policy result.", "",
              "| Same-controller independent repeats | Agreeing outcomes | Both succeed |", "|---|---:|---:|"]
    for name, value in repeat_agreement.items():
        lines.append(f"| {name} | {value['agreeing']}/{value['pairs']} | {value['both_success']}/{value['pairs']} |")
    lines += ["", "Agreement is descriptive over repeated body/physics blocks; neither repeated runs nor "
              "the seven controllers count as new independent recipients.", "",
              "## Source compilation and target execution", ""]
    if fit is None:
        lines.append("Source compilation has not run; all 84 source attempts are required. No target feedback result exists yet.")
    else:
        gate = fit["source_gate"]
        lines.append(f"Frozen source gate: **{'passed' if gate['proceed_to_target'] else 'failed'}**. "
                     f"New-success rows = {gate['new_repair_successes']}; represented repeats = {gate['new_repair_replicas']}; "
                     f"observable-minus-flat estimated mean gain = {gate['observable_minus_flat_mean_gain']:.4f}.")
        lines += ["", "| Source-only compiler | Worst physics gain | Mean physics gain |", "|---|---:|---:|"]
        for name in ("observable", "flat", "initial", "privileged"):
            v = fit["tables"][name]["compilation"]
            lines.append(f"| {name} | {v['worst_model_gain']:.4f} | {v['mean_model_gain']:.4f} |")
        lines += ["", "Conditional values include pessimistic zero values for unsupported cells. "
                  "Source estimates use replica-cross-fitted routers; the final router is refit on source only. "
                  "These are estimated finite-model gains, not real guarantees."]
    if target:
        lines += ["", "| Executed joint-shift controller | Full success |", "|---|---:|"]
        for name, value in target_counts.items():
            lines.append(f"| {name} | {value['successes']}/{value['attempts']} |")
    elif source_complete and fit and not fit["source_gate"]["proceed_to_target"]:
        lines += ["", "The 28 target attempts and actor training were not launched because the unchanged bank "
                  "did not support the predeclared feedback advantage."]
    lines += ["", "## Integrity, cost and interpretation", "",
              f"Charged admissions: {cost['charged_attempts']}/112 attempts. Finished-worker cost: "
              f"{cost['worker_seconds'] / 3600:.3f}/4 worker-hours. Recorded maximum concurrent sim workers: "
              f"{cost['peak_sim_workers']}."]
    if cost["peak_total_gpu_mib"] is not None:
        lines.append(f"Recorded peak total GPU use: {cost['peak_total_gpu_mib'] / 1024:.3f} GiB / 90 GiB.")
    if analysis:
        lines.append(f"Saved-state integrity checks pass: {analysis['integrity_pass']} over "
                     f"{analysis['completed_attempts']} analyzed attempts; "
                     f"{'all currently reported attempts' if analysis['completed_attempts'] == len(rows) else 'a partial snapshot of the reported attempts'}. "
                     f"Maximum initial cloth gap: {analysis['max_initial_vertex_gap_m'] * 1000:.3f} mm; "
                     f"maximum preintervention vertex gap: {analysis['max_preintervention_vertex_gap_m']:.3f} m.")
    lines += ["", *["- " + v for v in record["limits"]], "",
              "The [method specification](2026-10-05-feedback-repair-posttraining.md) defines the proposal, "
              "strongest prior-art objections and required student comparisons. "
              "The [research decision](2026-10-05-sim2real-posttraining-decision.md) also preserves "
              "the completed Q1–4 evidence and the rejected calibration screen."]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    return record


def watch(out, destination):
    """Update evidence after new records, then exit at the producer's endpoint."""
    from analyze_feedback_repair_pilot import analyze

    lock = (out / ".decision_reporter.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    while True:
        state = json.loads((out / "status.json").read_text())
        files = sorted((out / "completed").glob("*.json"))
        signature = (tuple(p.name for p in files), state["stage"],
                     digest(out / "source_fit.json") if (out / "source_fit.json").exists() else None)
        if signature != previous:
            try:
                analyze(out)
                record = render(out, destination)
            except json.JSONDecodeError:
                current = json.loads((out / "status.json").read_text())
                if current["stage"] != "fitting_source_routers" or not supervisor_alive(current["supervisor_pid"]):
                    raise
                # The frozen source fitter writes its JSON before returning.
                # A read during that write is an observation race, not failure.
                time.sleep(1)
                continue
            previous = signature
            print(json.dumps(dict(phase=record["phase"], attempts=record["cost"]["finished_attempts"],
                                  integrity=record["integrity_summary"]["integrity_pass"])), flush=True)
        write_json(out / "reporter_status.json", dict(pid=os.getpid(), producer_pid=state["supervisor_pid"],
                   updated_utc=datetime.now(timezone.utc).isoformat(), phase=record["phase"],
                   analyzed_attempts=record["integrity_summary"]["completed_attempts"], report=str(destination)))
        if record["terminal"]:
            if record["integrity_summary"]["completed_attempts"] != record["cost"]["finished_attempts"]:
                previous = None
                continue
            return record
        if not supervisor_alive(state["supervisor_pid"]):
            # Do not convert an incomplete record into a successful endpoint or
            # restart a simulator. The investigator must inspect the lost process.
            raise RuntimeError("Producer exited without a terminal status; saved report remains incomplete")
        time.sleep(30)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("output/uipc_manip/feedback_repairs_20261005"))
    p.add_argument("--report", type=Path, default=Path("agent_docs/performance/2026-10-05-feedback-repair-results.md"))
    p.add_argument("--watch", action="store_true", help="CPU-only analysis after each new batch; stop at producer endpoint")
    a = p.parse_args()
    r = (watch if a.watch else render)(a.out.resolve(), a.report.resolve())
    print(json.dumps({k: r[k] for k in ("phase", "source_complete", "cost", "source_gate")}, indent=2))
