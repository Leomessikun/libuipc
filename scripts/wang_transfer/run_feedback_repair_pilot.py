"""Finite source/held-physics feedback-repair pilot: one guarded IPC worker.

One invocation prepares frozen sources, collects at most 84 source attempts,
fits source-only routers, and executes 28 actual target controllers only when
the preregistered mechanism gate survives. No actor training is auto-queued.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from collect_scaled import COMMON, OFFSETS_MM, PY, ROOT
from eval_policies_batched import garment_cli
from feedback_repair_control import pilot_settings
from run_sim2real_audit import (PACKAGE, POLICIES, digest, environment, gpu_used_mib,
                               ledger, status, terminate_group, write_json)


OUT = ROOT / "output/uipc_manip/feedback_repairs_20261005"
CPU_PY = "/home/ge47gax/miniconda3/envs/curl/bin/python"
SOURCE_CONDITIONS = [("nominal", []), ("bending_x2", ["--cloth-bending-stiffness", ".2"]),
                     ("density_x1p5", ["--cloth-density", "1125"])]
TARGET = ("bending_x2_density_x1p5", ["--cloth-bending-stiffness", ".2", "--cloth-density", "1125"])


def source_controllers():
    return [dict(name="base", kind="base")] + [dict(name=f"p{p}_s{s}", kind="fixed", prefix=p, suffix=s)
                                                   for p in range(3) for s in range(2)]


def target_controllers():
    return [dict(name=kind, kind=kind) for kind in ("base", "flat", "observable", "privileged",
                                                  "initial", "shuffled", "resample")]


def prepare(out):
    if (out / "protocol.json").exists():
        return json.loads((out / "protocol.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for name in ("source", "completed", "runs", "configs", "attempts"):
        (out / name).mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p, out / "source" / p.name)
    jobs = []
    for repeat in (0, 1):
        for condition, extra in SOURCE_CONDITIONS:
            for body in (1032, 1041):
                jobs.append(dict(id=len(jobs), phase="source", condition=condition, extra=extra,
                                 body=body, repeat=repeat, seed=2026100510 + repeat))
    for repeat in (0, 1):
        for body in (1032, 1041):
            jobs.append(dict(id=len(jobs), phase="target", condition=TARGET[0], extra=TARGET[1],
                             body=body, repeat=repeat, seed=2026100510 + repeat))
    dependencies = [out / "source" / n for n in ("collect_garment.py", "feedback_repair_control.py",
                    "fit_feedback_repairs.py", "compile_feedback_repairs.py", "rollout_controls.py",
                    "physical_sleeve.py", "sim2real_perturbations.py", "run_feedback_repair_pilot.py")]
    dependencies += [PACKAGE / "uipc_manip" / n for n in ("dressing_env.py", "wang_bridge.py", "wang_client.py", "obs.py")]
    protocol = dict(version=1, created_unix=time.time(),
                    revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    checkpoint=POLICIES["fmvp_sim"], checkpoint_sha256=digest(POLICIES["fmvp_sim"]),
                    source_sha256={str(p): digest(p) for p in dependencies}, jobs=jobs,
                    settings=pilot_settings(), source_controllers=source_controllers(),
                    target_controllers=target_controllers(), bodies=[1032, 1041], garment="tshirt_26",
                    max_worker_hours=4., max_charged_attempts=112, memory_limit_gib=90.,
                    admission_below_gib=84., terminate_at_gib=89.75, max_concurrent_sim_workers=1,
                    source_gate="New repaired baseline failures in both independent replicas; OOF observable mixture mean gain >0 and worst gain >=0; >=.10 mean gain over robust flat mixture.",
                    router="Cost-sensitive decision stump; quartile thresholds; train on other replica for source value estimation; refit all source only for target execution.",
                    pairing="Matched static reset geometry, same body/seed/offset/config per block; independent settling, no bitwise counterfactual claim.",
                    source_models="Three physics conditions, averaging the two development bodies. Target joint shift never used for fitting.",
                    budget="Every worker admission charges seven attempts including failed setup/retries; all worker wall time including setup and saving is charged.",
                    interpretation="Rejection pilot only. No actor or hardware evidence; two repeats are not a statistical validation.")
    write_json(out / "protocol.json", protocol)
    status(out, "prepared", planned_source_attempts=84, maximum_attempts=112)
    return protocol


def summarize(out):
    batches = [json.loads(p.read_text()) for p in sorted((out / "completed").glob("*.json"))]
    rows = [r for b in batches for r in b["rows"]]
    counts = {}
    for row in rows:
        key = row["phase"] + "/" + row["condition"] + "/" + row["feedback_repair"]["controller"]["name"]
        cell = counts.setdefault(key, dict(successes=0, attempts=0))
        cell["attempts"] += 1
        cell["successes"] += int(row["accepted"])
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()] if (out / "events.jsonl").exists() else []
    ends = [e for e in events if e["event"] == "worker_end"]
    result = dict(completed_batches=len(batches), completed_attempts=len(rows), counts=counts,
                  charged_attempts=7 * sum(e["event"] == "worker_start" for e in events),
                  worker_seconds=sum(e["worker_seconds"] for e in ends),
                  peak_total_gpu_mib=max((e["peak_total_gpu_mib"] for e in ends), default=None))
    if (out / "source_fit.json").exists():
        result["source_gate"] = json.loads((out / "source_fit.json").read_text())["source_gate"]
    write_json(out / "summary.json", result)
    return result


def job_config(out, protocol, job):
    entries = protocol[job["phase"] + "_controllers"]
    # Predeclared rotation prevents one controller always occupying slot zero.
    k = job["id"] % len(entries)
    entries = entries[k:] + entries[:k]
    config = dict(settings=protocol["settings"], controllers=entries)
    if job["phase"] == "target":
        config["fitted"] = json.loads((out / "source_fit.json").read_text())
    repair = out / "configs" / f"{job['id']:03d}_repairs.json"
    profiles = out / "configs" / f"{job['id']:03d}_profiles.json"
    write_json(repair, config)
    write_json(profiles, [dict(name=e["name"]) for e in entries])
    return repair, profiles


def execute(out, protocol):
    # The persistent lock is shared by this pilot's invocations, independent of
    # the output directory, so a second launch cannot create a second worker.
    lock = (ROOT / "output/uipc_manip/.feedback_repair_worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for p, expected in protocol["source_sha256"].items():
        if digest(p) != expected:
            raise RuntimeError(f"Frozen source changed: {p}")
    if digest(protocol["checkpoint"]) != protocol["checkpoint_sha256"]:
        raise RuntimeError("Frozen fallback checkpoint changed")
    env = environment()
    summary = summarize(out)
    charged = summary["worker_seconds"]
    for job in protocol["jobs"]:
        destination = out / "completed" / f"{job['id']:03d}.json"
        if destination.exists():
            continue
        if (out / "attempts" / f"{job['id']:03d}.json").exists():
            status(out, "needs_inspection", job=job, reason="Admitted worker has no complete result; no uncharged automatic retry")
            return
        if job["phase"] == "target":
            if not (out / "source_fit.json").exists():
                status(out, "fitting_source_routers", completed_attempts=84)
                started = time.monotonic()
                with (out / "source_fit.log").open("w") as stream:
                    subprocess.run([CPU_PY, str(out / "source/fit_feedback_repairs.py"), "--out", str(out)],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                ledger(out, dict(event="source_fit_end", cpu_seconds=time.monotonic() - started,
                                 fit_sha256=digest(out / "source_fit.json")))
            fit = json.loads((out / "source_fit.json").read_text())
            if not fit["source_gate"]["proceed_to_target"]:
                status(out, "source_mechanism_rejected", gate=fit["source_gate"], worker_seconds=charged,
                       explanation="No supported advantage over the flat repair bank; do not spend target/actor budget on this unchanged candidate")
                summarize(out)
                return
        if (out / "STOP_REQUESTED.json").exists():
            status(out, "stopped_by_request")
            return
        if charged >= protocol["max_worker_hours"] * 3600 or summarize(out)["charged_attempts"] + 7 > protocol["max_charged_attempts"]:
            status(out, "budget_reached", worker_seconds=charged)
            return
        while gpu_used_mib(env) >= protocol["admission_below_gib"] * 1024:
            status(out, "waiting_for_gpu", job=job, total_gpu_mib=gpu_used_mib(env))
            if (out / "STOP_REQUESTED.json").exists():
                status(out, "stopped_by_request")
                return
            time.sleep(20)
        repair, profiles = job_config(out, protocol, job)
        run_dir = out / "runs" / f"job{job['id']:03d}_{job['phase']}_{job['condition']}_body{job['body']}_rep{job['repeat']}"
        garment_args, garment_env = garment_cli(protocol["garment"])
        cmd = [PY, str(out / "source/collect_garment.py"), *COMMON, *garment_args,
               "--checkpoint", protocol["checkpoint"], "--bodies", str(job["body"]), "--batch-bodies", "1",
               "--variants", *(["baseline"] * 7), "--profiles-json", str(profiles),
               "--feedback-repairs-json", str(repair), "--policy-device", "cpu", "--isolated-policy-clients",
               "--seed", str(job["seed"]), "--placement-offset-mm", *map(str, OFFSETS_MM[0]),
               "--out", str(run_dir), *job["extra"]]
        log = run_dir.with_suffix(".log")
        started, peak, reason = time.monotonic(), gpu_used_mib(env), None
        with log.open("w") as stream:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, **garment_env), stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            admission = dict(event="worker_start", job=job, pid=proc.pid, log=str(log), command=cmd)
            write_json(out / "attempts" / f"{job['id']:03d}.json", dict(admission, started_unix=time.time()))
            ledger(out, admission)
            print(f"[feedback] {job['phase']} batch={job['id']}/15 body={job['body']} condition={job['condition']} repeat={job['repeat']} pid={proc.pid}", flush=True)
            try:
                while proc.poll() is None:
                    used = gpu_used_mib(env)
                    peak = max(peak, used)
                    status(out, "running_" + job["phase"], job=job, worker_pid=proc.pid, log=str(log),
                           total_gpu_mib=used, worker_seconds_so_far=charged + time.monotonic() - started,
                           completed_attempts=7 * len(list((out / "completed").glob("*.json"))))
                    if used >= protocol["terminate_at_gib"] * 1024:
                        reason = "memory_guard"
                    if charged + time.monotonic() - started >= protocol["max_worker_hours"] * 3600:
                        reason = "worker_budget_reached"
                    if (out / "STOP_REQUESTED.json").exists():
                        reason = "stopped_by_request"
                    if reason:
                        break
                    time.sleep(2)
            finally:
                terminate_group(proc)
                elapsed = time.monotonic() - started
                charged += elapsed
                ledger(out, dict(event="worker_end", job_id=job["id"], pid=proc.pid,
                                 returncode=proc.returncode, reason=reason, worker_seconds=elapsed,
                                 peak_total_gpu_mib=peak))
        if reason or proc.returncode != 0:
            status(out, reason or "worker_error", job=job, log=str(log), returncode=proc.returncode)
            summarize(out)
            return
        rows = []
        for line in log.read_text().splitlines():
            if line.startswith("[result] "):
                rows.append(dict(json.loads(line[9:]), phase=job["phase"], condition=job["condition"],
                                 repeat=job["repeat"], job_id=job["id"], log=str(log), run_dir=str(run_dir)))
        if len(rows) != 7 or any("feedback_repair" not in r for r in rows):
            status(out, "incomplete_batch", job=job, log=str(log))
            summarize(out)
            return
        write_json(destination, dict(job=job, rows=rows, worker_seconds=elapsed))
        summary = summarize(out)
        print(f"[feedback] completed={summary['completed_attempts']} successes={sum(r['accepted'] for r in rows)}/7 charged_hours={charged / 3600:.3f}", flush=True)
    status(out, "complete", **summarize(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--run", action="store_true")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    a.out = a.out.resolve()
    if a.summary:
        print(json.dumps(summarize(a.out), indent=2))
        return
    protocol = prepare(a.out)
    if a.run:
        try:
            execute(a.out, protocol)
        except BaseException as exc:
            status(a.out, "supervisor_error", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
