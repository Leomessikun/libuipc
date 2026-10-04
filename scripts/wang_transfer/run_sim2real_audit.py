"""Fixed paired robustness screen; one guarded IPC worker, no training/old queues.

Prepare: python3 scripts/wang_transfer/run_sim2real_audit.py
Execute/resume: same command --run
Read current evidence: same command --summary
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

from collect_scaled import COMMON, OFFSETS_MM, PY, ROOT
from eval_policies_batched import FLOW_PROFILE, garment_cli, is_flow

DEFAULT_OUT = ROOT / "output/uipc_manip/sim2real_audit_20261005"
PACKAGE = ROOT / ".claude/worktrees/residual-rl/python"
NVML = Path("/tmp/claude-4102472/-home-ge47gax-kun-FoE/c79830ba-6713-4557-9c2b-89be28f42331/scratchpad/nvml/root/usr/lib/x86_64-linux-gnu")
POLICIES = {
    "fmvp_sim": "/home/ge47gax/Desktop/fmvp_sim.pt",
    "r1": str(ROOT / "output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt"),
    "flow": str(ROOT / "output/uipc_manip/dressing_dataset_20260927/flow_r1_h16/flow_policy_eval.pt"),
}
CONDITIONS = [
    ("nominal", []),
    ("friction_x2", ["--friction", ".6"]),
    ("youngs_x2", ["--cloth-youngs", "12000"]),
    ("bending_x2", ["--cloth-bending-stiffness", ".2"]),
    ("density_x1p5", ["--cloth-density", "1125"]),
    ("thickness_x1p5", ["--cloth-thickness", ".000225"]),
    ("point_noise_3mm", ["--observation-noise-m", ".003"]),
    ("point_dropout_30pct", ["--observation-dropout", ".3"]),
    ("voxel_origin_half_cell", ["--observation-voxel-shift-m", ".03125", ".03125", ".03125"]),
    ("nominal_repeat", []),
]


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ledger(out, row):
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps(dict(time=time.time(), **row)) + "\n")


def status(out, stage, **kwargs):
    write_json(out / "status.json", dict(stage=stage, supervisor_pid=os.getpid(), time=time.time(), **kwargs))


def prepare(out):
    path = out / "protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / "output/uipc_manip/dressing_dataset_20260927/index.json"
    index = json.loads(index_path.read_text())["episodes"]
    dev = sorted({int(e["body"]) for e in index if e["split"] == "test"})[:14]
    # A bounded screen: the first seven dev bodies and the original primary garment.
    # This choice does not depend on policy successes or perturbation outcomes.
    bodies, garment = dev[:7], "tshirt_26"
    source = Path(__file__).parent
    snap = out / "source"
    snap.mkdir(exist_ok=True)
    for file in source.glob("*.py"):
        shutil.copyfile(file, snap / file.name)
    write_json(out / "flow_profile.json", FLOW_PROFILE)
    hashes = {str(snap / name): digest(snap / name) for name in
              ("collect_garment.py", "sim2real_perturbations.py", "run_sim2real_audit.py", "rollout_controls.py")}
    for file in ("dressing_env.py", "wang_bridge.py", "wang_client.py", "flow_policy.py", "obs.py"):
        p = PACKAGE / "uipc_manip" / file
        hashes[str(p)] = digest(p)
    jobs = []
    names = list(POLICIES)
    for k, (condition, extra) in enumerate(CONDITIONS):
        # Rotate policy order to reduce a consistent time/order advantage.
        order = names[k % 3:] + names[:k % 3]
        for name in order:
            jobs.append(dict(id=len(jobs), condition=condition, policy=name, garment=garment,
                             bodies=bodies, extra=extra, seed=2026100500, offset_mm=list(OFFSETS_MM[0])))
    protocol = dict(
        version=1, created_unix=time.time(), brief_commit="67e0c5fa",
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        policies=POLICIES, checkpoint_sha256={k: digest(v) for k, v in POLICIES.items()},
        index_sha256=digest(index_path), source_sha256=hashes,
        development_bodies=dev, measured_bodies=bodies, garments=[garment], jobs=jobs,
        units_per_condition=len(bodies), planned_episodes=len(jobs) * len(bodies),
        seed_note="Same environment seed and fresh per-slot flow generator (seed 0) across conditions; nominal_repeat is an identical rerun, not new cases.",
        nominal_material=dict(friction=.3, cloth_youngs=6000., cloth_bending_stiffness=.1,
                              cloth_density=750., cloth_thickness=.00015, cloth_strain_rate=10.),
        nominal_bridge_voxel_m=.0625, max_worker_hours=24., memory_limit_gib=90.,
        admission_reserve_gib=6., max_concurrent_sim_workers=1,
        interpretation="Seven-body single-garment sensitivity screen, not a full garment benchmark or real-transfer result.",
        successful_criteria="Unchanged physical sleeve, armhole endpoint .7, valid grasp and 20-decision hold.",
        geometry_pairing="Fixed offset; no success-dependent restart/repositioning; require matching initial geometry hashes.",
        costs="30 sequential batches; historical 7-body median setup+rollout about 36 minutes, roughly 18 hours. Current concurrency differs; measure actual cost.",
    )
    write_json(path, protocol)
    status(out, "prepared", planned_batches=len(jobs), planned_episodes=protocol["planned_episodes"])
    return protocol


def environment():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1", CUDA_MPS_PIPE_DIRECTORY="/home/ge47gax/.mps_pipe",
               CUDA_MPS_LOG_DIRECTORY="/home/ge47gax/.mps_log")
    if NVML.exists():
        env["LD_LIBRARY_PATH"] = str(NVML) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def gpu_used_mib(env):
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     env=env, text=True, timeout=10).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError("This audit expects one GPU; select explicitly before changing this protocol")
    return float(output[0])


def exact_mcnemar(wins, losses):
    n = wins + losses
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.


def mean_ci(values):
    if not values:
        return dict(mean=None, bootstrap95=None)
    rng = random.Random(20261005)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return dict(mean=sum(values) / n, bootstrap95=[means[249], means[9749]])


def summarize(out, protocol):
    rows = []
    for path in sorted((out / "completed").glob("*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    results = {(r["condition"], r["policy"], r["body"]): r for r in rows if not r.get("no_legal_start")}
    report = dict(planned_episodes=protocol["planned_episodes"], recorded_rows=len(rows),
                  completed_batches=len(list((out / "completed").glob("*.json"))), conditions={}, repeatability={})
    for condition, _ in CONDITIONS:
        units = [b for b in protocol["measured_bodies"] if all((condition, p, b) in results for p in POLICIES)]
        mismatch = [b for b in units if len({results[condition, p, b]["initial_geometry_sha256"] for p in POLICIES}) != 1]
        units = [b for b in units if b not in mismatch]
        entry = dict(paired_units=len(units), initialization_mismatches=mismatch,
                     successes={p: sum(bool(results[condition, p, b]["accepted"]) for b in units) for p in POLICIES}, comparisons={})
        for p in ("r1", "flow"):
            diff = [int(results[condition, p, b]["accepted"]) - int(results[condition, "fmvp_sim", b]["accepted"]) for b in units]
            wins, losses = diff.count(1), diff.count(-1)
            paired = [b for b in units if all(("nominal", q, b) in results for q in POLICIES)
                      and all(results[condition, q, b]["initial_geometry_sha256"] == results["nominal", q, b]["initial_geometry_sha256"] for q in POLICIES)]
            interaction = [(int(results[condition, p, b]["accepted"]) - int(results[condition, "fmvp_sim", b]["accepted"]))
                           - (int(results["nominal", p, b]["accepted"]) - int(results["nominal", "fmvp_sim", b]["accepted"])) for b in paired]
            entry["comparisons"][p] = dict(wins=wins, losses=losses, exact_mcnemar_p=exact_mcnemar(wins, losses),
                                           gain=mean_ci(diff), gain_change_from_nominal=mean_ci(interaction),
                                           interaction_units=len(paired))
        report["conditions"][condition] = entry
    for p in POLICIES:
        units = [b for b in protocol["measured_bodies"] if all((c, p, b) in results for c in ("nominal", "nominal_repeat"))]
        agreement = [int(results["nominal", p, b]["accepted"] == results["nominal_repeat", p, b]["accepted"]) for b in units]
        report["repeatability"][p] = dict(units=len(units), agreement=mean_ci(agreement))
    events_path = out / "events.jsonl"
    events = [json.loads(l) for l in events_path.read_text().splitlines()] if events_path.exists() else []
    ends = [e for e in events if e["event"] == "worker_end"]
    report["cost"] = dict(worker_seconds=sum(e["worker_seconds"] for e in ends), attempts=len(ends),
                          peak_total_gpu_mib=max((e["peak_total_gpu_mib"] for e in ends), default=None))
    report["complete"] = report["completed_batches"] == len(protocol["jobs"])
    report["limits"] = ["Pointwise exploratory p-values; no multiplicity-adjusted discovery claim.",
                         "Percentile bootstrap over seven bodies is a coarse uncertainty summary.",
                         "Same-simulator parameter stress is not calibrated target physics or real-robot evidence."]
    write_json(out / "summary.json", report)
    return report


def terminate_group(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run(out, protocol):
    lock = (out / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = environment()
    (out / "completed").mkdir(exist_ok=True)
    for name, expected in protocol["checkpoint_sha256"].items():
        if digest(protocol["policies"][name]) != expected:
            raise RuntimeError(f"Frozen checkpoint changed: {name}")
    for path, expected in protocol["source_sha256"].items():
        if digest(path) != expected:
            raise RuntimeError(f"Frozen dependency changed: {path}")
    cap = protocol["memory_limit_gib"] * 1024
    reserve = protocol["admission_reserve_gib"] * 1024
    charged = summarize(out, protocol)["cost"]["worker_seconds"]
    for job in protocol["jobs"]:
        result_path = out / "completed" / f"{job['id']:03d}.json"
        if result_path.exists():
            continue
        attempts = sum(p.is_dir() for p in (out / "runs").glob(f"job{job['id']:03d}_*"))
        if attempts >= 3:
            status(out, "needs_inspection", job=job, reason="Three prior attempts; no automatic repeat")
            return
        while True:
            if (out / "STOP_REQUESTED.json").exists():
                status(out, "stopped_by_request")
                return
            used = gpu_used_mib(env)
            if used + reserve < cap:
                break
            status(out, "waiting_for_gpu", next_job=job, total_gpu_mib=used,
                   admission_below_mib=cap - reserve, completed_batches=len(list((out / "completed").glob("*.json"))))
            time.sleep(30)
        if charged >= protocol["max_worker_hours"] * 3600:
            status(out, "worker_budget_reached", worker_seconds=charged)
            return
        run_dir = out / "runs" / f"job{job['id']:03d}_{job['condition']}_{job['policy']}_attempt{attempts}"
        run_dir.parent.mkdir(exist_ok=True)
        garment_args, garment_env = garment_cli(job["garment"])
        cmd = [PY, str(out / "source/collect_garment.py"), *COMMON, *garment_args,
               "--checkpoint", protocol["policies"][job["policy"]], "--bodies", *map(str, job["bodies"]),
               "--batch-bodies", str(len(job["bodies"])), "--replicas", "1", "--policy-device", "cpu",
               "--isolated-policy-clients", "--seed", str(job["seed"]),
               "--placement-offset-mm", *map(str, job["offset_mm"]), "--out", str(run_dir), *job["extra"]]
        if is_flow(protocol["policies"][job["policy"]]):
            cmd += ["--native-policy-actions", "--profiles-json", str(out / "flow_profile.json")]
        log = run_dir.with_suffix(".log")
        started, peak, reason = time.monotonic(), used, None
        with log.open("w") as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    env=dict(env, **garment_env), start_new_session=True)
            ledger(out, dict(event="worker_start", job=job, pid=proc.pid, command=cmd, log=str(log)))
            print(f"[audit] job {job['id']} {job['condition']} {job['policy']} pid={proc.pid}", flush=True)
            try:
                while proc.poll() is None:
                    used = gpu_used_mib(env)
                    peak = max(peak, used)
                    status(out, "running", job=job, worker_pid=proc.pid, total_gpu_mib=used,
                           log=str(log), worker_seconds_so_far=charged + time.monotonic() - started)
                    if used >= cap - 256:
                        reason = "memory_guard"
                    if charged + time.monotonic() - started >= protocol["max_worker_hours"] * 3600:
                        reason = "worker_budget_reached"
                    if (out / "STOP_REQUESTED.json").exists():
                        reason = "stopped_by_request"
                    if reason:
                        terminate_group(proc)
                        break
                    time.sleep(2)
            finally:
                terminate_group(proc)
        elapsed = time.monotonic() - started
        charged += elapsed
        ledger(out, dict(event="worker_end", job_id=job["id"], pid=proc.pid, returncode=proc.returncode,
                         reason=reason, worker_seconds=elapsed, peak_total_gpu_mib=peak))
        if reason or proc.returncode != 0:
            status(out, reason or "worker_error", job=job, log=str(log), returncode=proc.returncode)
            summarize(out, protocol)
            return
        rows, skipped = [], set()
        for line in log.read_text().splitlines():
            if line.startswith("[result] "):
                rows.append(dict(json.loads(line[9:]), condition=job["condition"], policy=job["policy"],
                                 garment=job["garment"], job_id=job["id"], log=str(log)))
            elif line.startswith("[skip] "):
                row = json.loads(line[7:])
                skipped.update(row.get("bodies", [row["body"]]))
        seen = {r["body"] for r in rows}
        for b in skipped - seen:
            rows.append(dict(body=b, no_legal_start=True, condition=job["condition"], policy=job["policy"],
                             garment=job["garment"], job_id=job["id"], log=str(log)))
        if {r["body"] for r in rows} != set(job["bodies"]) or len(rows) != len(job["bodies"]):
            status(out, "incomplete_job", job=job, log=str(log))
            return
        write_json(result_path, dict(job=job, rows=rows, worker_seconds=elapsed))
        summarize(out, protocol)
    status(out, "complete", summary=str(out / "summary.json"), worker_seconds=charged)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--run", action="store_true")
    p.add_argument("--summary", action="store_true")
    a = p.parse_args()
    a.out = a.out.resolve()
    protocol = prepare(a.out)
    if a.run:
        run(a.out, protocol)
    elif a.summary:
        print(json.dumps(summarize(a.out, protocol), indent=2))
    else:
        print(json.dumps({k: protocol[k] for k in ("measured_bodies", "garments", "planned_episodes", "costs")}, indent=2))


if __name__ == "__main__":
    main()
