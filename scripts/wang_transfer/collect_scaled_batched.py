"""Scaled multi-garment collection in batched IPC worlds under CUDA MPS.

Same units, settings and ledger format as ``collect_scaled.py`` (``attempts.jsonl``, ``manifest.json``,
``no_legal_start.jsonl``), and it resumes the same ``--out`` directory, but each collector process runs
``--batch-bodies`` bodies of one garment in a single IPC world and every process joins the user's MPS
daemon. Measured on the RTX PRO 6000: separate 2-slot worlds time-slice the GPU at 2.9 body-decisions/s
in total; three 7-body worlds give 9.9 without MPS and 17.7 with it.

Per job (garment, bodies, placement offset, stage):
* ``policy``: the r1 policy, two replicas per body. Bodies with no legal start move to the next offset.
* bodies with no accepted replica get one ``retry`` job with a new seed at the same offset.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from collect_scaled import COMMON, EVAL_BODIES, GARMENTS, OFFSETS_MM, PY, ROOT, Ledger, garment_args

MPS = {"CUDA_MPS_PIPE_DIRECTORY": "/home/ge47gax/.mps_pipe", "CUDA_MPS_LOG_DIRECTORY": "/home/ge47gax/.mps_log"}


def run_batch(args, garment, bodies, offset_index, stage, seed):
    offset = OFFSETS_MM[offset_index]
    tag = f"{stage}_off{'_'.join(map(str, offset))}_{bodies[0]}x{len(bodies)}"
    out = args.out / garment / "batches" / tag
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(args.out / "collector.py"), *COMMON, *garment_args(garment), "--checkpoint", str(args.checkpoint),
           "--bodies", *map(str, bodies), "--batch-bodies", str(len(bodies)), "--replicas", str(args.replicas),
           "--seed", str(seed), "--placement-offset-mm", *map(str, offset), "--out", str(out)]
    log = out.with_suffix(".log")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", **MPS)
    with log.open("w") as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    results, skipped = [], set()
    for line in log.read_text().splitlines():
        if line.startswith("[result] "):
            results.append(json.loads(line[len("[result] "):]))
        elif line.startswith("[skip] "):
            row = json.loads(line[len("[skip] "):])
            skipped |= set(row.get("bodies", [row["body"]]))
    return results, skipped, log, offset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt")
    p.add_argument("--garments", nargs="+", default=GARMENTS)
    p.add_argument("--regions", type=int, nargs="+", default=list(range(27)))
    p.add_argument("--poses", type=int, nargs="+", default=list(range(40, 50)))
    p.add_argument("--exclude", type=int, nargs="*", default=EVAL_BODIES)
    p.add_argument("--batch", type=int, default=7)
    p.add_argument("--replicas", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=2026092600)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.collector, args.out / "collector.py")
    for helper in args.collector.parent.glob("*.py"):
        if helper.name != args.collector.name:
            shutil.copy(helper, args.out / helper.name)
    done = set()
    for name in ("attempts.jsonl", "no_legal_start.jsonl"):
        if (args.out / name).exists():
            done |= {(json.loads(l)["body"], json.loads(l)["garment"]) for l in (args.out / name).read_text().splitlines() if l.strip()}
    pending = {g: [1000 * (r + 1) + q for q in args.poses for r in args.regions
                   if 1000 * (r + 1) + q not in set(args.exclude) and (1000 * (r + 1) + q, g) not in done]
               for g in args.garments}
    ledger = Ledger(args.out)
    if (args.out / "manifest.json").exists():
        ledger.accepted = json.loads((args.out / "manifest.json").read_text())["accepted"]
    jobs = []                                   # interleave garments, pose-first within each garment
    longest = max(len(v) for v in pending.values())
    for k in range(0, longest, args.batch):
        for g in args.garments:
            chunk = pending[g][k:k + args.batch]
            if chunk:
                jobs.append((g, chunk, 0, "policy"))
    (args.out / "run_batched.json").write_text(json.dumps(dict(vars(args), jobs=len(jobs)), default=str, indent=1) + "\n")
    print(f"[batched] {sum(len(v) for v in pending.values())} units in {len(jobs)} jobs, {args.workers} workers", flush=True)
    lock = threading.Condition()
    queue, active = list(jobs), [0]

    def handle(job):
        garment, bodies, offset_index, stage = job
        seed = args.seed + 100000 * GARMENTS.index(garment) + (1 if stage == "retry" else 0)
        results, skipped, log, offset = run_batch(args, garment, bodies, offset_index, stage, seed)
        follow = []
        by_body = {}
        for row in results:
            ledger.add(row["body"], garment, stage, offset, row, log)
            by_body.setdefault(row["body"], []).append(bool(row.get("accepted")))
        moved = [b for b in bodies if b in skipped and b not in by_body]
        if moved:
            if offset_index + 1 < len(OFFSETS_MM):
                follow.append((garment, moved, offset_index + 1, stage))
            else:
                with ledger.lock, (args.out / "no_legal_start.jsonl").open("a") as f:
                    for b in moved:
                        f.write(json.dumps(dict(body=b, garment=garment, offsets_mm=OFFSETS_MM)) + "\n")
        failed = [b for b, acc in by_body.items() if not any(acc)]
        if failed and stage == "policy":
            follow.append((garment, failed, offset_index, "retry"))
        lost = [b for b in bodies if b not in by_body and b not in skipped]
        acc = sum(any(v) for v in by_body.values())
        print(f"[batched] {garment} {stage} off{offset_index} {bodies}: accepted {acc}/{len(by_body)}, "
              f"no start {len(moved)}, lost {lost}", flush=True)
        return follow

    def worker():
        while True:
            with lock:
                while not queue and active[0]:
                    lock.wait()
                if not queue:
                    lock.notify_all()
                    return
                job = queue.pop(0)
                active[0] += 1
            try:
                follow = handle(job)
            except Exception as exc:                # keep the pool alive; the job's log has the detail
                print(f"[batched] job {job} failed: {exc!r}", flush=True)
                follow = []
            with lock:
                queue[0:0] = follow                 # follow-ups first, so a batch finishes its bodies
                active[0] -= 1
                lock.notify_all()

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[batched] complete", flush=True)


if __name__ == "__main__":
    main()
