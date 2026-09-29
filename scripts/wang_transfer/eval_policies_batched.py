"""Paired closed-loop evaluation of dressing policies on the dataset index's held-out bodies.

Every (garment, test body) unit is attempted by every policy from the same start: the collection settings of
``collect_scaled_batched`` (armhole-aligned start, body-fit filter, physical-sleeve success), the same seed
and placement offset, one replica, batched IPC worlds under MPS. Units are all test bodies, not only those
with accepted demonstrations, so the success rate is not selected on the teacher's successes. A body with
no legal start at the first offset moves to the second; placement is deterministic, so all policies see the
same skip set. Results go to ``results.jsonl``; ``--summary`` prints the paired table.

Policies are ``name=checkpoint`` pairs. A checkpoint holding ``flow_policy`` runs with native actions and
the gripper load as its force input (the observation it was trained on); others run as in collection.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from collect_scaled import COMMON, GARMENTS, OFFSETS_MM, PY, ROOT, garment_args
from collect_scaled_batched import MPS

FLOW_PROFILE = [dict(name="flow_gripper", force_source="gripper", force_scale=1., force_clip=1e6, force_ema=1.)]


def is_flow(checkpoint):
    import zipfile
    try:
        with zipfile.ZipFile(checkpoint) as z:     # torch zip checkpoint: pickled dict keys appear in data.pkl
            return any(n.endswith("data.pkl") and (b"flow_policy" in z.read(n) or b"flow_e2e" in z.read(n))
                       for n in z.namelist())
    except zipfile.BadZipFile:
        return False


CN_ASSETS = ROOT / "output/uipc_manip/clothesnet_assets"


def garment_cli(garment):
    """Collector arguments and environment for one garment: Wang's five, or a prepared ClothesNet asset."""
    if not garment.startswith("cn_"):
        return garment_args(garment), {}
    args = garment_args("tshirt_26")
    i = args.index("--hang")
    args[args.index("--garment") + 1] = garment
    args[i + 1] = str(CN_ASSETS / "hang" / f"hang_{garment}.npz")
    env = dict(UIPC_MANIP_GARMENT_INDEX_MODULE=str(ROOT / "scripts/clothesnet/garment_idx_clothesnet.py"),
               UIPC_MANIP_RAW_GARMENT_DIR=str(CN_ASSETS / "raw"), CLOTHESNET_ASSETS=str(CN_ASSETS))
    return args + ["--sleeve-template", str(CN_ASSETS / "raw" / f"{garment}.obj")], env


def run_batch(args, policy, checkpoint, garment, bodies, offset_index):
    offset = OFFSETS_MM[offset_index]
    out = args.out / policy / garment / f"off{offset_index}_{bodies[0]}x{len(bodies)}"
    out.parent.mkdir(parents=True, exist_ok=True)
    garment_cmd, garment_env = garment_cli(garment)
    cmd = [PY, str(args.out / "collector.py"), *COMMON, *garment_cmd, "--checkpoint", str(checkpoint),
           "--bodies", *map(str, bodies), "--batch-bodies", str(len(bodies)), "--replicas", "1",
           "--seed", str(args.seed), "--placement-offset-mm", *map(str, offset), "--out", str(out)]
    if is_flow(checkpoint):
        cmd += ["--native-policy-actions", "--profiles-json", str(args.out / "flow_profile.json")]
    cmd += shlex.split(args.collector_args)
    log = out.with_suffix(".log")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1", **MPS, **garment_env)
    with log.open("w") as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    results, skipped = [], set()
    for line in log.read_text().splitlines():
        if line.startswith("[result] "):
            results.append(json.loads(line[len("[result] "):]))
        elif line.startswith("[skip] "):
            row = json.loads(line[len("[skip] "):])
            skipped |= set(row.get("bodies", [row["body"]]))
    return results, skipped, log


def summary(out):
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("no_legal_start")]
    res = {(r["policy"], r["garment"], r["body"]): r for r in rows}
    policies = sorted({r["policy"] for r in rows})
    print(f"{'garment':14s}" + "".join(f"{p:>22s}" for p in policies) + "   paired units")
    total = collections.Counter()
    for g in GARMENTS + sorted({r["garment"] for r in rows} - set(GARMENTS)):
        units = sorted({(r["garment"], r["body"]) for r in rows if r["garment"] == g})
        if not units:
            continue
        both = [u for u in units if all((p, *u) in res for p in policies)]
        cells = []
        for p in policies:
            ok = sum(bool(res[(p, *u)].get("accepted")) for u in both)
            grasp = sum("grasp" in str(res[(p, *u)].get("sim_error") or "") for u in both)
            total[p] += ok
            cells.append(f"{ok:3d}/{len(both):<3d} (grasp lost {grasp:2d})")
        total["n"] += len(both)
        print(f"{g:14s}" + "".join(f"{c:>22s}" for c in cells) + f"   {len(both)}")
    print(f"{'all':14s}" + "".join(f"{total[p]:>15d}/{total['n']:<6d}" for p in policies))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--index", type=Path, default=ROOT / "output/uipc_manip/dressing_dataset_20260927/index.json")
    p.add_argument("--split", default="test")
    p.add_argument("--policies", nargs="+", default=[], help="name=checkpoint")
    p.add_argument("--collector", type=Path, default=ROOT / "scripts/wang_transfer/collect_garment.py")
    p.add_argument("--garments", nargs="+", default=GARMENTS)
    p.add_argument("--batch", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026092700)
    p.add_argument("--summary", action="store_true")
    p.add_argument("--max-bodies", type=int, help="Evaluate only the first N bodies of the split.")
    p.add_argument("--collector-args", default="",
                   help="Extra collector options for every job, one quoted string (e.g. '--align-target cuff').")
    args = p.parse_args()
    args.out = args.out.resolve()
    if args.summary:
        return summary(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.collector, args.out / "collector.py")
    for helper in args.collector.parent.glob("*.py"):
        if helper.name != args.collector.name:
            shutil.copy(helper, args.out / helper.name)
    (args.out / "flow_profile.json").write_text(json.dumps(FLOW_PROFILE))
    policies = dict(s.split("=", 1) for s in args.policies)
    index = json.loads(args.index.read_text())["episodes"]
    bodies = sorted({e["body"] for e in index if e["split"] == args.split})[:args.max_bodies]
    done = set()
    if (args.out / "results.jsonl").exists():
        done = {(r["policy"], r["garment"], r["body"]) for r in map(json.loads, (args.out / "results.jsonl").read_text().splitlines())}
    jobs = []
    for k in range(0, len(bodies), args.batch):
        for g in args.garments:
            for name in policies:
                chunk = [b for b in bodies[k:k + args.batch] if (name, g, b) not in done]
                if chunk:
                    jobs.append((name, g, chunk, 0))
    (args.out / "run.json").write_text(json.dumps(dict(vars(args), policies=policies, bodies=bodies, jobs=len(jobs)),
                                                  default=str, indent=1) + "\n")
    print(f"[eval] {len(bodies)} {args.split} bodies x {len(args.garments)} garments x {len(policies)} policies: "
          f"{len(jobs)} jobs, {args.workers} workers", flush=True)
    lock = threading.Condition()
    queue, active = list(jobs), [0]

    def handle(job):
        name, garment, chunk, offset_index = job
        results, skipped, log = run_batch(args, name, policies[name], garment, chunk, offset_index)
        seen = set()
        with lock, (args.out / "results.jsonl").open("a") as f:
            for r in results:
                seen.add(r["body"])
                row = {k: v for k, v in r.items() if k != "profile"}
                f.write(json.dumps(dict(row, policy=name, garment=garment, offset_mm=OFFSETS_MM[offset_index],
                                        log=str(log))) + "\n")
            moved = [b for b in chunk if b in skipped and b not in seen]
            if moved and offset_index == 0:
                queue.insert(0, (name, garment, moved, 1))
            elif moved:
                for b in moved:
                    f.write(json.dumps(dict(policy=name, garment=garment, body=b, no_legal_start=True)) + "\n")
        acc = sum(bool(r.get("accepted")) for r in results)
        print(f"[eval] {name} {garment} off{offset_index} {chunk}: accepted {acc}/{len(results)}, "
              f"no start {len(moved)}", flush=True)

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
                handle(job)
            except Exception as exc:
                print(f"[eval] job {job} failed: {exc!r}", flush=True)
            with lock:
                active[0] -= 1
                lock.notify_all()

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("[eval] complete", flush=True)
    summary(args.out)


if __name__ == "__main__":
    main()
