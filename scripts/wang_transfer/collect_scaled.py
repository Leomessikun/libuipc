"""Scaled checkpoint collection: fast policy attempts first, IPC-lookahead rescue for hard bodies.

Per body, in a worker pool:

1. Run the policy (default: the DAgger-r1 fine-tune of fmvp_sim) with ``--replicas`` slots in one
   world. If the collector finds no legal gravity-hung start at the placement offset, try the next
   offset in ``OFFSETS_MM`` (model frame, relative to the FMVP-PyBullet gripper position).
2. If no slot is accepted, rerun that body once with the one-decision IPC lookahead
   (``ipc_filter_profile.json``, single slot, from decision 20, loads over 5 N).

Every attempt's collector output directory and log stay under ``--out``; ``attempts.jsonl`` holds
one line per result and ``manifest.json`` the accepted episodes. Evaluation bodies are excluded
(``--exclude``) so they stay clean for later model comparisons. The collector file is copied into
``--out`` first and that copy is what runs, so the run is reproducible while the source evolves.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = "/home/ge47gax/kun/genesis-world/.venv/bin/python"
OFFSETS_MM = [(0, 5, 0), (0, 15, 0), (0, 30, 0), (-20, 5, 0), (20, 5, 0), (0, 5, -20), (0, 5, 20)]
EVAL_BODIES = [14047, 5046, 10047, 2046, 22046, 14058, 9047, 22047, 4047, 14062, 14063, 14064, 14065,
               14066, 14068, 14070, 14073, 14075, 7046, 7047, 8047, 13047, 14061, 14067, 16047, 17047, 18046]
COMMON = ["--hang", "output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz", "--hang-key", "k300",
          "--variants", "baseline", "--steps", "750", "--hold", "20", "--success", ".7",
          "--success-geometry", "physical_sleeve", "--stop-proximal-upper", ".7", "--yaw", "267",
          "--rotation", "fmvp", "--collision-geometry", "full_body", "--abort-gripper-force", "1000",
          "--cloth-density", "750", "--cloth-strain-rate", "10"]


def run_collector(args, body, offset, seed, tag, extra):
    out = args.out / f"body_{body}" / tag
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(args.out / "collector.py"), *COMMON, "--checkpoint", str(args.checkpoint), "--bodies", str(body),
           "--seed", str(seed), "--placement-offset-mm", *map(str, offset), "--out", str(out), *extra]
    log = out.with_suffix(".log")
    env = dict(__import__("os").environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with log.open("w") as stream:
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env, check=False)
    results, skipped = [], False
    for line in log.read_text().splitlines():
        if line.startswith("[result] "):
            results.append(json.loads(line[len("[result] "):]))
        elif line.startswith("[skip] "):
            skipped = True
    return results, skipped, log


class Ledger:
    def __init__(self, out: Path):
        self.out, self.lock = out, threading.Lock()
        self.accepted = []

    def add(self, body, tag, offset, row, log):
        entry = dict(body=body, stage=tag, offset_mm=list(offset), log=str(log), **{
            k: row.get(k) for k in ("replica", "accepted", "success_state", "transitions", "gripper_p90_N",
                                    "gripper_peak_N", "sim_error", "path", "checkpoint_sha256", "seed")})
        with self.lock:
            with (self.out / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(entry) + "\n")
            if row.get("accepted"):
                self.accepted.append(entry)
                (self.out / "manifest.json").write_text(json.dumps(dict(accepted=self.accepted), indent=1) + "\n")


def collect_body(args, ledger, body):
    seed = args.seed + body
    for offset in OFFSETS_MM:
        results, skipped, log = run_collector(args, body, offset, seed, f"policy_off{'_'.join(map(str, offset))}",
                                              ["--replicas", str(args.replicas)])
        if skipped and not results:
            continue
        for row in results:
            ledger.add(body, "policy", offset, row, log)
        if any(r.get("accepted") for r in results):
            return f"{body}: policy {sum(bool(r.get('accepted')) for r in results)}/{len(results)}"
        if args.no_rescue:
            return f"{body}: policy 0/{len(results)}"
        results, _, log = run_collector(args, body, offset, seed + 1, "rescue",
                                        ["--profiles-json", "scripts/wang_transfer/ipc_filter_profile.json",
                                         "--lookahead-from", "20", "--lookahead-min-load", "5"])
        for row in results:
            ledger.add(body, "rescue", offset, row, log)
        return f"{body}: policy 0, rescue {'accepted' if any(r.get('accepted') for r in results) else 'failed'}"
    with ledger.lock:
        with (args.out / "no_legal_start.jsonl").open("a") as f:
            f.write(json.dumps(dict(body=body, offsets_mm=OFFSETS_MM)) + "\n")
    return f"{body}: no legal start at any offset"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True, help="Collector with --lookahead-from support.")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt")
    p.add_argument("--regions", type=int, nargs="+", default=list(range(27)))
    p.add_argument("--poses", type=int, nargs="+", default=list(range(40, 50)))
    p.add_argument("--exclude", type=int, nargs="*", default=EVAL_BODIES)
    p.add_argument("--replicas", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026092600)
    p.add_argument("--no-rescue", action="store_true")
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.collector, args.out / "collector.py")
    for helper in args.collector.parent.glob("*.py"):
        if helper.name != args.collector.name and not (args.out / helper.name).exists():
            shutil.copy(helper, args.out / helper.name)
    bodies = [1000 * (r + 1) + q for q in args.poses for r in args.regions]
    bodies = [b for b in bodies if b not in set(args.exclude)]
    done = set()
    if (args.out / "attempts.jsonl").exists():
        done = {json.loads(l)["body"] for l in (args.out / "attempts.jsonl").read_text().splitlines() if l.strip()}
    bodies = [b for b in bodies if b not in done]
    (args.out / "run.json").write_text(json.dumps(dict(vars(args), bodies=bodies, offsets_mm=OFFSETS_MM,
                                                       common=COMMON), default=str, indent=1) + "\n")
    ledger = Ledger(args.out)
    print(f"[scaled] {len(bodies)} bodies, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for message in pool.map(lambda b: collect_body(args, ledger, b), bodies):
            print(f"[scaled] {message}", flush=True)


if __name__ == "__main__":
    main()
