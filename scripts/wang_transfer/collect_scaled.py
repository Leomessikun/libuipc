"""Scaled checkpoint collection over garments and bodies: fast policy attempts, then a rescue.

Work units are (body, garment) pairs, ordered pose-first so a partial run stays balanced over
garments and pose regions. Per unit, in a worker pool:

1. Run the policy (default: the DAgger-r1 fine-tune of fmvp_sim) with ``--replicas`` slots in one
   world. If the collector finds no legal gravity-hung start at the placement offset, try the next
   offset in ``OFFSETS_MM`` (model frame, relative to the FMVP-PyBullet gripper position).
2. If no slot is accepted, rerun once with the one-decision IPC lookahead (``ipc_filter_profile.json``,
   single slot, from decision 20, loads over 5 N).

Every garment is accepted by the physical-sleeve test on its own cloth3d mesh (armhole seam at 0.7 of
the upper arm, sleeve wrapped); the collector must read ``<garment>.obj`` for the sections.

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
GARMENTS = ["tshirt_26", "tshirt_68", "tshirt_4", "tshirt_392", "hospital_gown"]
HANGS = {"tshirt_26": "output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz",
         **{g: f"output/uipc_manip/garment_demos_20260925/hang_{g}.npz" for g in GARMENTS[1:]}}
COMMON = ["--hang-key", "k300", "--variants", "baseline", "--steps", "750", "--hold", "20", "--success", ".7",
          "--yaw", "267", "--rotation", "fmvp", "--collision-geometry", "full_body", "--abort-gripper-force", "1000",
          "--cloth-density", "750", "--cloth-strain-rate", "10"]


def garment_args(garment):
    # Every garment is judged by the physical-sleeve test on its own cloth3d mesh; the legacy
    # upper-arm ratio disagrees with it in both directions on the non-tshirt_26 garments.
    # The endpoint is the armhole seam at 0.7 of the upper arm with the sleeve wrapped; the proximal
    # section it replaced sits too low on long sleeves to reach 0.7 even when they are fully on.
    # Bodies pass Wang's 18 cm fit filter, and each garment is sized to the body so the sleeve is 1.2x
    # the upper arm: in IPC a sleeve narrower than the arm cannot go on.
    return ["--garment", garment, "--hang", HANGS[garment], "--success-geometry", "physical_sleeve",
            "--stop-proximal-upper", ".7", "--armhole-endpoint", "--body-fit-filter", "0.18",
            "--fit-sleeve-ratio", "1.2", "--sections-wrap", "--align-armhole-axis"]


def run_collector(args, body, garment, offset, seed, tag, extra):
    out = args.out / garment / f"body_{body}" / tag
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(args.out / "collector.py"), *COMMON, *garment_args(garment), "--checkpoint", str(args.checkpoint),
           "--bodies", str(body),
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

    def add(self, body, garment, tag, offset, row, log):
        entry = dict(body=body, garment=garment, stage=tag, offset_mm=list(offset), log=str(log), **{
            k: row.get(k) for k in ("replica", "accepted", "success_state", "transitions", "gripper_p90_N",
                                    "gripper_peak_N", "sim_error", "path", "checkpoint_sha256", "seed")})
        with self.lock:
            with (self.out / "attempts.jsonl").open("a") as f:
                f.write(json.dumps(entry) + "\n")
            if row.get("accepted"):
                self.accepted.append(entry)
                (self.out / "manifest.json").write_text(json.dumps(dict(accepted=self.accepted), indent=1) + "\n")


def collect_unit(args, ledger, unit):
    body, garment = unit
    seed = args.seed + body + 100000 * GARMENTS.index(garment)
    for offset in OFFSETS_MM:
        results, skipped, log = run_collector(args, body, garment, offset, seed, f"policy_off{'_'.join(map(str, offset))}",
                                              ["--replicas", str(args.replicas)])
        if skipped and not results:
            continue
        for row in results:
            ledger.add(body, garment, "policy", offset, row, log)
        if any(r.get("accepted") for r in results):
            return f"{garment} {body}: policy {sum(bool(r.get('accepted')) for r in results)}/{len(results)}"
        if args.no_rescue:
            return f"{garment} {body}: policy 0/{len(results)}"
        extra, stage = ["--profiles-json", "scripts/wang_transfer/ipc_filter_profile.json",
                        "--lookahead-from", "20", "--lookahead-min-load", "5"], "rescue"
        results, _, log = run_collector(args, body, garment, offset, seed + 1, stage, extra)
        for row in results:
            ledger.add(body, garment, stage, offset, row, log)
        return f"{garment} {body}: policy 0, {stage} {sum(bool(r.get('accepted')) for r in results)}/{len(results)}"
    with ledger.lock:
        with (args.out / "no_legal_start.jsonl").open("a") as f:
            f.write(json.dumps(dict(body=body, garment=garment, offsets_mm=OFFSETS_MM)) + "\n")
    return f"{garment} {body}: no legal start at any offset"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collector", type=Path, required=True, help="Collector with --garment and --lookahead-from support.")
    p.add_argument("--checkpoint", type=Path,
                   default=ROOT / "output/uipc_manip/fmvp_ipc_dagger_r1_20260924/model/fmvp_ipc_bc.pt")
    p.add_argument("--garments", nargs="+", default=GARMENTS)
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
    units = [(1000 * (r + 1) + q, g) for q in args.poses for r in args.regions for g in args.garments]
    units = [u for u in units if u[0] not in set(args.exclude)]
    done = set()
    for name in ("attempts.jsonl", "no_legal_start.jsonl"):
        if (args.out / name).exists():
            done |= {(json.loads(l)["body"], json.loads(l)["garment"]) for l in (args.out / name).read_text().splitlines() if l.strip()}
    units = [u for u in units if u not in done]
    (args.out / "run.json").write_text(json.dumps(dict(vars(args), units=units, offsets_mm=OFFSETS_MM,
                                                       common=COMMON, hangs=HANGS), default=str, indent=1) + "\n")
    ledger = Ledger(args.out)
    print(f"[scaled] {len(units)} (body, garment) units, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as pool:
        for message in pool.map(lambda u: collect_unit(args, ledger, u), units):
            print(f"[scaled] {message}", flush=True)


if __name__ == "__main__":
    main()
