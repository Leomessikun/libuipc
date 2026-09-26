"""Finish a running CPU training stage, then audit and queue bounded IPC tests.

Observe only the explicitly identified training process. Never stop another
job, infer GPU-sharing consent, or call an unfinished checkpoint a trained run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from eval_static_flow import ROOT, digest, make_cases
from probe_arm_motion import save_json


def process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if raw[0] == "Z" else raw[19]  # starttime, including parenthesized comm
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--train-pid", type=int, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training-timeout", type=float, default=86400.)
    parser.add_argument("--wait-for-gpu", type=float, default=86400.)
    parser.add_argument("--wall-budget", type=float, default=7200.)
    args = parser.parse_args()
    if min(args.train_pid, args.training_timeout, args.wall_budget) <= 0 or args.wait_for_gpu < 0:
        raise ValueError("Invalid process or time budget")
    train, out = args.train_dir.resolve(), args.out.resolve()
    run = json.loads((train / "run.json").read_text())
    data = Path(run["data_manifest"]).parent
    expected_steps = int(run["arguments"]["steps"])
    identity = process_identity(args.train_pid)
    if identity is None:
        raise ValueError("Training process must still be alive when attaching")
    command = Path(f"/proc/{args.train_pid}/cmdline").read_bytes().split(b"\0")
    if b"uipc_manip.train_flow_bc" not in command or os.fsencode(run["arguments"]["out"]) not in command:
        raise ValueError("PID is not the explicitly selected flow training command")
    out.mkdir(parents=True, exist_ok=False)
    status = dict(state="waiting_for_training", training_pid=args.train_pid, training_start_identity=identity,
                  train_dir=str(train), expected_steps=expected_steps,
                  data_manifest_sha256=digest(data / "manifest.json"), reference_checkpoint=str(args.reference_checkpoint),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    def write_status():
        status["updated_unix_s"] = time.time()
        save_json(out / "status.tmp.json", status)
        (out / "status.tmp.json").replace(out / "status.json")

    def run_command(command, name, timeout):
        with (out / f"{name}.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)

    write_status()
    try:
        deadline = time.monotonic() + args.training_timeout
        while process_identity(args.train_pid) == identity:
            # Ignore a partially written last line while the trainer is active.
            raw = (train / "metrics.jsonl").read_text()
            lines = raw[:raw.rfind("\n") + 1].splitlines()
            if lines:
                status["last_evaluated_step"] = json.loads(lines[-1])["step"]
            write_status()
            if time.monotonic() >= deadline:
                raise TimeoutError("Training wait expired; the training job is left untouched")
            time.sleep(20.)
        import torch

        latest = torch.load(train / "latest.pt", map_location="cpu", weights_only=False)
        if latest["step"] != expected_steps or latest["metadata"]["data_manifest_sha256"] != status["data_manifest_sha256"]:
            raise RuntimeError("Training exited without its expected final checkpoint")
        checkpoint = train / "best.pt"
        cases = make_cases(data, checkpoint, out / "ipc", Path("/home/ge47gax/kun/libuipc"),
                           ["tshirt_26", "tshirt_4", "tshirt_68", "tshirt_392", "hospital_gown"], 750, "random")
        status.update(state="auditing", selected_checkpoint_sha256=digest(checkpoint),
                      last_evaluated_step=expected_steps, cases=cases)
        write_status()
        run_command([sys.executable, "-u", "-m", "uipc_manip.audit_flow_bc", "--data", str(data),
                     "--checkpoints", str(args.reference_checkpoint.resolve()), str(checkpoint),
                     "--out", str(out / "offline_phase_audit.json")], "offline_audit", 3600.)
        status.update(state="ipc_runner", ipc_status=str(out / "ipc/status.json"))
        write_status()
        run_command([sys.executable, "-u", str(ROOT / "scripts/wang_transfer/eval_static_flow.py"),
                     "--data", str(data), "--flow-checkpoint", str(checkpoint), "--out", str(out / "ipc"),
                     "--wait-for-gpu", str(args.wait_for_gpu), "--wall-budget", str(args.wall_budget)],
                    "ipc_runner", args.wait_for_gpu + args.wall_budget + 120.)
        ipc = json.loads((out / "ipc/status.json").read_text())
        if ipc["state"] != "complete":
            raise RuntimeError("IPC runner did not complete all cases")
        status["state"] = "complete"
        write_status()
    except Exception as exc:
        status.update(state="error", error=repr(exc))
        write_status()
        raise


if __name__ == "__main__":
    main()
