"""Temporarily stop only a collection dispatcher while queued IPC tests run.

Existing collector children finish and write their own files. SIGCONT resumes
the dispatcher, which then records completed jobs and launches its next batch.
On failure, stop only our evaluation subtree before restoring collection.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


def process(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if raw[0] == "Z" else (raw[0], raw[19])
    except FileNotFoundError:
        return None


def children(pid):
    result = set()
    try:
        tasks = list(Path(f"/proc/{pid}/task").iterdir())
    except FileNotFoundError:
        return result
    for task in tasks:
        try:
            result.update(map(int, (task / "children").read_text().split()))
        except FileNotFoundError:
            pass
    return result


def cancel_evaluation(pid, identity):
    info = process(pid)
    if info is None or info[1] != identity:
        return
    owned, pending = {}, [pid]
    while pending:
        current = pending.pop()
        info = process(current)
        if info is None or current in owned:
            continue
        owned[current] = info[1]
        pending.extend(children(current))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for current, start in reversed(list(owned.items())):
            info = process(current)
            if info is not None and info[1] == start:
                try:
                    os.kill(current, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            time.sleep(.5)


def lease(collector_pid, evaluation_pid, ipc_status, output, *, collection_dir=None,
          max_wait=3600., max_run=7500., poll=5.):
    if collector_pid == evaluation_pid or min(max_wait, max_run, poll) <= 0:
        raise ValueError("Need distinct processes and positive wait/run budgets")
    collector, evaluation = process(collector_pid), process(evaluation_pid)
    if collector is None or evaluation is None or collector[0] in ("T", "t"):
        raise ValueError("Need a running dispatcher and evaluator; do not take over an existing pause")
    for pid, expected in ((collector_pid, "collect_scaled_batched.py"), (evaluation_pid, "eval_static_flow.py")):
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if not any(Path(os.fsdecode(arg)).name == expected for arg in command if arg):
            raise ValueError(f"PID {pid} is not the expected {expected} process")
    ipc_status, output = Path(ipc_status), Path(output)
    status = json.loads(ipc_status.read_text())
    if status["state"] != "waiting_for_gpu":
        raise ValueError("Attach only to an evaluator already waiting for the GPU")
    record = dict(state="preparing", collector_pid=collector_pid, collector_start=collector[1],
                  evaluation_pid=evaluation_pid, evaluation_start=evaluation[1],
                  ipc_status=str(ipc_status.resolve()), max_wait_s=max_wait, max_run_s=max_run,
                  scope="SIGSTOP dispatcher only; existing collection children are never signaled")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(record, handle)

    def write():
        record["updated_unix_s"] = time.time()
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output)

    paused, run_started, began = False, None, time.monotonic()
    try:
        os.kill(collector_pid, signal.SIGSTOP)
        paused = True
        stop_deadline = time.monotonic() + 2.
        while process(collector_pid) is not None and process(collector_pid)[0] not in ("T", "t"):
            if time.monotonic() >= stop_deadline:
                raise TimeoutError("Dispatcher did not stop")
            time.sleep(.01)
        if collection_dir is not None:
            # The existing dispatcher rewrites its manifest directly. If it
            # stopped during that write, restore it rather than freeze a
            # partially written manifest for the evaluation window.
            collection_dir = Path(collection_dir).resolve()
            command = Path(f"/proc/{collector_pid}/cmdline").read_bytes().split(b"\0")
            source_out = Path(os.fsdecode(command[command.index(b"--out") + 1]))
            actual = (Path(f"/proc/{collector_pid}/cwd").resolve() / source_out).resolve()
            if actual != collection_dir:
                raise ValueError("Dispatcher output is not the selected collection directory")
            manifest = json.loads((collection_dir / "manifest.json").read_text())
            record.update(collection_dir=str(collection_dir), accepted_at_pause=len(manifest["accepted"]))
        record.update(state="draining_collection", collection_children=sorted(children(collector_pid)))
        write()
        while True:
            current = process(collector_pid)
            if current is None or current[1] != collector[1] or current[0] not in ("T", "t"):
                raise RuntimeError("Dispatcher was changed or resumed externally; relinquishing this window")
            status = json.loads(ipc_status.read_text())
            state = status["state"]
            record.update(ipc_state=state, active_collection_children=sorted(
                pid for pid in record["collection_children"] if process(pid) is not None))
            if state in ("complete", "error"):
                record["state"] = "evaluation_" + state
                break
            evaluator = process(evaluation_pid)
            if evaluator is None or evaluator[1] != evaluation[1]:
                raise RuntimeError("Evaluator exited without a final status")
            if state == "running" and run_started is None:
                run_started = time.monotonic()
            record["state"] = "evaluating" if run_started is not None else "draining_collection"
            write()
            if run_started is None and time.monotonic() - began > max_wait:
                raise TimeoutError("Evaluation did not start within the reserved drain/GPU window")
            if run_started is not None and time.monotonic() - run_started > max_run:
                raise TimeoutError("Evaluation exceeded its reserved execution window")
            time.sleep(poll)
    except BaseException as exc:
        record.update(state="error", error=repr(exc))
        if paused:
            cancel_evaluation(evaluation_pid, evaluation[1])
        raise
    finally:
        current = process(collector_pid)
        if paused and current is not None and current[1] == collector[1]:
            os.kill(collector_pid, signal.SIGCONT)
            record["collector_resumed"] = True
        else:
            record["collector_resumed"] = False
        write()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-pid", type=int, required=True)
    parser.add_argument("--evaluation-pid", type=int, required=True)
    parser.add_argument("--ipc-status", type=Path, required=True)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-wait", type=float, default=3600.)
    parser.add_argument("--max-run", type=float, default=7500.)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError(f"Lease interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    print(json.dumps(lease(args.collector_pid, args.evaluation_pid, args.ipc_status, args.out,
                           collection_dir=args.collection_dir, max_wait=args.max_wait, max_run=args.max_run), indent=2), flush=True)


if __name__ == "__main__":
    main()
