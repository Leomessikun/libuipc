"""Upload a running uipc_manip CSV run to Weights & Biases.

This intentionally runs out of process so existing training jobs do not need
to import wandb or be restarted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any


def _number(value: str) -> float | int | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return int(x) if x.is_integer() else x


def _rows(path: Path, kind: str) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return []
    out = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = _number(row.get("transitions", row.get("step", "")))
            if step is None:
                continue
            metrics = {}
            for key, value in row.items():
                if key in {"step", "transitions"}:
                    continue
                number = _number(value)
                if number is not None:
                    metrics[f"{kind}/{key}"] = number
            if metrics:
                out.append((int(step), metrics))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--project", default="libuipc-dressing")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--poll", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    import wandb

    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    init_kwargs = dict(
        project=args.project,
        name=args.name or run_dir.name,
        config=config,
        tags=["uipc_manip", "dressing", "dense-residual", "baseline"],
        save_code=False,
    )
    if args.entity:
        init_kwargs["entity"] = args.entity
    run = wandb.init(**init_kwargs)
    print(f"wandb_url={run.url}", flush=True)

    sent: set[tuple[str, int]] = set()
    while True:
        grouped: dict[int, dict[str, Any]] = {}
        for kind, filename in (("train", "train_log.csv"), ("eval", "eval_log.csv")):
            for step, metrics in _rows(run_dir / filename, kind):
                key = (kind, step)
                if key in sent:
                    continue
                sent.add(key)
                grouped.setdefault(step, {}).update(metrics)
        for step in sorted(grouped):
            grouped[step]["transitions"] = step
            wandb.log(grouped[step], step=step)
            print(f"uploaded transitions={step}", flush=True)

        alive = args.pid is not None and Path(f"/proc/{args.pid}").exists()
        if args.once or (args.pid is not None and not alive):
            break
        time.sleep(args.poll)

    checkpoint = run_dir / "checkpoints" / "best.pt"
    if checkpoint.exists():
        artifact = wandb.Artifact(f"{run.name}-best", type="model")
        artifact.add_file(str(checkpoint), name="best.pt")
        run.log_artifact(artifact)
    run.finish()
    print("wandb_finished", flush=True)


if __name__ == "__main__":
    main()
