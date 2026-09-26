"""Teacher-forced command errors by rollout phase, on fixed held-out windows.

These are errors in commanded translation/yaw, not realized tracking errors
or closed-loop success. All checkpoints use the same recorded observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .rollout_chunks import ChunkDataset
from .train_flow_bc import ChunkFlowPolicy, tensor_batch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(data, checkpoints, out, *, batch_size=16, seed=2026092701):
    report = dict(scope=__doc__, data_manifest_sha256=digest(data / "manifest.json"),
                  seed=seed, noise="Seeded Gaussian; first predicted command; recorded observation history",
                  models=[])
    for checkpoint in checkpoints:
        model, metadata = ChunkFlowPolicy.load(checkpoint, "cpu")
        source = Path(metadata["data_manifest"])
        if digest(source) != metadata["data_manifest_sha256"]:
            raise ValueError("Checkpoint's training manifest changed")
        trained = {r["body"] for r in json.loads(source.read_text())["episodes"] if r["split"] == "train"}
        dataset = ChunkDataset(data, "validation", model.history_length, model.horizon)
        if trained.intersection(dataset.bodies):
            raise ValueError("Audited validation bodies appear in checkpoint training")
        if metadata["contract"] != dataset.manifest["contract"]:
            raise ValueError("Checkpoint observation/action contract differs from audit data")
        scale = metadata["contract"]["max_translation_m"] * 1000.
        yaw_scale = metadata["contract"]["max_rotation_rad"] * 180. / np.pi
        phases = {}
        for phase_index, phase in enumerate(("start", "early", "middle", "before_hold", "final_hold")):
            torch.manual_seed(seed + phase_index)
            items = []
            for i, row in enumerate(dataset.episodes):
                length = row["transitions"]
                t = dict(start=0, early=min(10, length - 1), middle=length // 2,
                         before_hold=max(0, length - 21), final_hold=length - 1)[phase]
                items.append(dataset[int(dataset.starts[i] + t)])
            squared, zero = [], []
            for start in range(0, len(items), batch_size):
                subset = items[start:start + batch_size]
                batch = tensor_batch({k: np.stack([item[k] for item in subset]) for k in subset[0]}, "cpu")
                with torch.no_grad():
                    prediction = model.sample(batch["obs"], batch["history_valid"])[:, 0].numpy()
                target = batch["actions"][:, 0].numpy()
                squared.extend((prediction - target)**2)
                zero.extend(target**2)
            squared, zero = np.asarray(squared), np.asarray(zero)
            groups = {"all": np.ones(len(items), bool)}
            for index in sorted({r.get("source_index", 0) for r in dataset.episodes}):
                groups[f"source_{index}"] = np.array([r.get("source_index", 0) == index for r in dataset.episodes])
            phases[phase] = {}
            for name, mask in groups.items():
                phases[phase][name] = dict(episodes=int(mask.sum()),
                    translation_command_rms_mm=float(np.sqrt(squared[mask, :3].sum(1).mean()) * scale),
                    yaw_command_rms_deg=float(np.sqrt(squared[mask, 5].mean()) * yaw_scale),
                    zero_translation_command_rms_mm=float(np.sqrt(zero[mask, :3].sum(1).mean()) * scale),
                    zero_yaw_command_rms_deg=float(np.sqrt(zero[mask, 5].mean()) * yaw_scale))
        report["models"].append(dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                                     training_manifest_sha256=metadata["data_manifest_sha256"], phases=phases))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    report = audit(args.data, args.checkpoints, args.out)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
