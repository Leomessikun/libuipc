"""Prepare accepted FMVP recordings for causal action-chunk imitation.

Only observation/command arrays are copied; the source cloth meshes stay in
place. The audit checks recorded validity, not independent physical success.
Run ``python -m uipc_manip.rollout_chunks --source DATA --out CACHE``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

from .obs import ObsSpec


def split_bodies(episodes, fraction, seed):
    """Keep every replica and garment of a body ID on the same side."""
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    bodies = sorted({e["body"] for e in episodes})
    if len(bodies) < 2:
        raise ValueError("At least two body IDs are needed for a held-out split")
    order = np.random.default_rng(seed).permutation(bodies)
    n = min(len(bodies) - 1, max(1, round(len(bodies) * fraction)))
    validation = set(order[:n].tolist())
    for episode in episodes:
        episode["split"] = "validation" if episode["body"] in validation else "train"


def read_episode(row):
    path = Path(row["log"]).with_suffix("") / row["path"]
    config = json.loads((path.parent / "config.json").read_text())["config"]
    with np.load(path, allow_pickle=False) as data:
        obs = np.asarray(data["obs"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        grasp = data["grasp_valid"]
        controller = data["controller_id"]
        metadata = json.loads(str(data["metadata_json"]))
    spec = ObsSpec(int(config["point_budget"]))
    steps = len(actions)
    if steps < 1 or actions.shape != (steps, 6) or obs.shape != (steps + 1, spec.dim):
        raise ValueError("Expected T six-axis commands and T+1 observations")
    if not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.abs(actions).max() > 1 + 1e-6:
        raise ValueError("Nonfinite observation/command or command outside [-1,1]")
    if grasp.shape != (steps,) or not grasp.all() or controller.shape != (steps,):
        raise ValueError("Invalid grasp or misaligned controller labels")
    if (not all(metadata.get(k) for k in ("accepted", "stable_success", "hold_complete", "valid_grasp"))
            or metadata.get("sim_error") or row.get("sim_error")):
        raise ValueError("Episode lacks recorded stable, valid-grasp acceptance")
    if int(metadata["body"]) != int(row["body"]) or int(row["transitions"]) != steps:
        raise ValueError("Manifest identity/length disagrees with recording")
    contract = {k: metadata[k] for k in ("max_translation_m", "max_rotation_rad", "decision_dt_s",
                                        "tcp_rotation_convention", "collision_geometry")}
    contract.update(point_budget=spec.point_budget, obs_mode=config["obs"]["mode"], action_dim=6,
                    action_labels="actions: normalized commands submitted to the environment")
    if (not np.isclose(contract["max_translation_m"], config["max_translation"])
            or not np.isclose(contract["max_rotation_rad"], config["max_rotation"])
            or not np.isclose(contract["decision_dt_s"], config["dt"] * config["action_repeat"])):
        raise ValueError("Command scale/period differs between configuration and metadata")
    digest = hashlib.sha256()
    for array in (obs, actions):
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    record = dict(source_path=str(path.resolve()), body=int(row["body"]), garment=row["garment"],
                  replica=row["replica"], seed=row["seed"], transitions=steps,
                  training_arrays_sha256=digest.hexdigest(), checkpoint_sha256=metadata["checkpoint_sha256"],
                  success_geometry=metadata["success_geometry"], stop_proximal_upper=metadata.get("stop_proximal_upper"),
                  controller_counts={str(k): int(v) for k, v in zip(*np.unique(controller, return_counts=True))})
    return obs, actions, contract, record


def prepare(source, out, *, val_fraction=.2, seed=20260926):
    source, out = Path(source), Path(out)
    if not 0 < val_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    raw = (source / "manifest.json").read_bytes()
    rows = json.loads(raw)["accepted"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "source_manifest.json").write_bytes(raw)
    episodes, rejected, duplicates, seen = [], [], [], {}
    contract = None
    action_min, action_max = np.full(6, np.inf), np.full(6, -np.inf)
    for i, row in enumerate(rows):
        try:
            if not row.get("accepted"):
                raise ValueError("Manifest entry is not accepted")
            obs, actions, current, record = read_episode(row)
            if contract is not None and current != contract:
                raise ValueError("Observation/action contract differs from the dataset")
            key = record["training_arrays_sha256"]
            if key in seen:
                duplicates.append(dict(source_path=record["source_path"], duplicate_of=seen[key]))
                continue
            contract = current
            seen[key] = record["source_path"]
            stem = f"episode_{len(episodes):04d}"
            np.save(out / f"{stem}_obs.npy", obs, allow_pickle=False)
            np.save(out / f"{stem}_actions.npy", actions, allow_pickle=False)
            record.update(obs=f"{stem}_obs.npy", actions=f"{stem}_actions.npy")
            episodes.append(record)
            action_min = np.minimum(action_min, actions.min(0))
            action_max = np.maximum(action_max, actions.max(0))
        except (OSError, ValueError, KeyError) as error:
            rejected.append(dict(log=row.get("log"), path=row.get("path"), reason=str(error)))
        if (i + 1) % 100 == 0:
            print(f"[prepare] {i + 1}/{len(rows)} inspected, {len(episodes)} retained", flush=True)
    split_bodies(episodes, val_fraction, seed)
    summary = {}
    for split in ("train", "validation"):
        subset = [e for e in episodes if e["split"] == split]
        summary[split] = dict(episodes=len(subset), bodies=len({e["body"] for e in subset}),
                              transitions=sum(e["transitions"] for e in subset),
                              garments=dict(Counter(e["garment"] for e in subset)))
    payload = dict(format="fmvp_action_chunks_v1", source=str(source.resolve()),
                   source_manifest_sha256=hashlib.sha256(raw).hexdigest(), seed=seed,
                   validation_fraction=val_fraction, split_unit="body ID across all garments and replicas",
                   validation_scope="Student imitation holdout; teacher may have seen these bodies. Not a garment holdout.",
                   audit_scope="Recorded acceptance, shapes, finiteness, command bounds, grasp and contract consistency; no mesh revalidation.",
                   contract=contract, action_min=action_min.tolist(), action_max=action_max.tolist(),
                   summary=summary, episodes=episodes, rejected=rejected, duplicates=duplicates)
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(summary=summary, rejected=len(rejected), duplicates=len(duplicates)), indent=2), flush=True)
    return payload


class ChunkDataset:
    """Causal history ending at t, commands starting at t; neither crosses an episode."""

    def __init__(self, directory, split, history=3, horizon=8):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["format"] != "fmvp_action_chunks_v1" or min(history, horizon) < 1:
            raise ValueError("Invalid dataset format or window length")
        self.episodes = [e for e in self.manifest["episodes"] if e["split"] == split]
        if not self.episodes:
            raise ValueError(f"No episodes in {split}")
        self.history, self.horizon = int(history), int(horizon)
        self.ends = np.cumsum([e["transitions"] for e in self.episodes])
        self.starts = np.r_[0, self.ends[:-1]]
        self.bodies = sorted({e["body"] for e in self.episodes})
        self.body_episodes = {b: [i for i, e in enumerate(self.episodes) if e["body"] == b] for b in self.bodies}

    def __len__(self):
        return int(self.ends[-1])

    @lru_cache(maxsize=16)
    def arrays(self, episode):
        row = self.episodes[episode]
        return tuple(np.load(self.directory / row[key], mmap_mode="r", allow_pickle=False) for key in ("obs", "actions"))

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        t = int(index - self.starts[episode])
        obs, actions = self.arrays(episode)
        past = np.arange(t - self.history + 1, t + 1)
        future = np.arange(t, t + self.horizon)
        valid = future < len(actions)
        target = np.zeros((self.horizon, 6), np.float32)
        target[valid] = actions[future[valid]]
        return dict(obs=np.array(obs[np.maximum(past, 0)], copy=True), history_valid=past >= 0,
                    actions=target, action_valid=valid)

    def sample(self, batch_size, rng):
        """Uniform body, then episode, then decision; retain recorded hold decisions."""
        items = []
        for _ in range(batch_size):
            body = int(rng.choice(self.bodies))
            episode = int(rng.choice(self.body_episodes[body]))
            index = self.starts[episode] + int(rng.integers(self.episodes[episode]["transitions"]))
            items.append(self[int(index)])
        return {key: np.stack([item[key] for item in items]) for key in items[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    prepare(args.source, args.out, val_fraction=args.val_fraction, seed=args.seed)


if __name__ == "__main__":
    main()
