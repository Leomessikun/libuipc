"""Offline trajectory pretraining of the actor's representation from sequence replay snapshots.

    PYTHONPATH=python python -m uipc_manip.pretrain_offline --replay RUN/checkpoints/replay_latest [...] \\
        --out output/uipc_manip/pretrain_r13 --history-length 8 --history-kind rlt --steps 20000

The network flags are ``train_sac``'s (``--actor --encoder --hidden-dim --point-budget
--history-length --history-kind --rlt-*``), so the protocol of the saved representation is the one
a later online run with the same flags builds; ``--init-representation`` there transfers it.

What trains: the actor's spatial encoder and, above one frame, its recurrent history, through the
action-conditioned :class:`~uipc_manip.rlt.TrajectoryPretrainingHead` — from the state after each
recorded frame and the command taken there, predict the privileged state reached next and the
reward. Every recorded episode qualifies, failed ones included. The policy trunk is not trained
and not transferred. At ``--history-length 1`` the state is the frame latent alone, which is the
control the "does history improve prediction" question needs on the same episodes.

Episodes, not rows, are split into training and validation by a seeded hash; target statistics
are fitted on the training split only; validation reports the per-target losses next to the
constant predictor's (the training mean), which is the number that says whether the
representation predicts anything at all. A low loss here is not a dressing result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .obs import ObsSpec
from .replay import FlatReplayBuffer
from .rlt import TrajectoryPretrainingHead


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", nargs="+", required=True, help="Sequence replay snapshot directories (replay.json, or a replay_set.json with buffer_XX).")
    p.add_argument("--out", required=True, help="Run directory for the log and checkpoints.")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=8, help="Validation windows per evaluation, batch-size each, the same windows every time.")
    p.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of episodes held out for validation.")
    p.add_argument("--lr", type=float, default=1.0e-4, help="Spatial encoder and head learning rate.")
    p.add_argument("--history-lr", type=float, default=None, help="Recurrent history learning rate (RESeL: a context encoder wants its own); default --lr.")
    p.add_argument("--priv-weight", type=float, default=1.0)
    p.add_argument("--reward-weight", type=float, default=1.0)
    p.add_argument("--command-weight", type=float, default=0.0, help="Command regression weight; zero keeps failed behaviour out of the objective.")
    p.add_argument("--log-interval", type=int, default=50)
    return p


# ------------------------------------------------------------------------ snapshots
def read_snapshot(directory: str | Path) -> list[dict]:
    """The buffers of a snapshot directory, each as chronological arrays plus its ``replay.json``."""
    directory = Path(directory)
    if (directory / "replay_set.json").exists():
        keys = json.loads((directory / "replay_set.json").read_text())["keys"]
        parts = [directory / f"buffer_{i:02d}" for i in range(len(keys))]
    elif (directory / "replay.json").exists():
        parts = [directory]
    else:
        raise FileNotFoundError(f"{directory} holds neither replay.json nor replay_set.json")
    out = []
    for part in parts:
        payload = json.loads((part / "replay.json").read_text())
        if not payload.get("sequence"):
            raise ValueError(f"{part} is a flat replay snapshot; offline pretraining needs episode identities (--sequence-replay)")
        with np.load(part / "replay.npz") as data:
            arrays = {name: data[name] for name in data.files}
        with (part / "replay.npz").open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        out.append({"path": str(part.resolve()), "sha256": digest, "payload": payload, "arrays": arrays})
    return out


def split_episodes(sources: list[dict], val_fraction: float, seed: int) -> tuple[set, set]:
    """Episode keys ``(source, stream, episode)`` for training and validation, split by a seeded shuffle."""
    keys = []
    for index, source in enumerate(sources):
        arrays = source["arrays"]
        pairs = np.stack([arrays["stream_ids"], arrays["episode_ids"]], axis=1)
        for stream, episode in np.unique(pairs, axis=0):
            keys.append((index, int(stream), int(episode)))
    if len(keys) < 2:
        raise ValueError("Offline pretraining needs at least two recorded episodes, one to validate on")
    order = np.random.default_rng(seed).permutation(len(keys))
    n_val = min(len(keys) - 1, max(1, int(round(val_fraction * len(keys)))))
    val = {keys[i] for i in order[:n_val]}
    return set(keys) - val, val


def fill_buffers(sources: list[dict], train_keys: set, val_keys: set, batch_size: int, device) -> tuple[FlatReplayBuffer, FlatReplayBuffer]:
    """Re-add every row into a training or a validation buffer, chronologically, streams renumbered per source."""
    obs_dim = int(sources[0]["payload"]["obs_dim"])
    action_dim = int(sources[0]["payload"]["action_dim"])
    priv_dim = int(sources[0]["payload"].get("priv_dim", 0))
    for source in sources:
        p = source["payload"]
        if (int(p["obs_dim"]), int(p["action_dim"]), int(p.get("priv_dim", 0))) != (obs_dim, action_dim, priv_dim):
            raise ValueError("Replay snapshots disagree on observation, action or privileged dimensions")
    rows = {"train": 0, "val": 0}
    for source_index, source in enumerate(sources):
        for index in range(int(source["payload"]["size"])):
            key = (source_index, int(source["arrays"]["stream_ids"][index]), int(source["arrays"]["episode_ids"][index]))
            rows["val" if key in val_keys else "train"] += 1
    buffers = {name: FlatReplayBuffer(obs_dim, action_dim, max(1, n), batch_size, device, priv_dim=priv_dim, sequence=True)
               for name, n in rows.items()}
    stream_base = 0
    for source_index, source in enumerate(sources):
        a = source["arrays"]
        for index in range(int(source["payload"]["size"])):
            stream, episode = int(a["stream_ids"][index]), int(a["episode_ids"][index])
            key = (source_index, stream, episode)
            buffer = buffers["val" if key in val_keys else "train"]
            buffer.add(
                a["obs"][index], a["actions"][index], float(a["rewards"][index, 0]), a["next_obs"][index], bool(a["not_dones"][index, 0] < 0.5),
                a["priv"][index] if priv_dim else None, a["next_priv"][index] if priv_dim else None,
                stream_id=stream_base + stream, episode_id=episode, episode_step=int(a["episode_steps"][index]),
                episode_end=bool(a["episode_ends"][index]),
            )
        stream_base += int(a["stream_ids"].max(initial=-1)) + 1
    return buffers["train"], buffers["val"]


def target_statistics(buffer: FlatReplayBuffer) -> dict:
    """Mean and standard deviation of the successor privileged state and the reward over the training rows."""
    n = buffer.size
    stats = {"reward_mean": float(buffer._rewards[:n].mean()), "reward_std": float(max(buffer._rewards[:n].std(), 1.0e-6))}
    if buffer.priv_dim:
        stats["priv_mean"] = buffer._next_priv[:n].mean(0).tolist()
        stats["priv_std"] = np.maximum(buffer._next_priv[:n].std(0), 1.0e-6).tolist()
    return stats


# ------------------------------------------------------------------------ the model
class Representation:
    """The actor's representation and the prediction head, with the state function of each history kind."""

    def __init__(self, agent, priv_dim: int, weights: dict, device) -> None:
        actor = agent.actor
        if actor.history is not None and not hasattr(actor.history, "sequence"):
            raise ValueError("The frame concatenation has no per-position state; pretrain --history-length 1 or --history-kind rlt")
        self.agent, self.actor = agent, actor
        self.length = int(agent.cfg.history_length)
        dim = actor.history.out_dim if actor.history is not None else int(actor.trunk[0].in_features)
        self.head = TrajectoryPretrainingHead(dim, agent.action_dim, priv_dim, reward=True, **weights).to(device)

    def modules(self) -> dict:
        out = {"encoder": self.actor.encoder}
        if self.actor.history is not None:
            out["history"] = self.actor.history
        return out

    def states(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        length = self.length
        frames = self.agent._unpack(batch.obs[:, :length].reshape(-1, self.agent.spec.dim))
        valid = batch.valid.bool()
        latent = self.actor._frame_latent(frames).reshape(*valid.shape, -1)
        if self.actor.history is None:
            return latent, valid
        return self.actor.history.sequence(latent, valid, batch.actions[:, :-1]), valid

    def losses(self, batch, stats: dict) -> dict:
        state, valid = self.states(batch)
        device = state.device
        next_priv = None
        if self.head.priv is not None:
            if batch.priv is None:
                raise ValueError("The snapshot carries no privileged state; record with --record-privileged or set --priv-weight 0")
            mean = torch.as_tensor(stats["priv_mean"], device=device, dtype=state.dtype)
            std = torch.as_tensor(stats["priv_std"], device=device, dtype=state.dtype)
            next_priv = (batch.priv[:, 1 : self.length + 1] - mean) / std
        rewards = (batch.rewards - stats["reward_mean"]) / stats["reward_std"]
        out = self.head(state, valid, batch.actions, next_priv, rewards)
        # The constant predictor on the same windows: what "no representation" scores in these units.
        with torch.no_grad():
            if next_priv is not None:
                out["priv_baseline"] = next_priv[valid].square().mean()
            out["reward_baseline"] = rewards[valid].square().mean()
        return out


def _sample(buffer: FlatReplayBuffer, length: int, batch_size: int):
    return buffer.sample_sequences(length, batch_size, pad=True, strict_context=True)


def evaluate(model: Representation, buffer: FlatReplayBuffer, stats: dict, batch_size: int, batches: int, seed: int) -> dict:
    """Per-target validation losses over the same windows every time."""
    state = np.random.get_state()
    np.random.seed(seed)
    totals: dict[str, float] = {}
    try:
        with torch.no_grad():
            for _ in range(batches):
                out = model.losses(_sample(buffer, model.length, batch_size), stats)
                for key, value in out.items():
                    totals[key] = totals.get(key, 0.0) + float(value.item())
    finally:
        np.random.set_state(state)
    return {f"val_{key}": value / batches for key, value in totals.items()}


# ------------------------------------------------------------------------ the run
def main(argv: list[str] | None = None) -> Path:
    from . import train_sac
    from .sac import SACAgent

    args, rest = build_parser().parse_known_args(argv)
    targs = train_sac.build_parser().parse_args(rest)
    train_sac.resolve_defaults(targs)
    if targs.resume or targs.eval_only:
        raise ValueError("Offline pretraining starts a representation from scratch; it neither resumes nor plays back")
    if int(args.steps) < 1 or int(args.eval_every) < 1 or int(args.eval_batches) < 1:
        raise ValueError("--steps, --eval-every and --eval-batches must be positive")
    weights = {"command_weight": float(args.command_weight), "priv_weight": float(args.priv_weight), "reward_weight": float(args.reward_weight)}
    device = torch.device(targs.device)
    torch.manual_seed(int(targs.seed))
    np.random.seed(int(targs.seed))

    sources = read_snapshot_list(args.replay)
    train_keys, val_keys = split_episodes(sources, float(args.val_fraction), int(targs.seed))
    train, val = fill_buffers(sources, train_keys, val_keys, int(targs.batch_size), device)
    if train.priv_dim == 0 and weights["priv_weight"] > 0.0:
        raise ValueError("The snapshots carry no privileged state; record with --record-privileged or set --priv-weight 0")
    stats = target_statistics(train)

    spec = ObsSpec(targs.point_budget)
    if spec.dim != train.obs_dim:
        raise ValueError(f"--point-budget {targs.point_budget} gives {spec.dim}-float observations; the snapshots hold {train.obs_dim}")
    cfg = train_sac.build_sac_config(targs)
    agent = SACAgent(spec, train.action_dim, cfg, device)
    model = Representation(agent, train.priv_dim if weights["priv_weight"] > 0.0 else 0, weights, device)
    history_lr = float(args.lr if args.history_lr is None else args.history_lr)
    groups = [{"params": list(model.actor.encoder.parameters()) + list(model.head.parameters()), "lr": float(args.lr)}]
    if model.actor.history is not None:
        groups.append({"params": list(model.actor.history.parameters()), "lr": history_lr})
    optimizer = torch.optim.Adam(groups)

    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    provenance = {
        "sources": [{"path": s["path"], "sha256": s["sha256"], "rows": int(s["payload"]["size"]), "metadata": s["payload"].get("metadata", {})} for s in sources],
        "split": {"val_fraction": float(args.val_fraction), "seed": int(targs.seed), "train_episodes": len(train_keys), "val_episodes": len(val_keys),
                  "train_rows": int(train.size), "val_rows": int(val.size), "val_keys": sorted(val_keys)},
        "normalization": stats, "weights": weights, "lr": float(args.lr), "history_lr": history_lr,
        "history_length": int(cfg.history_length), "history_kind": str(cfg.history_kind), "components": sorted(model.modules()),
        "command": ["--replay", *args.replay, "--out", args.out, *rest],
    }
    (out / "config.json").write_text(json.dumps({"args": vars(args), "sac_config": cfg.to_dict(), "pretraining": provenance}, indent=2, default=str) + "\n")

    def save(name: str, step: int, record: dict) -> Path:
        payload = {"step": int(step), "actor": agent.actor.state_dict(), "head": model.head.state_dict(), "sac_config": cfg.to_dict(),
                   "protocol": agent.protocol(), "metadata": {"pretraining": {**provenance, **record}}}
        path = out / "checkpoints" / f"{name}.pt"
        torch.save(payload, path)
        path.with_suffix(".json").write_text(json.dumps({"step": int(step), "protocol": payload["protocol"], "metadata": payload["metadata"]}, indent=2, default=str) + "\n")
        return path

    log_path = out / "pretrain_log.csv"
    fields: list[str] = []
    best: tuple[float, int] | None = None
    t0 = time.time()
    with log_path.open("w", newline="") as handle:
        writer = None
        for step in range(1, int(args.steps) + 1):
            agent.actor.train()
            out_losses = model.losses(_sample(train, model.length, int(targs.batch_size)), stats)
            optimizer.zero_grad()
            out_losses["loss"].backward()
            optimizer.step()
            row = {"step": step, "seconds": time.time() - t0, **{k: float(v.item()) for k, v in out_losses.items()}}
            if step % int(args.eval_every) == 0 or step == int(args.steps):
                agent.actor.eval()
                row.update(evaluate(model, val, stats, int(targs.batch_size), int(args.eval_batches), int(targs.seed) + 1))
                record = {"steps": step, "val": {k: v for k, v in row.items() if k.startswith("val_")}}
                if best is None or row["val_loss"] < best[0]:
                    best = (row["val_loss"], step)
                    save("pretrain_best", step, {**record, "best": {"step": step, "val_loss": row["val_loss"]}})
                print(f"[pretrain] step={step} loss={row['loss']:.4f} val_loss={row['val_loss']:.4f} "
                      + " ".join(f"{k}={v:.4f}" for k, v in row.items() if k.startswith("val_") and k != "val_loss"), flush=True)
            elif step % int(args.log_interval) == 0:
                print(f"[pretrain] step={step} loss={row['loss']:.4f}", flush=True)
            if writer is None:
                fields = list(row) + [f"val_{k}" for k in ("loss", "next_priv", "reward", "next_command", "priv_baseline", "reward_baseline") if f"val_{k}" not in row]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
            handle.flush()
    final = save("pretrain_final", int(args.steps), {"steps": int(args.steps), "best": {"step": best[1], "val_loss": best[0]} if best else None,
                                                    "val": {k: v for k, v in row.items() if k.startswith("val_")}})
    print(f"[pretrain] done best_val={best[0]:.4f}@{best[1]} checkpoints={out / 'checkpoints'}", flush=True)
    return final


def read_snapshot_list(directories) -> list[dict]:
    sources = []
    for directory in directories:
        sources.extend(read_snapshot(directory))
    return sources


if __name__ == "__main__":
    main()
