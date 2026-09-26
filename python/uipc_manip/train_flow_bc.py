"""Offline flow-matching imitation of FMVP action chunks, without simulator calls.

Reuses the FQL vector-field API and tool-point encoder. There is no critic or
RL update. The encoder starts from scratch; this is not FMVP weight conversion.
Example: ``python -m uipc_manip.train_flow_bc --data CACHE --out RUN --device cpu``.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .fql import FQLConfig, FlowPolicy
from .models import EncoderConfig, trunk_layers
from .obs import ObsSpec
from .rollout_chunks import ChunkDataset


def compact_encoder():
    return EncoderConfig(sa_mlp=[[32, 32, 64], [64, 64, 128], [128, 128, 256]],
                         fp_mlp=[[128, 128], [128, 64], [64, 64, 64]],
                         linear_mlp=[64, 64], output_dim=50)


class ChunkFlowPolicy(FlowPolicy):
    """An ordered, masked point-cloud history conditions one joint 6H flow."""

    def __init__(self, spec, cfg, history=3, horizon=8):
        if min(history, horizon) < 1:
            raise ValueError("History and action horizon must be positive")
        super().__init__(spec, 6 * horizon, cfg, timed=True)
        self.cfg, self.history_length, self.horizon = cfg, int(history), int(horizon)
        width = self.history_length * (self.encoder.feature_dim + spec.extra_dim + 1) + 6 * self.horizon + 1
        self.trunk = trunk_layers(width, cfg.hidden_dim, 6 * self.horizon, "residual", cfg.trunk_blocks)

    def encode_history(self, observations, valid):
        if observations.ndim != 3 or observations.shape[1:] != (self.history_length, self.spec.dim):
            raise ValueError("Expected [batch, history, observation_dim]")
        if valid.shape != observations.shape[:2] or not valid[:, -1].all():
            raise ValueError("History mask must include the current observation")
        pos, feat, point_valid, extra = self.spec.unpack_torch(observations.flatten(0, 1))
        # Drop shared trailing padding before the dense PointNet++ neighborhoods.
        used = point_valid.any(dim=0).nonzero()
        end = int(used.max()) + 1 if used.numel() else 1
        latent = self._frame_latent((pos[:, :end], feat[:, :end], point_valid[:, :end], extra))
        latent = latent.reshape(len(observations), self.history_length, -1)
        latent = latent * valid[..., None]
        return torch.cat([latent, valid[..., None].to(latent.dtype)], dim=-1).flatten(1)

    def forward(self, observations, valid, x, time):
        return self.vector(self.encode_history(observations, valid), x, time)

    def loss(self, batch):
        encoded = self.encode_history(batch["obs"], batch["history_valid"])
        target = batch["actions"].flatten(1)
        noise, t = torch.randn_like(target), torch.rand((len(target), 1), device=target.device)
        velocity = self.vector(encoded, (1 - t) * noise + t * target, t)
        error = (velocity - (target - noise)).reshape(-1, self.horizon, 6).square()
        mask = batch["action_valid"][..., None]
        return (error * mask).sum() / (mask.sum().clamp_min(1) * 6)

    @torch.no_grad()
    def sample(self, observations, valid, *, noise=None):
        encoded = self.encode_history(observations, valid)
        x = torch.randn((len(observations), self.horizon * 6), device=observations.device) if noise is None else noise.clone()
        if x.shape != (len(observations), self.horizon * 6):
            raise ValueError("Noise shape must match the flattened action chunk")
        for k in range(self.cfg.flow_steps):
            t = x.new_full((len(x), 1), k / self.cfg.flow_steps)
            x = x + self.vector(encoded, x, t) / self.cfg.flow_steps
        return x.clamp(-1, 1).reshape(-1, self.horizon, 6)

    def save(self, path, *, metadata, step, optimizer=None):
        path = Path(path)
        payload = dict(format="dressing_flow_bc_chunks_v1", config=asdict(self.cfg),
                       point_budget=self.spec.point_budget, history=self.history_length, horizon=self.horizon,
                       model=self.state_dict(), metadata=metadata, step=step,
                       optimizer=optimizer.state_dict() if optimizer is not None else None)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "dressing_flow_bc_chunks_v1":
            raise ValueError("Expected an action-chunk flow BC checkpoint")
        model = cls(ObsSpec(payload["point_budget"]), FQLConfig(**payload["config"]),
                    payload["history"], payload["horizon"]).to(device)
        model.load_state_dict(payload["model"])
        return model.eval(), payload["metadata"]


def tensor_batch(batch, device):
    return {key: torch.as_tensor(value, device=device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, *, batch_size, batches, seed, device):
    """Fixed windows and noise; loss/MSE are diagnostics, not dressing success."""
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    values = []
    model.eval()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        for _ in range(batches):
            batch = tensor_batch(dataset.sample(batch_size, rng), device)
            loss = model.loss(batch)
            sample = model.sample(batch["obs"], batch["history_valid"])
            mask = batch["action_valid"][..., None]
            denom = mask.sum() * 6
            mse = ((sample - batch["actions"]).square() * mask).sum() / denom
            zero = (batch["actions"].square() * mask).sum() / denom
            values.append([float(loss), float(mse), float(zero)])
    model.train()
    return dict(zip(("val_flow_loss", "val_sample_mse", "val_zero_command_mse"), np.mean(values, axis=0).tolist()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.eval_every, args.eval_batches, args.cpu_threads) < 1:
        raise ValueError("Step, batch, evaluation and thread counts must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng, device = np.random.default_rng(args.seed), torch.device(args.device)
    train = ChunkDataset(args.data, "train", args.history, args.horizon)
    validation = ChunkDataset(args.data, "validation", args.history, args.horizon)
    cfg = FQLConfig(encoder=compact_encoder(), hidden_dim=args.hidden_dim,
                    flow_steps=args.flow_steps, learning_rate=args.lr)
    model = ChunkFlowPolicy(ObsSpec(train.manifest["contract"]["point_budget"]), cfg, args.history, args.horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    contract=train.manifest["contract"], data_summary=train.manifest["summary"],
                    data_manifest=str((args.data / "manifest.json").resolve()),
                    data_manifest_sha256=hashlib.sha256((args.data / "manifest.json").read_bytes()).hexdigest(),
                    sampling="uniform body, episode, decision; recorded holds retained; masked future padding",
                    encoder_initialization="random compact PointNet++; no FMVP weights loaded",
                    deployment="Execute the first command, append the next observation, then replan at the recorded decision_dt_s.",
                    scope="Offline static imitation baseline. Offline metrics do not establish closed-loop success.")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start, best = time.monotonic(), float("inf")
    with (args.out / "metrics.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            batch = tensor_batch(train.sample(args.batch_size, rng), device)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite flow loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics = dict(step=step, train_flow_loss=float(loss.detach()), elapsed_s=time.monotonic() - start)
                metrics.update(evaluate(model, validation, batch_size=args.batch_size, batches=args.eval_batches,
                                        seed=args.seed + 1, device=device))
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                print(json.dumps(metrics), flush=True)
                model.save(args.out / "latest.pt", metadata=metadata, step=step, optimizer=optimizer)
                if metrics["val_flow_loss"] < best:
                    best = metrics["val_flow_loss"]
                    model.save(args.out / "best.pt", metadata=metadata, step=step)
    return args.out / "best.pt"


if __name__ == "__main__":
    main()
