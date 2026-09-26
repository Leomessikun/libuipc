"""Isolate a causal flow policy from the collector's external FMVP package.

The collector imports only this file's NumPy/subprocess client. Its child loads
the flow checkpoint with this worktree's models and keeps separate histories
and seeded Gaussian streams for each simulated slot. Returned actions already
use the environment's normalized world-frame command convention.
"""
from __future__ import annotations

from collections import deque
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np


def read_exact(stream, count):
    parts = bytearray()
    while len(parts) < count:
        chunk = stream.read(count - len(parts))
        if not chunk:
            if not parts:
                return None
            raise EOFError("Partial flow-policy message")
        parts.extend(chunk)
    return bytes(parts)


class FlowController:
    def __init__(self, model, metadata, *, noise="random"):
        if noise not in ("random", "zero"):
            raise ValueError("Unknown flow sampling mode")
        self.model, self.metadata, self.noise = model, metadata, noise
        self.reset(0)

    def reset(self, seed, contract=None):
        if contract is not None:
            expected = self.metadata["contract"]
            for key in ("point_budget", "obs_mode", "collision_geometry"):
                if expected[key] != contract[key]:
                    raise ValueError(f"Flow checkpoint {key} differs from the environment")
            for key in ("max_translation_m", "max_rotation_rad", "decision_dt_s"):
                if not np.isclose(expected[key], contract[key], rtol=1e-7, atol=1e-10):
                    raise ValueError(f"Flow checkpoint {key} differs from the environment")
        self.seed, self.histories, self.generators = int(seed), {}, {}

    def act(self, observation, slot):
        import torch

        observation = np.asarray(observation, np.float32)
        if observation.shape != (self.model.spec.dim,) or not np.isfinite(observation).all():
            raise ValueError("Malformed flow observation")
        if slot not in self.histories:
            self.histories[slot] = deque(maxlen=self.model.history_length)
            self.generators[slot] = torch.Generator(device="cpu").manual_seed(self.seed + 1009 * int(slot))
        history = self.histories[slot]
        history.append(observation.copy())
        missing = self.model.history_length - len(history)
        frames = np.stack([history[0]] * missing + list(history))[None]
        valid = np.array([[False] * missing + [True] * len(history)])
        shape = (1, self.model.horizon * 6)
        noise = (torch.randn(shape, generator=self.generators[slot]) if self.noise == "random"
                 else torch.zeros(shape))
        action = self.model.sample(torch.from_numpy(frames), torch.from_numpy(valid), noise=noise)[0, 0]
        result = action.numpy().astype(np.float32, copy=True)
        if result.shape != (6,) or not np.isfinite(result).all():
            raise RuntimeError("Flow policy produced an invalid command")
        return result


class FlowPolicyClient:
    """CPU inference; IPC remains free to use CUDA in the parent process."""

    def __init__(self, checkpoint, *, noise="random"):
        root = str(Path(__file__).resolve().parent.parent)
        env = dict(os.environ, PYTHONPATH=root, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.proc = subprocess.Popen([sys.executable, "-u", "-m", "uipc_manip.flow_client", "--serve",
                                      "--checkpoint", str(checkpoint), "--noise", noise],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        startup = bytearray()
        while True:
            line = self.proc.stderr.readline()
            startup.extend(line)
            if line.strip() == b"[flow] ready":
                break
            if not line:
                self.proc.wait()
                raise RuntimeError("Flow policy startup failed: " + startup.decode(errors="replace"))

    def request(self, **arrays):
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        payload = buffer.getvalue()
        self.proc.stdin.write(struct.pack("<Q", len(payload)) + payload)
        self.proc.stdin.flush()
        raw = read_exact(self.proc.stdout, 24)
        if raw is None:
            raise RuntimeError("Flow policy closed: " + self.proc.stderr.read().decode(errors="replace"))
        return np.frombuffer(raw, dtype="<f4").copy()

    def reset(self, seed, contract):
        self.request(operation="reset", seed=seed, contract=json.dumps(contract))

    def act(self, observation, slot):
        return self.request(operation="act", observation=np.asarray(observation, np.float32), slot=slot)

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


def serve(checkpoint, noise):
    import torch
    from .train_flow_bc import ChunkFlowPolicy

    torch.set_num_threads(1)
    model, metadata = ChunkFlowPolicy.load(checkpoint, "cpu")
    controller = FlowController(model, metadata, noise=noise)
    print("[flow] ready", file=sys.stderr, flush=True)
    while (header := read_exact(sys.stdin.buffer, 8)) is not None:
        size = struct.unpack("<Q", header)[0]
        payload = read_exact(sys.stdin.buffer, size)
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            operation = str(data["operation"])
            if operation == "reset":
                controller.reset(int(data["seed"]), json.loads(str(data["contract"])))
                action = np.zeros(6, np.float32)
            elif operation == "act":
                action = controller.act(data["observation"], int(data["slot"]))
            else:
                raise ValueError(f"Unknown flow request: {operation}")
        sys.stdout.buffer.write(action.astype("<f4").tobytes())
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--noise", choices=("random", "zero"), default="random")
    args = parser.parse_args()
    serve(args.checkpoint, args.noise)
