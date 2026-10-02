"""Small history adapter on frozen r1 features; same-capacity current-only control.

The encoder and base actor remain frozen. This is a supervised baseline, not
a new algorithm. The server runs in curl's Python; the client in Genesis'.
"""
from __future__ import annotations

from collections import deque
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

from motion_observation import history_inputs


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter(frames):
    import torch
    model = torch.nn.Sequential(torch.nn.Linear(frames * 53, 256), torch.nn.ReLU(),
                                torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Linear(128, 6))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def encode(policy, pos, flags):
    """The existing FMVP feature contract, with zero force and fixed FPS RNG.

    Both extraction and deployment reset the local CPU RNG for each frame;
    feature noise must not masquerade as motion or differ across students.
    """
    import torch
    from torch_geometric.data import Batch, Data
    from uipc_manip.wang_bridge import GRIPPER, to_reference_cloud
    points, labels = to_reference_cloud(pos, flags, yaw_deg=policy.yaw_deg, voxel=policy.voxel)
    batch = Batch.from_data_list([Data(x=torch.from_numpy(labels), pos=torch.from_numpy(points))])
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        feature, _ = policy.encoder(batch, torch.zeros(1, 3)) if policy.film else policy.encoder(batch)
        row = feature[batch.x[:, GRIPPER] == 1]
        logits = policy.trunk(row)[0, :6]
    return row[0].numpy(), logits.numpy()


class StudentPolicy:
    def __init__(self, checkpoint, package_root, yaw):
        sys.path.insert(0, str(package_root))
        import torch
        from uipc_manip.wang_bridge import ReferencePolicy
        torch.set_num_threads(1)
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        self.config = payload["dynamic_student"]
        if sha256(self.config["base_checkpoint"]) != self.config["base_sha256"]:
            raise ValueError("Student base checkpoint changed")
        if abs(float(yaw) - self.config["yaw"]) > 1e-6:
            raise ValueError("Student coordinate yaw differs from training")
        self.base = ReferencePolicy(self.config["base_checkpoint"], device="cpu", yaw_deg=yaw)
        self.head = adapter(self.config["frames"])
        self.head.load_state_dict(payload["adapter_state_dict"], strict=True)
        self.head.eval()
        self.reset()

    def reset(self):
        self.features = deque(maxlen=self.config["frames"])
        self.tools = deque(maxlen=self.config["frames"])

    def act(self, pos, flags, tool):
        import torch
        feature, logits = encode(self.base, pos, flags)
        self.features.append(feature)
        self.tools.append(np.asarray(tool) @ self.base.rotation.T)
        x = history_inputs(np.asarray(self.features), np.asarray(self.tools), self.config["frames"],
                           current_only=self.config["current_only"])[-1:]
        with torch.no_grad():
            action = (torch.from_numpy(logits) + self.head(torch.from_numpy(x).float())[0]).tanh().numpy()
        return np.clip(np.concatenate((action[:3] @ self.base.rotation, action[3:] @ self.base.rotation)), -1, 1)


class StudentClient:
    def __init__(self, checkpoint, package_root, yaw):
        cmd = ["/home/ge47gax/miniconda3/envs/curl/bin/python", "-u", str(Path(__file__).resolve()),
               "--checkpoint", str(checkpoint), "--package-root", str(package_root), "--yaw", str(yaw)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="", PYTHONPATH=f"/home/ge47gax/kun:{package_root}")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        startup = bytearray()
        while True:
            line = self.proc.stderr.readline(); startup.extend(line)
            if line.strip() == b"[student] ready":
                break
            if not line:
                raise RuntimeError("Student startup failed: " + startup.decode(errors="replace"))

    def request(self, **arrays):
        buffer = io.BytesIO(); np.savez(buffer, **arrays)
        payload = buffer.getvalue()
        self.proc.stdin.write(struct.pack("<Q", len(payload)) + payload); self.proc.stdin.flush()
        result = bytearray()
        while len(result) < 24:
            chunk = self.proc.stdout.read(24 - len(result))
            if not chunk:
                raise RuntimeError("Student closed: " + self.proc.stderr.read().decode(errors="replace"))
            result.extend(chunk)
        return np.frombuffer(result, dtype=np.float32).copy()

    def reset(self):
        self.request(reset=np.array(True))

    def act(self, pos, flags, tool):
        action = self.request(pos=np.asarray(pos, np.float32), flags=np.asarray(flags, np.float32),
                              tool=np.asarray(tool, np.float32))
        if not np.isfinite(action).all() or np.max(np.abs(action)) > 1.00001:
            raise RuntimeError("Invalid student action or corrupted binary bridge reply")
        return action

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close(); self.proc.wait(timeout=30)


def serve(args):
    # FMVP prints a FiLM banner while loading. Keep the binary channel clean,
    # including when Python is unbuffered or the model prints during inference.
    output = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        policy = StudentPolicy(args.checkpoint, args.package_root, args.yaw)
    print("[student] ready", file=sys.stderr, flush=True)
    while True:
        header = sys.stdin.buffer.read(8)
        if len(header) != 8:
            break
        size = struct.unpack("<Q", header)[0]
        raw = bytearray()
        while len(raw) < size:
            chunk = sys.stdin.buffer.read(size - len(raw))
            if not chunk:
                return
            raw.extend(chunk)
        with np.load(io.BytesIO(raw), allow_pickle=False) as request:
            if "reset" in request:
                policy.reset(); action = np.zeros(6)
            else:
                with redirect_stdout(sys.stderr):
                    action = policy.act(request["pos"], request["flags"], request["tool"])
        output.write(np.asarray(action, np.float32).tobytes()); output.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--yaw", type=float, default=267.)
    serve(parser.parse_args())
