"""Call Wang's released policy from this project's environment, across a process boundary.

Their model needs the `curl` conda environment (python 3.9, torch_geometric, torch_scatter) and
ours needs the Genesis venv (python 3.13), so the policy runs as a subprocess speaking the framing
in :func:`uipc_manip.wang_bridge.serve`. Nothing here imports torch_geometric.
"""

from __future__ import annotations

import io
import struct
import subprocess
from pathlib import Path

import numpy as np

CURL_PYTHON = "/home/ge47gax/miniconda3/envs/curl/bin/python"
REFERENCE_ROOT = "/home/ge47gax/kun"
DEFAULT_CHECKPOINT = "/home/ge47gax/Desktop/vision_based_policy.pt"


class WangPolicyClient:
    """A subprocess running their actor, driven one observation at a time."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, *, yaw_deg: float = 267.0,
                 device: str = "cpu", voxel: float | None = None,
                 python: str = CURL_PYTHON, package_root: str | Path | None = None) -> None:
        root = str(Path(package_root or Path(__file__).resolve().parent.parent))
        argv = [python, "-m", "uipc_manip.wang_bridge", "--serve",
                "--checkpoint", str(checkpoint), "--device", device, "--yaw", repr(float(yaw_deg))]
        if voxel is not None:
            argv += ["--voxel", repr(float(voxel))]
        self.yaw_deg = float(yaw_deg)
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            # One thread: torch otherwise starts a pool per core in every policy process, and a dozen
            # collectors' policies (79 threads, 250 % CPU each) starved the IPC solvers of the CPU.
            env={"PYTHONPATH": f"{REFERENCE_ROOT}:{root}", "PATH": "/usr/bin:/bin",
                 "CUDA_VISIBLE_DEVICES": "" if device == "cpu" else "0",
                 "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        # CUDA/library warnings can precede the readiness marker. Preserve them
        # for a startup error without treating the first warning as a failure.
        startup = bytearray()
        while True:
            line = self.proc.stderr.readline()
            startup.extend(line)
            if line.strip() == b"[bridge] ready":
                break
            if not line:
                self.proc.wait()
                raise RuntimeError("The policy process did not come up:\n"
                                   + startup.decode(errors="replace"))

    def act(self, pos_rel: np.ndarray, flags: np.ndarray, force: np.ndarray | None = None) -> np.ndarray:
        """The deterministic six-dimensional action for one observation, in our frame.

        ``force`` (our frame) feeds the FiLM layers of FMVP's fine-tunes; see ``ReferencePolicy.act``.
        """
        buffer = io.BytesIO()
        extra = {} if force is None else {"force": np.asarray(force, dtype=np.float32).reshape(3)}
        np.savez(buffer, pos=np.asarray(pos_rel, dtype=np.float32),
                 flags=np.asarray(flags, dtype=np.float32), **extra)
        payload = buffer.getvalue()
        self.proc.stdin.write(struct.pack("<Q", len(payload)) + payload)
        self.proc.stdin.flush()
        raw = b""
        while len(raw) < 24:
            chunk = self.proc.stdout.read(24 - len(raw))
            if not chunk:
                raise RuntimeError("The policy process closed: "
                                   + self.proc.stderr.read(4000).decode(errors="replace"))
            raw += chunk
        return np.frombuffer(raw, dtype=np.float32).copy()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)

    def __enter__(self) -> "WangPolicyClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
