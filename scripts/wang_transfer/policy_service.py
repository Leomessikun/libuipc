"""Keep one CUDA checkpoint resident across rollout worker lifetimes.

Local Unix socket only. Requests contain NumPy arrays, never pickle objects.
Closing a rollout client does not unload the model. An unused server exits
after its idle timeout; the dataset records its socket and checkpoint identity.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / '.claude/worktrees/residual-rl/python'
MAX_FRAME = 16 * 1024 * 1024


def read_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise EOFError('Policy connection closed before the frame was complete')
        data.extend(chunk)
    return bytes(data)


def read_frame(connection):
    size = struct.unpack('<Q', read_exact(connection, 8))[0]
    if size > MAX_FRAME:
        raise ValueError('Policy frame exceeds the size limit')
    return read_exact(connection, size)


def frame(payload):
    return struct.pack('<Q', len(payload)) + payload


def exchange(path, operation, payload=b''):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(300)
        connection.connect(str(path))
        connection.sendall(operation + frame(payload))
        failed = read_exact(connection, 1) != b'\x00'
        response = read_frame(connection)
        if failed:
            raise RuntimeError(response.decode(errors='replace'))
        return response


class PersistentPolicyClient:
    def __init__(self, socket_path, *, checkpoint, yaw_deg, device, voxel=None, **unused):
        self.socket_path = str(socket_path)
        self.info = json.loads(exchange(self.socket_path, b'I'))
        requested_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        expected = dict(checkpoint_sha256=requested_hash, device=device,
                        yaw_deg=float(yaw_deg), voxel=.0625 if voxel is None else float(voxel))
        for key, value in expected.items():
            if self.info[key] != value:
                raise ValueError(f'Policy server {key} mismatch: {self.info[key]!r} != {value!r}')

    def act(self, pos_rel, flags, force=None):
        buffer = io.BytesIO()
        np.savez(buffer, pos=np.asarray(pos_rel, np.float32), flags=np.asarray(flags, np.float32),
                 force=np.zeros(3, np.float32) if force is None else np.asarray(force, np.float32).reshape(3))
        response = exchange(self.socket_path, b'A', buffer.getvalue())
        if len(response) != 24:
            raise RuntimeError('Policy server did not return six float32 actions')
        action = np.frombuffer(response, dtype=np.float32).copy()
        if not np.isfinite(action).all():
            raise RuntimeError('Policy server returned a non-finite action')
        return action

    def close(self):
        # The dataset owns this server, not the individual rollout worker.
        pass


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', type=Path, required=True)
    p.add_argument('--status', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--warmup-episode', type=Path, required=True)
    p.add_argument('--yaw', type=float, default=267.)
    p.add_argument('--voxel', type=float, default=.0625)
    p.add_argument('--idle-timeout', type=float, default=900.)
    a = p.parse_args()
    if a.socket.exists():
        raise FileExistsError(f'Refusing to replace an existing policy socket: {a.socket}')
    sys.path[:0] = [str(PACKAGE), '/home/ge47gax/kun']
    from uipc_manip.wang_bridge import ReferencePolicy
    from uipc_manip.obs import ObsSpec
    import torch

    started = time.monotonic()
    info = dict(pid=os.getpid(), phase='warming', device='cuda', yaw_deg=a.yaw, voxel=a.voxel,
                checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
                socket_path=str(a.socket), torch=torch.__version__, calls=0, connections=0)
    write_json(a.status, info)
    policy = ReferencePolicy(a.checkpoint, device='cuda', yaw_deg=a.yaw, voxel=a.voxel)
    if not all(p.is_cuda for m in (policy.encoder, policy.trunk) for p in m.parameters()):
        raise RuntimeError('Policy parameters are not all on CUDA')
    with np.load(a.warmup_episode, allow_pickle=False) as data:
        observations = data['obs']
    spec = ObsSpec((observations.shape[1] - 7) // 7)
    pos, flags, valid, _ = spec.unpack_numpy(observations)
    for t in (0, len(observations) // 2, len(observations) - 1):
        keep = valid[t].astype(bool)
        policy.act(pos[t, keep], flags[t, keep], np.zeros(3, np.float32))
    torch.cuda.synchronize()
    info.update(warmup_s=time.monotonic() - started, parameter_device=str(next(policy.encoder.parameters()).device))
    durations = []
    last_request = time.monotonic()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(a.socket))
        os.chmod(a.socket, 0o600)
        listener.listen(8)
        listener.settimeout(30.)
        info['phase'] = 'ready'
        write_json(a.status, info)
        try:
            while time.monotonic() - last_request < a.idle_timeout:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                last_request = time.monotonic()
                info['connections'] += 1
                with connection:
                    connection.settimeout(30.)
                    try:
                        operation = read_exact(connection, 1)
                        payload = read_frame(connection)
                        if operation == b'I':
                            response = json.dumps(info).encode()
                        elif operation == b'A':
                            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                                tick = time.monotonic()
                                action = policy.act(data['pos'], data['flags'], data['force'])
                            durations.append(time.monotonic() - tick)
                            durations = durations[-100:]
                            info['calls'] += 1
                            response = np.asarray(action, np.float32).tobytes()
                        else:
                            raise ValueError('Unknown policy operation')
                        connection.sendall(b'\x00' + frame(response))
                    except Exception as exc:
                        try:
                            connection.sendall(b'\x01' + frame(repr(exc).encode()))
                        except OSError:
                            pass
                        info['last_error'] = repr(exc)
                        write_json(a.status, info)
                if info['calls'] % 50 == 0:
                    info.update(updated_unix=time.time(), median_inference_ms=float(np.median(durations) * 1000) if durations else None)
                    write_json(a.status, info)
        finally:
            a.socket.unlink(missing_ok=True)
            info['phase'] = 'stopped'
            write_json(a.status, info)


if __name__ == '__main__':
    main()
