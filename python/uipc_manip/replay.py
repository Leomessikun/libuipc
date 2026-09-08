"""Dense replay buffer for flat point-cloud observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class FlatReplayBuffer:
    """Circular buffer holding float32 observation vectors and transitions."""

    def __init__(self, obs_dim: int, action_dim: int, capacity: int, batch_size: int, device) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = device
        self._obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_obs = np.empty((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self._not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self._idx = 0
        self._full = False
        self.total_added = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        np.copyto(self._obs[self._idx], np.asarray(obs, dtype=np.float32).reshape(self.obs_dim))
        np.copyto(self._next_obs[self._idx], np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim))
        np.copyto(self._actions[self._idx], np.asarray(action, dtype=np.float32).reshape(self.action_dim))
        self._rewards[self._idx, 0] = float(reward)
        self._not_dones[self._idx, 0] = 0.0 if done else 1.0
        self._idx = (self._idx + 1) % self.capacity
        self._full = self._full or self._idx == 0
        self.total_added += 1

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._idx

    def sample(self, batch_size: int | None = None):
        upper = self.size
        if upper == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        n = self.batch_size if batch_size is None else int(batch_size)
        idx = np.random.randint(0, upper, size=n)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        return to(self._obs), to(self._actions), to(self._rewards), to(self._next_obs), to(self._not_dones)

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        n = self.size
        order = np.arange(n) if not self._full else np.concatenate([np.arange(self._idx, n), np.arange(0, self._idx)])
        np.savez_compressed(
            directory / "replay.npz",
            obs=self._obs[order],
            next_obs=self._next_obs[order],
            actions=self._actions[order],
            rewards=self._rewards[order],
            not_dones=self._not_dones[order],
        )
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "size": int(n),
            "total_added": int(self.total_added),
            "metadata": metadata or {},
        }
        (directory / "replay.json").write_text(json.dumps(payload, indent=2) + "\n")

    def load(self, directory: str | Path) -> dict:
        directory = Path(directory)
        payload = json.loads((directory / "replay.json").read_text())
        if int(payload["obs_dim"]) != self.obs_dim or int(payload["action_dim"]) != self.action_dim:
            raise ValueError("Replay snapshot dimensions do not match this buffer")
        data = np.load(directory / "replay.npz")
        n = min(int(payload["size"]), self.capacity)
        self._obs[:n] = data["obs"][-n:]
        self._next_obs[:n] = data["next_obs"][-n:]
        self._actions[:n] = data["actions"][-n:]
        self._rewards[:n] = data["rewards"][-n:]
        self._not_dones[:n] = data["not_dones"][-n:]
        self._idx = n % self.capacity
        self._full = n == self.capacity
        self.total_added = int(payload.get("total_added", n))
        return payload.get("metadata", {})
