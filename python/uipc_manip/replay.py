"""Dense replay buffer for flat point-cloud observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class FlatReplayBuffer:
    """Circular buffer holding float32 observation vectors and transitions.

    With ``priv_dim > 0`` every transition also carries the simulator's
    privileged state before and after it, for an asymmetric critic; the
    observation the actor reads is unchanged. With ``labelled`` every
    transition carries an integer label, the arm-pose region whose teacher
    Wang's distillation loss applies to it.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, capacity: int, batch_size: int, device, priv_dim: int = 0, labelled: bool = False
    ) -> None:
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
        self.priv_dim = int(priv_dim)
        self._priv = np.empty((self.capacity, self.priv_dim), dtype=np.float32)
        self._next_priv = np.empty((self.capacity, self.priv_dim), dtype=np.float32)
        self.labelled = bool(labelled)
        self._labels = np.zeros(self.capacity if self.labelled else 0, dtype=np.int64)
        self._idx = 0
        self._full = False
        self.total_added = 0

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, priv=None, next_priv=None,
        label: int | None = None,
    ) -> None:
        if self.labelled:
            if label is None:
                raise ValueError("This buffer labels every transition; pass label")
            self._labels[self._idx] = int(label)
        if self.priv_dim:
            if priv is None or next_priv is None:
                raise ValueError("This buffer stores the privileged state; pass priv and next_priv")
            np.copyto(self._priv[self._idx], np.asarray(priv, dtype=np.float32).reshape(self.priv_dim))
            np.copyto(self._next_priv[self._idx], np.asarray(next_priv, dtype=np.float32).reshape(self.priv_dim))
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
        """``(obs, action, reward, next_obs, not_done)``, then ``(priv, next_priv)`` when the buffer stores them,
        then the labels when it is labelled."""
        upper = self.size
        if upper == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        n = self.batch_size if batch_size is None else int(batch_size)
        idx = np.random.randint(0, upper, size=n)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)  # noqa: E731
        batch = (to(self._obs), to(self._actions), to(self._rewards), to(self._next_obs), to(self._not_dones))
        if self.priv_dim:
            batch += (to(self._priv), to(self._next_priv))
        if self.labelled:
            batch += (to(self._labels),)
        return batch

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        n = self.size
        order = np.arange(n) if not self._full else np.concatenate([np.arange(self._idx, n), np.arange(0, self._idx)])
        arrays = {
            "obs": self._obs[order],
            "next_obs": self._next_obs[order],
            "actions": self._actions[order],
            "rewards": self._rewards[order],
            "not_dones": self._not_dones[order],
        }
        if self.priv_dim:
            arrays.update(priv=self._priv[order], next_priv=self._next_priv[order])
        if self.labelled:
            arrays.update(labels=self._labels[order])
        np.savez_compressed(directory / "replay.npz", **arrays)
        payload = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "priv_dim": self.priv_dim,
            "labelled": self.labelled,
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
        saved_priv = int(payload.get("priv_dim", 0))
        if saved_priv != self.priv_dim:
            raise ValueError(
                f"Replay snapshot stores a {saved_priv}-float privileged state and this buffer {self.priv_dim}; "
                "a run with the other critic input cannot resume from it"
            )
        if bool(payload.get("labelled", False)) != self.labelled:
            raise ValueError("Replay snapshot and buffer disagree on region labels; a distillation run and a plain one do not resume each other")
        data = np.load(directory / "replay.npz")
        n = min(int(payload["size"]), self.capacity)
        self._obs[:n] = data["obs"][-n:]
        self._next_obs[:n] = data["next_obs"][-n:]
        self._actions[:n] = data["actions"][-n:]
        self._rewards[:n] = data["rewards"][-n:]
        self._not_dones[:n] = data["not_dones"][-n:]
        if self.priv_dim:
            self._priv[:n] = data["priv"][-n:]
            self._next_priv[:n] = data["next_priv"][-n:]
        if self.labelled:
            self._labels[:n] = data["labels"][-n:]
        self._idx = n % self.capacity
        self._full = n == self.capacity
        self.total_added = int(payload.get("total_added", n))
        return payload.get("metadata", {})


class ReplaySet:
    """Wang RSS 2023's separate replay buffers, one per garment or per arm-pose region.

    The reference keeps ``replay_buffer_num`` buffers of ``capacity // replay_buffer_num``
    transitions each (``train.py:435``), writes every transition to the buffer of its
    episode's key, and draws each update's whole batch from one buffer chosen uniformly
    (``train.py:595``, ``sample_replay_buffer_num = 1``). :meth:`sample` returns that
    buffer's batch with the buffer index appended; the index selects the batch's entropy
    temperature when the agent keeps one per buffer, and a student's region teacher reads
    the rows' labels as before. Only buffers holding more than a batch are drawn. Wang's
    student waits until every buffer does (``check_rb_min_size``), which a rotation that
    spreads its slots over the keys reaches on its first vector step.
    """

    indexed = True

    def __init__(
        self, keys, obs_dim: int, action_dim: int, capacity: int, batch_size: int, device, priv_dim: int = 0, labelled: bool = False
    ) -> None:
        self.keys = list(keys)
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError(f"A replay set needs distinct keys, got {self.keys}")
        self._index = {key: i for i, key in enumerate(self.keys)}
        self.obs_dim, self.action_dim, self.batch_size = int(obs_dim), int(action_dim), int(batch_size)
        self.priv_dim, self.labelled = int(priv_dim), bool(labelled)
        per_buffer = int(capacity) // len(self.keys)
        self.buffers = [
            FlatReplayBuffer(obs_dim, action_dim, per_buffer, batch_size, device, priv_dim=priv_dim, labelled=labelled) for _ in self.keys
        ]
        self.capacity = per_buffer * len(self.keys)

    def add(self, key, obs, action, reward, next_obs, done, priv=None, next_priv=None, label: int | None = None) -> None:
        self.buffers[self._index[key]].add(obs, action, reward, next_obs, done, priv=priv, next_priv=next_priv, label=label)

    @property
    def size(self) -> int:
        return sum(b.size for b in self.buffers)

    @property
    def total_added(self) -> int:
        return sum(b.total_added for b in self.buffers)

    def sizes(self) -> list[int]:
        return [b.size for b in self.buffers]

    def sample(self, batch_size: int | None = None):
        """One buffer's batch, as :meth:`FlatReplayBuffer.sample` returns it, then that buffer's index."""
        n = self.batch_size if batch_size is None else int(batch_size)
        ready = [i for i, b in enumerate(self.buffers) if b.size > n]
        if not ready:
            raise RuntimeError(f"No buffer of the replay set holds more than {n} transitions")
        index = ready[np.random.randint(len(ready))]
        return self.buffers[index].sample(n) + (index,)

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        directory = Path(directory)
        for i, buffer in enumerate(self.buffers):
            buffer.save(directory / f"buffer_{i:02d}", metadata=metadata)
        (directory / "replay_set.json").write_text(json.dumps({"keys": self.keys, "metadata": metadata or {}}, indent=2) + "\n")

    def load(self, directory: str | Path) -> dict:
        directory = Path(directory)
        payload = json.loads((directory / "replay_set.json").read_text())
        if list(payload["keys"]) != self.keys:
            raise ValueError(f"Replay snapshot holds buffers {payload['keys']}; this run keeps {self.keys}")
        for i, buffer in enumerate(self.buffers):
            buffer.load(directory / f"buffer_{i:02d}")
        return payload.get("metadata", {})
