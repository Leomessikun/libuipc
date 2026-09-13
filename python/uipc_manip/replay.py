"""Dense replay buffer for flat point-cloud observations."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


SEQUENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SequenceBatch:
    """Windows: observations/states have L+1 steps; other tensors have L.

    Rewards and bootstrap masks retain their trailing singleton dimension.
    Identity, end and label tensors have shape [batch, length]. The final
    observation is the saved pre-reset successor of the final transition.

    `valid` marks recorded transitions. A padded window begins before its
    episode did: its leading positions carry zeros, identity -1 and valid
    False, which is the same empty history a deployed policy starts from.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    not_dones: torch.Tensor
    episode_ends: torch.Tensor
    stream_ids: torch.Tensor
    episode_ids: torch.Tensor
    episode_steps: torch.Tensor
    valid: torch.Tensor
    priv: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    buffer_index: int | None = None


class _IndexPool:
    """Uniformly sampled integer set with constant-time insertion/removal."""

    def __init__(self, capacity: int, indices=()) -> None:
        self.values = list(map(int, indices))
        self.positions = np.full(capacity, -1, dtype=np.int64)
        if self.values:
            self.positions[self.values] = np.arange(len(self.values))

    def add(self, index: int) -> None:
        if self.positions[index] < 0:
            self.positions[index] = len(self.values)
            self.values.append(index)

    def discard(self, index: int) -> None:
        position = int(self.positions[index])
        if position < 0:
            return
        last = self.values.pop()
        if position < len(self.values):
            self.values[position] = last
            self.positions[last] = position
        self.positions[index] = -1


def _nonnegative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if result < 0 or result > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must fit a nonnegative int64")
    return result


class FlatReplayBuffer:
    """Circular buffer holding float32 observation vectors and transitions.

    With ``priv_dim > 0`` every transition also carries the simulator's
    privileged state before and after it, for an asymmetric critic; the
    observation the actor reads is unchanged. With ``labelled`` every
    transition carries an integer label, the arm-pose region whose teacher
    Wang's distillation loss applies to it.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, capacity: int, batch_size: int, device, priv_dim: int = 0, labelled: bool = False, sequence: bool = False
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.device = device
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.sequence = bool(sequence)
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
        self._next_episode_id = 0
        if self.sequence:
            self._stream_ids = np.empty(self.capacity, dtype=np.int64)
            self._episode_ids = np.empty(self.capacity, dtype=np.int64)
            self._episode_steps = np.empty(self.capacity, dtype=np.int64)
            self._episode_ends = np.empty(self.capacity, dtype=bool)
            self._prev = np.full(self.capacity, -1, dtype=np.int64)
            self._next = np.full(self.capacity, -1, dtype=np.int64)
            self._tails: dict[tuple[int, int], int] = {}
            self._stream_tails: dict[int, int] = {}
            self._windows: dict[int, _IndexPool] = {}
            self._strict_windows: dict[int, _IndexPool] = {}

    def add(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool, priv=None, next_priv=None,
        label: int | None = None, *, stream_id: int | None = None, episode_id: int | None = None,
        episode_step: int | None = None, episode_end: bool | None = None,
    ) -> None:
        identity = None
        if self.sequence:
            identity = tuple(_nonnegative_int(value, name) for value, name in (
                (stream_id, "stream_id"), (episode_id, "episode_id"), (episode_step, "episode_step")))
            if not isinstance(episode_end, (bool, np.bool_)):
                raise ValueError("sequence replay requires boolean episode_end")
            if done and not episode_end:
                raise ValueError("A Bellman terminal transition must also end its episode")
            previous = self._tails.get(identity[:2])
            if previous is not None and identity[2] <= self._episode_steps[previous]:
                raise ValueError("Episode steps must increase within a stream and episode")
        elif any(value is not None for value in (stream_id, episode_id, episode_step, episode_end)):
            raise ValueError("Episode metadata requires sequence=True")
        # Validate all row shapes before mutating ring links or existing data.
        obs = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        reward = float(reward)
        if self.priv_dim:
            if priv is None or next_priv is None:
                raise ValueError("This buffer stores the privileged state; pass priv and next_priv")
            priv = np.asarray(priv, dtype=np.float32).reshape(self.priv_dim)
            next_priv = np.asarray(next_priv, dtype=np.float32).reshape(self.priv_dim)
        if self.labelled and label is not None:
            label = int(label)
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
        if identity is not None:
            self._append_identity(identity, bool(episode_end))
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

    @property
    def next_episode_id(self) -> int:
        """Unused global ID, retaining the historical high-water mark across eviction."""
        return self._next_episode_id

    def _invalidate_followers(self, index: int, *, include_self: bool) -> None:
        for length, pool, _ in self._window_pools():
            current = index if include_self else int(self._next[index])
            for _ in range(length if include_self else length - 1):
                if current < 0:
                    break
                pool.discard(current)
                current = int(self._next[current])

    def _window_pools(self):
        for length, pool in self._windows.items():
            yield length, pool, False
        for length, pool in self._strict_windows.items():
            yield length, pool, True

    def _append_identity(self, identity: tuple[int, int, int], episode_end: bool) -> None:
        index = self._idx
        if self._full:
            self._invalidate_followers(index, include_self=True)
            before, after = int(self._prev[index]), int(self._next[index])
            if before >= 0:
                self._next[before] = -1
            if after >= 0:
                self._prev[after] = -1
            old_key = (int(self._stream_ids[index]), int(self._episode_ids[index]))
            if self._tails.get(old_key) == index:
                del self._tails[old_key]
            if self._stream_tails.get(old_key[0]) == index:
                del self._stream_tails[old_key[0]]
        self._prev[index] = self._next[index] = -1
        stream, episode, step = identity
        previous = self._tails.get((stream, episode), -1)
        if (previous >= 0 and self._stream_tails.get(stream) == previous
                and not self._episode_ends[previous] and self._episode_steps[previous] + 1 == step):
            self._prev[index] = previous
            self._next[previous] = index
        self._stream_ids[index], self._episode_ids[index], self._episode_steps[index] = identity
        self._episode_ends[index] = episode_end
        self._tails[(stream, episode)] = index
        self._stream_tails[stream] = index
        self._next_episode_id = max(self._next_episode_id, episode + 1)
        for length, pool, opening_allowed in self._window_pools():
            current = index
            natural_opening = opening_allowed and self._episode_steps[current] == 0
            for _ in range(length - 1):
                current = int(self._prev[current])
                if current < 0:
                    break
                natural_opening |= opening_allowed and self._episode_steps[current] == 0
            if current >= 0 or natural_opening:
                pool.add(index)

    def close_episode(self, stream_id: int, episode_id: int) -> None:
        """Close the last retained row without changing its Bellman bootstrap mask."""
        if not self.sequence:
            raise ValueError("Closing episodes requires sequence=True")
        key = (_nonnegative_int(stream_id, "stream_id"), _nonnegative_int(episode_id, "episode_id"))
        index = self._tails.get(key)
        if index is not None:
            self._invalidate_followers(index, include_self=False)
            after = int(self._next[index])
            if after >= 0:
                self._prev[after] = -1
            self._next[index] = -1
            self._episode_ends[index] = True

    def _sequence_endpoints(self, length: int) -> _IndexPool:
        if not self.sequence:
            raise ValueError("Sequence sampling requires sequence=True")
        length = _nonnegative_int(length, "length")
        if length == 0:
            raise ValueError("Sequence length must be positive")
        if length > self.capacity:
            raise RuntimeError("No complete sequence fits this replay capacity")
        if length not in self._windows:
            # One vectorized scan per requested length; subsequent appends/overwrites
            # maintain its candidate set in O(length), not O(replay capacity).
            ends = np.arange(self.size)
            current = ends.copy()
            valid = np.ones(self.size, dtype=bool)
            for _ in range(length - 1):
                current = self._prev[np.maximum(current, 0)]
                valid &= current >= 0
            self._windows[length] = _IndexPool(self.capacity, ends[valid])
        return self._windows[length]

    def _strict_endpoints(self, length: int) -> _IndexPool:
        """Full context or a genuine episode opening, never an evicted/gapped prefix."""
        length = _nonnegative_int(length, "length")
        if length < 1 or not self.sequence:
            raise ValueError("Strict context needs sequence=True and a positive length")
        if length not in self._strict_windows:
            ends = np.arange(self.size)
            current = ends.copy()
            counts = np.ones(self.size, dtype=np.int64)
            active = np.ones(self.size, dtype=bool)
            for _ in range(length - 1):
                previous = self._prev[np.maximum(current, 0)]
                active &= previous >= 0
                counts += active
                current = np.where(active, previous, current)
            valid = (counts == length) | (counts == self._episode_steps[ends] + 1)
            self._strict_windows[length] = _IndexPool(self.capacity, ends[valid])
        return self._strict_windows[length]

    def sequence_ready(self, length: int, *, pad: bool = False, strict_context: bool = False) -> bool:
        """Whether a window of this length can be drawn under the requested padding."""
        if strict_context:
            if not pad:
                raise ValueError("strict_context requires pad=True")
            return bool(self._strict_endpoints(length).values)
        if pad:
            _nonnegative_int(length, "length")
            if not self.sequence:
                raise ValueError("Sequence sampling requires sequence=True")
            return self.size > 0
        return bool(self._sequence_endpoints(length).values)

    def _window_indices(self, length: int, n: int, pad: bool, strict_context: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Endpoint-uniform windows walked back through the episode links.

        Without padding, endpoints are restricted to rows holding a complete
        prefix. With padding, every retained row is an endpoint and a shorter
        prefix is reported through the mask rather than silently extended into
        an earlier episode. Padded columns repeat their successor's row so the
        gathers stay in bounds; their content is discarded by the caller.
        """
        indices = np.empty((n, length), dtype=np.int64)
        valid = np.zeros((n, length), dtype=bool)
        if strict_context:
            pool = self._strict_endpoints(length)
            if not pool.values:
                raise RuntimeError("No endpoint has its required context or a genuine episode opening")
            indices[:, -1] = np.take(pool.values, np.random.randint(len(pool.values), size=n))
        elif pad:
            if self.size == 0:
                raise RuntimeError("An empty replay contains no sequence window")
            indices[:, -1] = np.random.randint(self.size, size=n)
        else:
            pool = self._sequence_endpoints(length)
            if not pool.values:
                raise RuntimeError("No complete sequence of the requested length is available")
            indices[:, -1] = np.take(pool.values, np.random.randint(len(pool.values), size=n))
        valid[:, -1] = True
        for column in range(length - 2, -1, -1):
            previous = self._prev[indices[:, column + 1]]
            present = valid[:, column + 1] & (previous >= 0)
            indices[:, column] = np.where(present, previous, indices[:, column + 1])
            valid[:, column] = present
        return indices, valid

    def sample_sequences(self, length: int, batch_size: int | None = None, *, pad: bool = False,
                         strict_context: bool = False) -> SequenceBatch:
        """Sample same-episode windows uniformly over their endpoints, with replacement.

        `pad=False` returns only complete windows. `pad=True` also draws windows
        whose episode began fewer than `length` steps earlier, left-padding them:
        every recorded transition then becomes a learning step, including the
        episode openings a deployed policy always has to act through.
        """
        if not self.sequence:
            raise ValueError("Sequence sampling requires sequence=True")
        length = _nonnegative_int(length, "length")
        if length == 0:
            raise ValueError("Sequence length must be positive")
        n = self.batch_size if batch_size is None else _nonnegative_int(batch_size, "batch_size")
        if n <= 0:
            raise ValueError("Sequence batch size must be positive")
        if strict_context and not pad:
            raise ValueError("strict_context requires pad=True")
        indices, valid = self._window_indices(length, n, pad, strict_context)
        blank = ~valid
        to = lambda value: torch.as_tensor(value, device=self.device)  # noqa: E731

        def frames(current: np.ndarray, successor: np.ndarray) -> torch.Tensor:
            window = np.concatenate([current[indices], successor[indices[:, -1], None]], axis=1)
            window[:, :length][blank] = 0.0
            return to(window)

        def steps(values: np.ndarray, empty) -> torch.Tensor:
            window = values[indices].copy()
            window[blank] = empty
            return to(window)

        return SequenceBatch(
            obs=frames(self._obs, self._next_obs), actions=steps(self._actions, 0.0),
            rewards=steps(self._rewards, 0.0), not_dones=steps(self._not_dones, 0.0),
            episode_ends=steps(self._episode_ends, False), stream_ids=steps(self._stream_ids, -1),
            episode_ids=steps(self._episode_ids, -1), episode_steps=steps(self._episode_steps, -1),
            valid=to(valid), priv=frames(self._priv, self._next_priv) if self.priv_dim else None,
            labels=steps(self._labels, -1) if self.labelled else None,
        )

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
        if self.sequence:
            arrays.update(stream_ids=self._stream_ids[order], episode_ids=self._episode_ids[order],
                          episode_steps=self._episode_steps[order], episode_ends=self._episode_ends[order])
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
        if self.sequence:
            payload.update(sequence=True, sequence_schema_version=SEQUENCE_SCHEMA_VERSION,
                           next_episode_id=self.next_episode_id)
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
        if bool(payload.get("sequence", False)) != self.sequence:
            raise ValueError("Replay snapshot and buffer disagree on sequence metadata")
        if self.sequence and payload.get("sequence_schema_version") != SEQUENCE_SCHEMA_VERSION:
            raise ValueError("Unsupported sequence replay schema version")
        with np.load(directory / "replay.npz") as data:
            saved_size = int(payload["size"])
            n = min(saved_size, self.capacity)
            retained = slice(saved_size - n, saved_size)
            self._obs[:n] = data["obs"][retained]
            self._next_obs[:n] = data["next_obs"][retained]
            self._actions[:n] = data["actions"][retained]
            self._rewards[:n] = data["rewards"][retained]
            self._not_dones[:n] = data["not_dones"][retained]
            if self.priv_dim:
                self._priv[:n] = data["priv"][retained]
                self._next_priv[:n] = data["next_priv"][retained]
            if self.labelled:
                self._labels[:n] = data["labels"][retained]
            if self.sequence:
                names = ("stream_ids", "episode_ids", "episode_steps", "episode_ends")
                for name in names:
                    if name not in data:
                        raise ValueError(f"Sequence replay snapshot is missing {name}")
                saved_identity = {name: data[name] for name in names}
                if any(value.shape != (saved_size,) for value in saved_identity.values()):
                    raise ValueError("Sequence replay metadata has invalid dimensions")
                for name in names[:3]:
                    values = saved_identity[name]
                    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
                        raise ValueError(f"Invalid nonnegative integer metadata in {name}")
                if saved_identity["episode_ends"].dtype != np.bool_:
                    raise ValueError("Invalid boolean episode end metadata")
                self._prev.fill(-1)
                self._next.fill(-1)
                self._tails.clear()
                self._stream_tails.clear()
                self._windows.clear()
                self._strict_windows.clear()
                self._full = False
                self._next_episode_id = max(int(payload.get("next_episode_id", 0)),
                                            int(saved_identity["episode_ids"].max(initial=-1)) + 1)
                for index in range(n):
                    self._idx = index
                    identity = tuple(_nonnegative_int(saved_identity[name][retained][index], name) for name in names[:3])
                    end = saved_identity["episode_ends"][retained][index]
                    if not isinstance(end, (bool, np.bool_)) or (self._not_dones[index, 0] == 0 and not end):
                        raise ValueError("Invalid episode end metadata in sequence snapshot")
                    previous = self._tails.get(identity[:2])
                    if previous is not None and identity[2] <= self._episode_steps[previous]:
                        raise ValueError("Non-increasing episode steps in sequence snapshot")
                    self._append_identity(identity, bool(end))
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
        self, keys, obs_dim: int, action_dim: int, capacity: int, batch_size: int, device, priv_dim: int = 0, labelled: bool = False, sequence: bool = False
    ) -> None:
        self.keys = list(keys)
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError(f"A replay set needs distinct keys, got {self.keys}")
        self._index = {key: i for i, key in enumerate(self.keys)}
        self.obs_dim, self.action_dim, self.batch_size = int(obs_dim), int(action_dim), int(batch_size)
        self.priv_dim, self.labelled = int(priv_dim), bool(labelled)
        self.sequence = bool(sequence)
        per_buffer = int(capacity) // len(self.keys)
        self.buffers = [
            FlatReplayBuffer(obs_dim, action_dim, per_buffer, batch_size, device, priv_dim=priv_dim, labelled=labelled, sequence=sequence) for _ in self.keys
        ]
        self.capacity = per_buffer * len(self.keys)

    def add(self, key, obs, action, reward, next_obs, done, priv=None, next_priv=None, label: int | None = None,
            *, stream_id: int | None = None, episode_id: int | None = None,
            episode_step: int | None = None, episode_end: bool | None = None) -> None:
        self.buffers[self._index[key]].add(obs, action, reward, next_obs, done, priv=priv, next_priv=next_priv, label=label,
                                          stream_id=stream_id, episode_id=episode_id, episode_step=episode_step,
                                          episode_end=episode_end)

    @property
    def next_episode_id(self) -> int:
        return max(buffer.next_episode_id for buffer in self.buffers)

    def close_episode(self, stream_id: int, episode_id: int) -> None:
        for buffer in self.buffers:
            buffer.close_episode(stream_id, episode_id)

    def sample_sequences(self, length: int, batch_size: int | None = None, *, pad: bool = False,
                         strict_context: bool = False) -> SequenceBatch:
        """Draw one complete batch from one uniformly chosen sequence-ready buffer.

        Padded sampling keeps :meth:`sample`'s buffer rule, uniform over the buffers holding more
        than a batch, so a history run and a single-frame run draw garments and temperatures alike.
        """
        n = self.batch_size if batch_size is None else int(batch_size)
        ready = [i for i, buffer in enumerate(self.buffers)
                 if buffer.sequence_ready(length, pad=pad, strict_context=strict_context) and (not pad or buffer.size > n)]
        if not ready:
            raise RuntimeError("No replay buffer contains a sequence window of the requested length")
        index = int(ready[np.random.randint(len(ready))])
        return replace(self.buffers[index].sample_sequences(length, batch_size, pad=pad, strict_context=strict_context), buffer_index=index)

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
