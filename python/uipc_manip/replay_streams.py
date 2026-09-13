"""Episode identities for vector collectors, independent of Bellman termination."""

from __future__ import annotations

import numpy as np


class ReplayStreams:
    """Track physical steps even when a slot is excluded from replay.

    A fresh collector starts new episodes after loading replay: the old physics
    has not been restored. Evaluation/rebuild interruptions close the last valid
    transition without changing its bootstrap flag or fabricating a successor.
    """

    def __init__(self, replay, num_streams: int) -> None:
        self.replay = replay
        self.enabled = bool(replay.sequence)
        self._next_id = int(replay.next_episode_id) if self.enabled else 0
        self.episode_ids = np.empty(0, dtype=np.int64)
        self.steps = np.empty(0, dtype=np.int64)
        self.reset(num_streams)

    def close(self) -> None:
        if self.enabled:
            for stream, episode in enumerate(self.episode_ids):
                self.replay.close_episode(stream, int(episode))

    def reset(self, num_streams: int | None = None) -> None:
        """Close active streams and allocate new IDs after a physical reset."""
        n = len(self.steps) if num_streams is None else int(num_streams)
        if n < 1:
            raise ValueError("A collector needs at least one stream")
        self.close()
        self.episode_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n
        self.steps = np.zeros(n, dtype=np.int64)

    def fields(self, stream: int, episode_end: bool) -> dict:
        if not self.enabled:
            return {}
        return {
            "stream_id": int(stream), "episode_id": int(self.episode_ids[stream]),
            "episode_step": int(self.steps[stream]), "episode_end": bool(episode_end),
        }

    def advance(self, episode_ends) -> None:
        """Call once after each valid vector step, including unrecorded slots."""
        ends = np.asarray(episode_ends, dtype=bool)
        if ends.shape != self.steps.shape:
            raise ValueError("Episode-end mask must match the collector streams")
        self.steps += 1
        for stream in np.flatnonzero(ends):
            if self.enabled:
                self.replay.close_episode(int(stream), int(self.episode_ids[stream]))
            self.episode_ids[stream] = self._next_id
            self._next_id += 1
            self.steps[stream] = 0
