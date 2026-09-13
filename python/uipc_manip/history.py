"""Finite observation/command history, shared by collection and replay."""

from __future__ import annotations

import numpy as np


class RolloutHistory:
    """The streaming counterpart of a padded replay window.

    State is the raw prefix, not encoded features. A weight update therefore
    cannot leave it stale: every decision re-encodes the retained frames under
    the current parameters, which is what makes streaming and window evaluation
    comparable at all. The cost is `length` encoder passes per decision.

    At decision `t` a stream holds observations `t-length+1..t` and the commands
    that followed the earlier ones, `t-length+1..t-1`. Before enough steps have
    elapsed the prefix is shorter; the missing frames stay zero and the mask
    reports them, exactly as `sample_sequences(..., pad=True)` reports them.
    """

    def __init__(self, num_streams: int, obs_dim: int, action_dim: int, length: int) -> None:
        self.num_streams, self.length = int(num_streams), int(length)
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        if self.num_streams < 1:
            raise ValueError("A rollout history needs at least one stream")
        if self.length < 1:
            raise ValueError("History length must be positive; 1 is the single-frame policy")
        self._obs = np.zeros((self.num_streams, self.length - 1, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.num_streams, self.length - 1, self.action_dim), dtype=np.float32)
        self._filled = np.zeros(self.num_streams, dtype=np.int64)

    def reset(self, streams=None) -> None:
        """Forget the prefix of the given streams; the default resets all of them.

        Call this on every physical discontinuity: episode end, world rebuild,
        slot reassignment and resume into fresh physics. A retained prefix from
        different physics is not a shorter history, it is a wrong one.
        """
        if streams is None:
            rows = slice(None)
        else:
            rows = np.asarray(streams)
            if rows.dtype == bool:
                if rows.shape != (self.num_streams,):
                    raise ValueError("A reset mask must cover every stream")
                rows = np.flatnonzero(rows)
            rows = rows.astype(np.int64, copy=False).reshape(-1)
            if rows.size and (rows.min() < 0 or rows.max() >= self.num_streams):
                raise IndexError("Reset stream out of range")
        self._obs[rows] = 0.0
        self._actions[rows] = 0.0
        self._filled[rows] = 0

    def window(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(obs[N,length,D], actions[N,length-1,A], valid[N,length])` for this decision.

        The current observation occupies the last position, which is always
        valid. A command is valid exactly where its observation is, because a
        retained prefix is contiguous.
        """
        obs = np.asarray(obs, dtype=np.float32).reshape(self.num_streams, self.obs_dim)
        window = np.concatenate([self._obs, obs[:, None]], axis=1)
        positions = np.arange(self.length, dtype=np.int64)
        valid = positions[None] >= (self.length - 1 - self._filled)[:, None]
        return window, self._actions.copy(), valid

    def push(self, obs: np.ndarray, action: np.ndarray) -> None:
        """Record the decision just taken, so the next window ends one step later."""
        obs = np.asarray(obs, dtype=np.float32).reshape(self.num_streams, self.obs_dim)
        action = np.asarray(action, dtype=np.float32).reshape(self.num_streams, self.action_dim)
        if self.length > 1:
            self._obs[:, :-1] = self._obs[:, 1:]
            self._actions[:, :-1] = self._actions[:, 1:]
            self._obs[:, -1] = obs
            self._actions[:, -1] = action
        np.minimum(self._filled + 1, self.length - 1, out=self._filled)
