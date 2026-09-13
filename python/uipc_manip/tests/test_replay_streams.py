"""Collector lifecycle boundaries must survive interleaving and resets."""

import numpy as np
import pytest

pytest.importorskip("torch")

from uipc_manip.replay import FlatReplayBuffer
from uipc_manip.replay_streams import ReplayStreams


def _buffer(sequence=True):
    return FlatReplayBuffer(1, 1, 32, 4, "cpu", sequence=sequence)


def _record(replay, streams, slot, value, end=False):
    replay.add([value], [0.2], 1.0, [value + 1], False, **streams.fields(slot, end))


def test_unrecorded_steps_do_not_become_adjacent_replay_frames():
    replay = _buffer()
    streams = ReplayStreams(replay, 2)
    _record(replay, streams, 0, 0)
    _record(replay, streams, 1, 100)
    streams.advance([False, False])
    # Curriculum exclusion skips slot 0 at physical step 1.
    _record(replay, streams, 1, 101)
    streams.advance([False, False])
    _record(replay, streams, 0, 2)
    _record(replay, streams, 1, 102)
    batch = replay.sample_sequences(3, 8)
    assert (batch.stream_ids == 1).all()
    np.testing.assert_array_equal(batch.obs[0, :, 0], [100, 101, 102, 103])


def test_external_reset_closes_prior_history_without_removing_bootstrap(tmp_path):
    replay = _buffer()
    streams = ReplayStreams(replay, 1)
    _record(replay, streams, 0, 0)
    streams.advance([False])
    _record(replay, streams, 0, 1)
    streams.reset()  # Evaluation or a simulator error interrupts the episode.
    _record(replay, streams, 0, 200)
    batch = replay.sample_sequences(2, 4)
    assert batch.episode_ends[:, -1].all()
    assert (batch.not_dones == 1).all()
    np.testing.assert_array_equal(batch.obs[0, :, 0], [0, 1, 2])
    replay.save(tmp_path)
    restored = _buffer()
    restored.load(tmp_path)
    resumed = ReplayStreams(restored, 1)
    assert resumed.episode_ids[0] > streams.episode_ids[0]
    assert resumed.steps[0] == 0


def test_auto_reset_allocates_only_ended_streams_and_keeps_other_history():
    replay = _buffer()
    streams = ReplayStreams(replay, 2)
    initial = streams.episode_ids.copy()
    for slot in range(2):
        _record(replay, streams, slot, slot * 100, end=slot == 0)
    streams.advance([True, False])
    assert streams.episode_ids[0] > initial.max()
    assert streams.episode_ids[1] == initial[1]
    np.testing.assert_array_equal(streams.steps, [0, 1])
    with pytest.raises(ValueError, match="mask"):
        streams.advance([False])


def test_disabled_recording_preserves_flat_call_contract():
    replay = _buffer(sequence=False)
    streams = ReplayStreams(replay, 1)
    assert streams.fields(0, False) == {}
    _record(replay, streams, 0, 0)
    streams.advance([True])
    streams.reset()
    assert len(replay.sample()) == 5
