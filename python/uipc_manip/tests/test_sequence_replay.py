"""CPU checks for episode-safe replay windows and resumable metadata."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip.replay import FlatReplayBuffer, ReplaySet  # noqa: E402


def _buffer(capacity=32, batch_size=4, **kwargs):
    return FlatReplayBuffer(3, 2, capacity, batch_size, "cpu", sequence=True, **kwargs)


def _add(replay, step, *, stream=0, episode=0, end=False, done=False, key=None):
    metadata = dict(stream_id=stream, episode_id=episode, episode_step=step, episode_end=end)
    if replay.priv_dim:
        metadata.update(priv=[step + 100], next_priv=[step + 101])
    if replay.labelled:
        metadata["label"] = episode + 7
    args = ([stream, episode, step], [step, -step], step + 0.25, [stream, episode, step + 1], done)
    replay.add(*args, **metadata) if key is None else replay.add(key, *args, **metadata)


def _assert_aligned(batch, batch_size, length, *, priv=False, labelled=False):
    assert batch.obs.shape == (batch_size, length + 1, 3)
    assert batch.actions.shape == (batch_size, length, 2)
    assert batch.rewards.shape == batch.not_dones.shape == (batch_size, length, 1)
    for name in ("episode_ends", "stream_ids", "episode_ids", "episode_steps"):
        assert getattr(batch, name).shape == (batch_size, length)
    assert batch.episode_ends.dtype == torch.bool
    assert batch.stream_ids.dtype == batch.episode_ids.dtype == batch.episode_steps.dtype == torch.int64
    assert batch.obs.dtype == batch.actions.dtype == batch.rewards.dtype == torch.float32
    assert batch.obs.device.type == "cpu"
    assert not batch.episode_ends[:, :-1].any()
    torch.testing.assert_close(batch.episode_steps[:, 1:], batch.episode_steps[:, :-1] + 1)
    torch.testing.assert_close(batch.stream_ids, batch.stream_ids[:, :1].expand(-1, length))
    torch.testing.assert_close(batch.episode_ids, batch.episode_ids[:, :1].expand(-1, length))
    torch.testing.assert_close(batch.obs[:, :-1, 0], batch.stream_ids.float())
    torch.testing.assert_close(batch.obs[:, :-1, 1], batch.episode_ids.float())
    torch.testing.assert_close(batch.obs[:, :-1, 2], batch.episode_steps.float())
    torch.testing.assert_close(batch.obs[:, -1, :2], batch.obs[:, -2, :2])
    torch.testing.assert_close(batch.obs[:, -1, 2], batch.episode_steps[:, -1].float() + 1)
    torch.testing.assert_close(batch.actions[:, :, 0], batch.episode_steps.float())
    torch.testing.assert_close(batch.actions[:, :, 1], -batch.episode_steps.float())
    torch.testing.assert_close(batch.rewards[:, :, 0], batch.episode_steps.float() + 0.25)
    if priv:
        assert batch.priv.shape == (batch_size, length + 1, 1)
        torch.testing.assert_close(batch.priv[:, :, 0], batch.obs[:, :, 2] + 100)
    else:
        assert batch.priv is None
    if labelled:
        assert batch.labels.shape == (batch_size, length)
        torch.testing.assert_close(batch.labels, batch.episode_ids + 7)
    else:
        assert batch.labels is None


@pytest.mark.parametrize("length", [1, 2, 4])
def test_interleaved_vector_slots_preserve_time_and_identity(length):
    replay = _buffer()
    for step in range(4):
        for stream in (0, 1, 2):
            # Episode IDs are only unique together with their stream IDs.
            _add(replay, step, stream=stream, episode=9, end=step == 3)
    batch = replay.sample_sequences(length, batch_size=32)
    _assert_aligned(batch, 32, length)
    assert batch.buffer_index is None
    assert (batch.not_dones == 1).all()


@pytest.mark.parametrize("terminal", [False, True])
def test_episode_end_is_separate_from_bellman_terminal(terminal):
    replay = _buffer()
    _add(replay, 0, episode=10)
    _add(replay, 1, episode=10, end=True, done=terminal)
    _add(replay, 0, episode=11, end=True)
    batch = replay.sample_sequences(2)
    _assert_aligned(batch, 4, 2)
    assert batch.episode_ends[:, -1].all()
    assert (batch.not_dones[:, 0] == 1).all()
    assert (batch.not_dones[:, -1] == (0 if terminal else 1)).all()
    with pytest.raises(RuntimeError):
        replay.sample_sequences(3)


def test_gaps_and_closed_segments_do_not_form_a_window():
    replay = _buffer()
    _add(replay, 0)
    _add(replay, 2)
    _add(replay, 3, end=True)
    _add(replay, 4)
    batch = replay.sample_sequences(2)
    _assert_aligned(batch, 4, 2)
    assert batch.episode_steps.tolist() == [[2, 3]] * 4
    with pytest.raises(RuntimeError):
        replay.sample_sequences(3)
    for step in (4, 1):
        with pytest.raises(ValueError):
            _add(replay, step)
    assert replay.size == replay.total_added == 4


def test_intervening_episode_on_same_stream_breaks_old_episode_link():
    replay = _buffer()
    _add(replay, 0, episode=1)
    _add(replay, 0, episode=2)
    _add(replay, 1, episode=1)
    with pytest.raises(RuntimeError):
        replay.sample_sequences(2)
    _add(replay, 2, episode=1)
    batch = replay.sample_sequences(2)
    _assert_aligned(batch, 4, 2)
    assert batch.episode_steps.tolist() == [[1, 2]] * 4


@pytest.mark.parametrize("length", [0, -1])
def test_nonpositive_window_length_rejected(length):
    with pytest.raises(ValueError):
        _buffer().sample_sequences(length)


def test_no_padding_or_cross_episode_stitching():
    replay = _buffer()
    with pytest.raises(RuntimeError):
        replay.sample_sequences(1)
    for episode in range(8):
        _add(replay, 0, episode=episode, end=True)
    with pytest.raises(RuntimeError):
        replay.sample_sequences(2)


@pytest.mark.parametrize("field", ["stream_id", "episode_id", "episode_step", "episode_end"])
def test_sequence_metadata_is_required(field):
    replay = _buffer()
    metadata = dict(stream_id=0, episode_id=0, episode_step=0, episode_end=False)
    del metadata[field]
    with pytest.raises(ValueError):
        replay.add([0, 0, 0], [0, 0], 0, [0, 0, 1], False, **metadata)
    assert replay.size == replay.total_added == 0


@pytest.mark.parametrize("field", ["stream_id", "episode_id", "episode_step"])
@pytest.mark.parametrize("value", [-1, 1.5])
def test_invalid_sequence_identity_rejected(field, value):
    replay = _buffer()
    metadata = dict(stream_id=0, episode_id=0, episode_step=0, episode_end=False)
    metadata[field] = value
    with pytest.raises(ValueError):
        replay.add([0, 0, 0], [0, 0], 0, [0, 0, 1], False, **metadata)
    assert replay.size == replay.total_added == 0


def test_terminal_must_end_episode_and_flat_mode_rejects_metadata():
    replay = _buffer()
    with pytest.raises(ValueError):
        _add(replay, 0, done=True, end=False)
    assert replay.size == 0
    flat = FlatReplayBuffer(3, 2, 8, 2, "cpu")
    with pytest.raises(ValueError):
        _add(flat, 0)
    with pytest.raises((ValueError, RuntimeError)):
        flat.sample_sequences(1)


def test_ring_overwrite_does_not_link_stale_slots():
    replay = _buffer(capacity=5)
    for step in range(5):
        _add(replay, step)
    # The retained physical rows now wrap: [step 5, step 6, step 2, step 3, step 4].
    _add(replay, 5)
    _add(replay, 6)
    batch = replay.sample_sequences(5)
    _assert_aligned(batch, 4, 5)
    assert batch.episode_steps.tolist() == [[2, 3, 4, 5, 6]] * 4
    _add(replay, 0, stream=1, episode=1, end=True)
    with pytest.raises(RuntimeError):
        replay.sample_sequences(5)
    batch = replay.sample_sequences(4)
    assert batch.episode_steps.tolist() == [[3, 4, 5, 6]] * 4
    _assert_aligned(batch, 4, 4)


def test_cached_windows_update_after_interleaved_overwrites():
    replay = _buffer(capacity=6)
    for step in range(3):
        for stream in (0, 1):
            _add(replay, step, stream=stream)
    for length in (1, 2, 3):
        _assert_aligned(replay.sample_sequences(length), 4, length)
    for step in range(3, 8):
        for stream in (0, 1):
            _add(replay, step, stream=stream)
            for length in (1, 2, 3):
                batch = replay.sample_sequences(length, batch_size=16)
                _assert_aligned(batch, 16, length)
                oldest = step - 2 - ((stream == 0) & (batch.stream_ids[:, 0] == 1)).long()
                assert (batch.episode_steps[:, 0] >= oldest).all()
    # New short episodes evict every old window; cached candidates must disappear.
    for episode in range(1, 7):
        _add(replay, 0, episode=episode, end=True)
    with pytest.raises(RuntimeError):
        replay.sample_sequences(2)


def test_randomized_ring_windows_match_retained_episode_history(monkeypatch):
    rng = np.random.default_rng(6023)
    replay = _buffer(capacity=11)
    retained = []
    episodes, steps = [0] * 3, [0] * 3

    def cycle_indices(low, high=None, size=None):
        low, high = (0, low) if high is None else (low, high)
        return low + np.arange(size) % (high - low)

    # Cover every candidate deterministically without depending on random luck.
    monkeypatch.setattr(np.random, "randint", cycle_indices)
    for _ in range(100):
        stream = int(rng.integers(3))
        steps[stream] += int(rng.random() < 0.15)
        episode, step = episodes[stream], steps[stream]
        end = bool(rng.random() < 0.18)
        _add(replay, step, stream=stream, episode=episode, end=end)
        retained = (retained + [(stream, episode, step, end)])[-replay.capacity :]
        if end:
            episodes[stream] += 1
            steps[stream] = 0
        else:
            steps[stream] += 1

        histories = {}
        for row in retained:
            histories.setdefault(row[:2], []).append(row)
        for length in (1, 2, 3, 4):
            expected = set()
            for history in histories.values():
                for start in range(len(history) - length + 1):
                    rows = history[start : start + length]
                    if any(left[3] or right[2] != left[2] + 1 for left, right in zip(rows, rows[1:])):
                        continue
                    expected.add(tuple(row[:3] for row in rows))
            if not expected:
                with pytest.raises(RuntimeError):
                    replay.sample_sequences(length)
                continue
            batch = replay.sample_sequences(length, batch_size=replay.capacity)
            _assert_aligned(batch, replay.capacity, length)
            sampled = set()
            for streams, ids, positions in zip(batch.stream_ids.tolist(), batch.episode_ids.tolist(), batch.episode_steps.tolist()):
                sampled.add(tuple(zip(streams, ids, positions)))
            assert sampled == expected


def test_close_episode_keeps_bootstrap_and_only_closes_selected_stream():
    replay = _buffer()
    for step in range(2):
        for stream in (0, 1):
            _add(replay, step, stream=stream, episode=4)
    replay.close_episode(stream_id=0, episode_id=4)
    replay.close_episode(stream_id=99, episode_id=99)
    for stream in (0, 1):
        _add(replay, 2, stream=stream, episode=4)
    batch = replay.sample_sequences(3)
    _assert_aligned(batch, 4, 3)
    assert (batch.stream_ids == 1).all()
    batch = replay.sample_sequences(2, batch_size=32)
    _assert_aligned(batch, 32, 2)
    assert (batch.not_dones == 1).all()


def test_save_load_shrink_retains_recent_windows_and_episode_highwater(tmp_path):
    replay = _buffer(capacity=6, priv_dim=1, labelled=True)
    _add(replay, 0, episode=900, end=True)
    for step in range(8):
        _add(replay, step, episode=7, end=step == 7)
    assert replay.next_episode_id == 901
    replay.save(tmp_path, metadata={"run": "sequence-test"})
    snapshot = json.loads((tmp_path / "replay.json").read_text())
    assert snapshot["sequence"] is True
    restored = _buffer(capacity=3, priv_dim=1, labelled=True)
    assert restored.load(tmp_path) == {"run": "sequence-test"}
    assert restored.size == 3 and restored.total_added == 9
    assert restored.next_episode_id == 901
    batch = restored.sample_sequences(3)
    _assert_aligned(batch, 4, 3, priv=True, labelled=True)
    assert batch.episode_steps.tolist() == [[5, 6, 7]] * 4
    assert batch.episode_ends[:, -1].all()
    _add(restored, 0, episode=restored.next_episode_id, end=True)
    assert restored.next_episode_id == 902
    with pytest.raises(RuntimeError):
        restored.sample_sequences(3)


def test_empty_sequence_snapshot_roundtrip(tmp_path):
    replay = _buffer()
    replay.save(tmp_path)
    restored = _buffer(capacity=3)
    assert restored.load(tmp_path) == {}
    assert restored.size == restored.total_added == restored.next_episode_id == 0
    with pytest.raises(RuntimeError):
        restored.sample_sequences(1)


def test_flat_and_sequence_snapshots_are_not_interchangeable(tmp_path):
    flat = FlatReplayBuffer(3, 2, 8, 2, "cpu")
    flat.add([0, 0, 0], [0, 0], 0, [0, 0, 1], False)
    flat.save(tmp_path / "flat")
    with pytest.raises(ValueError):
        _buffer().load(tmp_path / "flat")
    sequence = _buffer()
    _add(sequence, 0)
    sequence.save(tmp_path / "sequence")
    with pytest.raises(ValueError):
        flat.load(tmp_path / "sequence")


@pytest.mark.parametrize("version", [None, 999])
def test_unknown_or_missing_sequence_schema_rejected(tmp_path, version):
    replay = _buffer()
    _add(replay, 0)
    replay.save(tmp_path)
    path = tmp_path / "replay.json"
    payload = json.loads(path.read_text())
    if version is None:
        del payload["sequence_schema_version"]
    else:
        payload["sequence_schema_version"] = version
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="schema"):
        _buffer().load(tmp_path)


@pytest.mark.parametrize("corruption", ["missing_ids", "negative_step", "duplicate_step", "terminal_without_end"])
def test_invalid_snapshot_sequence_metadata_rejected(tmp_path, corruption):
    replay = _buffer()
    _add(replay, 0)
    _add(replay, 1, end=True, done=True)
    replay.save(tmp_path)
    with np.load(tmp_path / "replay.npz") as data:
        arrays = {name: data[name].copy() for name in data.files}
    if corruption == "missing_ids":
        del arrays["stream_ids"]
    elif corruption == "negative_step":
        arrays["episode_steps"][0] = -1
    elif corruption == "duplicate_step":
        arrays["episode_steps"][1] = 0
    else:
        arrays["episode_ends"][1] = False
    np.savez_compressed(tmp_path / "replay.npz", **arrays)
    with pytest.raises(ValueError):
        _buffer().load(tmp_path)


def test_replay_set_sequences_preserve_labels_and_selected_buffer(tmp_path):
    replay = ReplaySet(["a", "b"], 3, 2, 16, 3, "cpu", priv_dim=1, labelled=True, sequence=True)
    for episode in range(4):
        _add(replay, 0, episode=episode, end=True, key="a")
    # Only b can supply full two-step windows, despite a holding more rows.
    _add(replay, 0, stream=2, episode=50, key="b")
    _add(replay, 1, stream=2, episode=50, key="b")
    batch = replay.sample_sequences(2)
    _assert_aligned(batch, 3, 2, priv=True, labelled=True)
    assert batch.buffer_index == 1
    assert replay.next_episode_id == 51
    replay.close_episode(stream_id=2, episode_id=50)
    assert replay.sample_sequences(2).episode_ends[:, -1].all()
    replay.save(tmp_path, metadata={"group": "garment"})
    restored = ReplaySet(["a", "b"], 3, 2, 16, 3, "cpu", priv_dim=1, labelled=True, sequence=True)
    assert restored.load(tmp_path) == {"group": "garment"}
    assert restored.next_episode_id == 51
    batch = restored.sample_sequences(2, batch_size=7)
    _assert_aligned(batch, 7, 2, priv=True, labelled=True)
    assert batch.buffer_index == 1
    assert batch.episode_ends[:, -1].all()
    with pytest.raises(RuntimeError):
        restored.sample_sequences(3)


def test_padded_windows_open_every_episode_that_complete_windows_refuse():
    replay = _buffer()
    for step in range(3):
        _add(replay, step, episode=4, end=step == 2)
    with pytest.raises(RuntimeError):
        replay.sample_sequences(5)
    batch = replay.sample_sequences(5, batch_size=64, pad=True)
    assert batch.obs.shape == (64, 6, 3) and batch.valid.shape == (64, 5)
    assert batch.valid.dtype == torch.bool
    ends = batch.episode_steps[:, -1]
    assert set(ends.tolist()) == {0, 1, 2}, "every recorded transition must be a learning step"
    lengths = batch.valid.sum(dim=1)
    torch.testing.assert_close(lengths, ends + 1)
    # A prefix is contiguous and right-aligned: no hole may appear inside it.
    torch.testing.assert_close(batch.valid, torch.arange(5)[None] >= (5 - lengths)[:, None])


def test_padded_positions_carry_zeros_and_no_identity():
    replay = _buffer()
    _add(replay, 0, episode=4)
    _add(replay, 1, episode=4, end=True)
    batch = replay.sample_sequences(4, batch_size=32, pad=True)
    blank = ~batch.valid
    assert blank.any()
    assert not batch.obs[:, :4][blank].any() and not batch.actions[blank].any()
    assert not batch.rewards[blank].any() and not batch.not_dones[blank].any()
    assert not batch.episode_ends[blank].any()
    for name in ("stream_ids", "episode_ids", "episode_steps"):
        assert (getattr(batch, name)[blank] == -1).all(), f"{name} must not fake an identity"
        assert (getattr(batch, name)[batch.valid] >= 0).all()
    # The observation following the window is real even when the prefix is padded.
    torch.testing.assert_close(batch.obs[:, -1, 2], batch.episode_steps[:, -1].float() + 1)


def test_padding_never_reaches_into_the_previous_episode():
    replay = _buffer()
    for episode in (4, 5):
        for step in range(2):
            _add(replay, step, episode=episode, end=step == 1)
    batch = replay.sample_sequences(3, batch_size=64, pad=True)
    for row in range(64):
        valid = batch.valid[row]
        assert batch.episode_ids[row][valid].unique().numel() == 1
        assert not batch.episode_ends[row][valid][:-1].any()
    assert set(batch.episode_ids[:, -1].tolist()) == {4, 5}


def test_a_complete_window_is_identical_with_and_without_padding():
    """Padding only adds episode openings; it must not disturb the windows that already existed."""
    replay = _buffer()
    for step in range(6):
        _add(replay, step, episode=4)
    padded = replay.sample_sequences(3, batch_size=96, pad=True)
    complete = replay.sample_sequences(3, batch_size=96)
    assert complete.valid.all()
    assert set(complete.episode_steps[:, -1].tolist()) == {2, 3, 4, 5}
    assert padded.valid.all(dim=1).eq(padded.episode_steps[:, -1] >= 2).all()
    for end in (2, 3, 4, 5):
        left = padded.obs[padded.episode_steps[:, -1] == end][:1]
        right = complete.obs[complete.episode_steps[:, -1] == end][:1]
        torch.testing.assert_close(left, right)


def test_padding_admits_buffers_and_replays_that_hold_no_complete_window():
    replay = _buffer()
    with pytest.raises(RuntimeError):
        replay.sample_sequences(2, pad=True)
    assert not replay.sequence_ready(2, pad=True)
    every = ReplaySet(["a", "b"], 3, 2, 16, 3, "cpu", sequence=True)
    for episode in range(6):
        _add(every, 0, episode=episode, end=True, key="a")
    with pytest.raises(RuntimeError):
        every.sample_sequences(2)
    # The flat rule: a buffer must hold more than a batch before it is drawn from.
    with pytest.raises(RuntimeError):
        every.sample_sequences(2, batch_size=6, pad=True)
    batch = every.sample_sequences(2, batch_size=5, pad=True)
    assert batch.buffer_index == 0
    assert batch.valid[:, 0].sum() == 0 and batch.valid[:, 1].all()


def test_flat_replay_refuses_padded_sequences():
    flat = FlatReplayBuffer(3, 2, 8, 2, "cpu")
    with pytest.raises(ValueError):
        flat.sample_sequences(2, pad=True)
    with pytest.raises(ValueError):
        flat.sequence_ready(2, pad=True)
