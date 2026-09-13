"""CPU checks that streaming rollout state and replay windows agree."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip.history import RolloutHistory  # noqa: E402
from uipc_manip.replay import FlatReplayBuffer  # noqa: E402


def _observation(stream, episode, step):
    return np.array([stream, episode, step], dtype=np.float32)


def _command(stream, step):
    return np.array([step, -step - stream], dtype=np.float32)


def test_a_single_frame_history_is_the_current_observation_alone():
    history = RolloutHistory(2, 3, 2, length=1)
    obs, actions, valid = history.window(np.zeros((2, 3), dtype=np.float32) + 5.0)
    assert obs.shape == (2, 1, 3) and actions.shape == (2, 0, 2)
    assert valid.all()
    history.push(np.zeros((2, 3), dtype=np.float32) + 5.0, np.zeros((2, 2), dtype=np.float32))
    obs, _, valid = history.window(np.zeros((2, 3), dtype=np.float32) + 6.0)
    assert valid.all() and np.array_equal(obs[:, 0], np.full((2, 3), 6.0, dtype=np.float32))


def test_an_episode_opening_is_masked_and_zero_padded_until_the_prefix_fills():
    history = RolloutHistory(1, 3, 2, length=4)
    for step in range(6):
        obs, actions, valid = history.window(_observation(0, 0, step)[None])
        assert valid[0].tolist() == [position >= 3 - min(step, 3) for position in range(4)]
        assert not obs[0][~valid[0]].any(), "padded observations must stay zero"
        assert not actions[0][~valid[0, :3]].any(), "padded commands must stay zero"
        np.testing.assert_array_equal(obs[0, -1], _observation(0, 0, step))
        if step:
            np.testing.assert_array_equal(actions[0, -1], _command(0, step - 1))
        history.push(_observation(0, 0, step)[None], _command(0, step)[None])


def test_a_reset_forgets_only_the_named_streams():
    history = RolloutHistory(3, 3, 2, length=3)
    for step in range(4):
        history.push(np.stack([_observation(s, 0, step) for s in range(3)]),
                     np.stack([_command(s, step) for s in range(3)]))
    history.reset(np.array([True, False, True]))
    _, actions, valid = history.window(np.zeros((3, 3), dtype=np.float32))
    assert valid[0].tolist() == valid[2].tolist() == [False, False, True]
    assert valid[1].all()
    assert not actions[0].any() and not actions[2].any(), "a reset stream keeps no stale command"
    np.testing.assert_array_equal(actions[1, -1], _command(1, 3))
    history.reset()
    assert not history.window(np.zeros((3, 3), dtype=np.float32))[2][:, :-1].any()


def test_a_reset_rejects_a_mask_that_does_not_cover_the_streams():
    history = RolloutHistory(3, 3, 2, length=2)
    with pytest.raises(ValueError):
        history.reset(np.array([True, False]))
    with pytest.raises(IndexError):
        history.reset(np.array([3]))


@pytest.mark.parametrize("length", [1, 2, 4, 7])
def test_streaming_windows_match_the_padded_replay_windows_of_the_same_episode(length):
    """The Stage 2 parity gate: collection and replay must see the same prefix."""
    steps = 5
    replay = FlatReplayBuffer(3, 2, 64, 4, "cpu", sequence=True)
    for step in range(steps):
        replay.add(_observation(0, 0, step), _command(0, step), float(step),
                   _observation(0, 0, step + 1), False, stream_id=0, episode_id=0,
                   episode_step=step, episode_end=step == steps - 1)

    history = RolloutHistory(1, 3, 2, length=length)
    for step in range(steps):
        obs, actions, valid = history.window(_observation(0, 0, step)[None])
        window = replay.sample_sequences(length, batch_size=1, pad=True)
        # Endpoints are drawn uniformly; redraw until the window ends at this step.
        while int(window.episode_steps[0, -1]) != step:
            window = replay.sample_sequences(length, batch_size=1, pad=True)
        np.testing.assert_array_equal(valid, window.valid.numpy())
        np.testing.assert_allclose(obs, window.obs[:, :length].numpy())
        np.testing.assert_allclose(actions, window.actions[:, :length - 1].numpy())
        history.push(_observation(0, 0, step)[None], _command(0, step)[None])
