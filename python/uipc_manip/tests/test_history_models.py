"""CPU checks for the ordered finite history in the actor and dense critic."""

import pytest
import torch

from uipc_manip.models import Actor, Critic, EncoderConfig, FrameHistory, WangFlowActor
from uipc_manip.obs import EXTRA_DIM, FEATURE_DIM, ObsSpec

ACTION_DIM, POINTS, BATCH, LENGTH = 6, 24, 3, 4


def _cfg():
    return EncoderConfig(sa_mlp=[[8, 8], [8, 8], [8, 8]], fp_mlp=[[8, 8], [8, 8], [8, 8]], linear_mlp=[8],
                         output_dim=5, sa_neighbors=[2, 2], sa_ratio=[1.0, 1.0])


def _frames(rows, seed=0):
    torch.manual_seed(seed)
    feat = torch.randn(rows, POINTS, FEATURE_DIM)
    feat[:, 0, -1] = 5.0  # one unmistakable tool point per cloud, for the Wang head
    return (torch.randn(rows, POINTS, 3), feat, torch.ones(rows, POINTS, dtype=torch.bool), torch.randn(rows, EXTRA_DIM))


def _window(valid_prefix=LENGTH, seed=0):
    """A window with `valid_prefix` real frames, right-aligned, plus the commands between frames."""
    frames = _frames(BATCH * LENGTH, seed)
    valid = torch.arange(LENGTH)[None].expand(BATCH, -1) >= LENGTH - valid_prefix
    torch.manual_seed(seed + 1)
    commands = torch.randn(BATCH, LENGTH - 1, ACTION_DIM)
    return frames, valid, commands


def _replace_frames(window, rows, seed):
    """Overwrite the given flattened rows of a window with unrelated content."""
    frames, valid, commands = window
    garbage = _frames(BATCH * LENGTH, seed)
    frames = tuple(torch.where(rows.view(-1, *([1] * (t.dim() - 1))), g, t) for t, g in zip(frames, garbage))
    return frames, valid, commands


def _actor(cls, length, **kw):
    torch.manual_seed(7)
    return cls(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), True, -10.0, 2.0, "plain", 2, history_length=length, **kw).eval()


@pytest.mark.parametrize("cls", [Actor, WangFlowActor])
def test_length_one_is_the_single_frame_network_exactly(cls):
    torch.manual_seed(7)
    plain = cls(ObsSpec(POINTS), ACTION_DIM, 16, _cfg()).eval()
    single = _actor(cls, 1)
    assert single.history is None
    assert list(plain.state_dict()) == list(single.state_dict())
    obs = _frames(BATCH)
    with torch.no_grad():
        torch.testing.assert_close(plain.head(obs)[0], single.head(obs)[0])


def test_length_one_critic_is_the_single_frame_critic_exactly():
    torch.manual_seed(7)
    plain = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg()).eval()
    torch.manual_seed(7)
    single = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=1).eval()
    assert single.history is None and list(plain.state_dict()) == list(single.state_dict())
    obs, action = _frames(BATCH), torch.randn(BATCH, ACTION_DIM)
    with torch.no_grad():
        torch.testing.assert_close(plain(obs, action)[0], single(obs, action)[0])


def test_the_actor_layout_carries_commands_and_the_dense_critic_layout_does_not():
    frame_dim = 5 + EXTRA_DIM
    actor, critic = _actor(Actor, LENGTH), Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=LENGTH)
    assert actor.history.out_dim == LENGTH * (frame_dim + 1) + (LENGTH - 1) * ACTION_DIM
    assert critic.history.out_dim == LENGTH * (frame_dim + 1)
    assert actor.trunk[0].in_features == actor.history.out_dim
    assert critic.Q1.trunk[0].in_features == critic.history.out_dim


@pytest.mark.parametrize("cls", [Actor, WangFlowActor])
def test_padded_frames_and_their_commands_cannot_reach_the_policy(cls):
    actor = _actor(cls, LENGTH)
    window = _window(valid_prefix=2)
    frames, valid, commands = window
    padded_rows = ~valid.reshape(-1)
    other = _replace_frames(window, padded_rows, seed=99)
    torch.manual_seed(5)
    other = (other[0], valid, torch.where(valid[:, :-1, None], commands, torch.randn_like(commands)))
    with torch.no_grad():
        torch.testing.assert_close(actor.head(window)[0], actor.head(other)[0])
    assert actor.head(window)[0].shape == (BATCH, ACTION_DIM)


def test_padded_frames_cannot_reach_the_critic_but_recorded_ones_do():
    critic = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=LENGTH).eval()
    window = _window(valid_prefix=2)
    frames, valid, commands = window
    action = torch.randn(BATCH, ACTION_DIM)
    padded = _replace_frames(window, ~valid.reshape(-1), seed=99)
    with torch.no_grad():
        torch.testing.assert_close(critic(window, action)[0], critic(padded, action)[0])
        # The oldest recorded frame and the command that followed it are both read.
        oldest = torch.zeros(BATCH, LENGTH, dtype=torch.bool)
        oldest[:, LENGTH - 2] = True
        changed = _replace_frames(window, oldest.reshape(-1), seed=98)
        assert not torch.allclose(critic(window, action)[0], critic(changed, action)[0])
        other_commands = commands.clone()
        other_commands[:, LENGTH - 2] += 1.0
        assert not torch.allclose(critic(window, action)[0], critic((frames, valid, other_commands), action)[0])


@pytest.mark.parametrize("cls", [Actor, WangFlowActor])
def test_the_oldest_recorded_frame_and_command_reach_the_policy(cls):
    actor = _actor(cls, LENGTH)
    window = _window()
    frames, valid, commands = window
    first = torch.zeros(BATCH, LENGTH, dtype=torch.bool)
    first[:, 0] = True
    with torch.no_grad():
        mu = actor.head(window)[0]
        assert not torch.allclose(mu, actor.head(_replace_frames(window, first.reshape(-1), seed=98))[0])
        shifted = commands.clone()
        shifted[:, 0] += 1.0
        assert not torch.allclose(mu, actor.head((frames, valid, shifted))[0])


def test_the_candidate_action_enters_only_the_current_frame_and_keeps_its_gradient():
    critic = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=LENGTH).eval()
    frames, valid, commands = _window()
    a, b = torch.zeros(BATCH, ACTION_DIM), torch.ones(BATCH, ACTION_DIM)
    with torch.no_grad():
        for candidate in (a, b):
            per_frame = torch.cat([commands, candidate[:, None]], dim=1).reshape(-1, ACTION_DIM)
            latent = critic._frame_latent(frames, per_frame).reshape(BATCH, LENGTH, -1)
            if candidate is a:
                past = latent[:, :-1]
            else:
                torch.testing.assert_close(past, latent[:, :-1])
                assert not torch.allclose(past, latent[:, 1:])
    action = torch.randn(BATCH, ACTION_DIM, requires_grad=True)
    q1, q2 = critic((frames, valid, commands), action)
    (q1.sum() + q2.sum()).backward()
    assert action.grad is not None and action.grad.abs().sum() > 0.0
    assert commands.grad is None, "scoring a candidate must not touch the recorded past"


def test_unsupported_history_combinations_fail_before_any_forward():
    with pytest.raises(ValueError):
        Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), action_mode="latent", history_length=LENGTH)
    with pytest.raises(ValueError):
        FrameHistory(5, ACTION_DIM, 1)
    with pytest.raises(ValueError):
        FrameHistory(5, ACTION_DIM, LENGTH)(torch.zeros(BATCH, LENGTH + 1, 5), torch.ones(BATCH, LENGTH + 1, dtype=torch.bool))
