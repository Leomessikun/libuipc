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


# ---------------------------------------------------------------- the recurrent looped history
from uipc_manip.rlt import RLTConfig, RecurrentLoopedHistory  # noqa: E402


def _rlt():
    return RLTConfig(dim=16, layers=1, heads=2, window=3)


@pytest.mark.parametrize("cls", [Actor, WangFlowActor])
def test_the_rlt_actor_acts_from_the_last_state_of_the_same_run_it_learns_from(cls):
    actor = _actor(cls, LENGTH, history_kind="rlt", rlt=_rlt())
    assert isinstance(actor.history, RecurrentLoopedHistory) and actor.trunk[0].in_features == 16
    window = _window(valid_prefix=2)
    frames, valid, commands = window
    with torch.no_grad():
        mu, log_std = actor.head(window)
        mu_all, log_std_all = actor.head_sequence(frames, valid, commands)
    assert mu_all.shape == (BATCH, LENGTH, ACTION_DIM)
    torch.testing.assert_close(mu, mu_all[:, -1])
    torch.testing.assert_close(log_std, log_std_all[:, -1])
    # Padded frames and the commands next to them are unreadable here too.
    other = _replace_frames(window, ~valid.reshape(-1), seed=99)
    torch.manual_seed(5)
    other = (other[0], valid, torch.where(valid[:, :-1, None], commands, torch.randn_like(commands)))
    with torch.no_grad():
        torch.testing.assert_close(actor.head(window)[0], actor.head(other)[0])
    assert torch.isfinite(mu_all).all()


def test_the_rlt_critic_scores_the_recorded_command_and_a_candidate_from_the_same_prefix():
    torch.manual_seed(7)
    critic = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=LENGTH, history_kind="rlt", rlt=_rlt()).eval()
    window = _window(valid_prefix=3)
    frames, valid, commands = window
    candidate = torch.randn(BATCH, ACTION_DIM)
    recorded = torch.cat([commands, candidate[:, None]], dim=1)
    with torch.no_grad():
        q_window = critic(window, candidate)[0]
        run = critic.run_sequence(frames, valid, recorded)
        q_seq = critic.q_sequence(run)[0]
        q_branch = critic.q_branch(run, frames, valid, recorded)[0]
        # The acting path, the recorded run and a branch with the recorded command agree.
        torch.testing.assert_close(q_window, q_seq[:, -1])
        torch.testing.assert_close(q_branch[valid], q_seq[valid])
        # A different candidate at t changes Q at t and nowhere before it.
        other = recorded.clone()
        other[:, 1] += 1.0
        q_other = critic.q_branch(run, frames, valid, other)[0]
        assert not torch.allclose(q_other[:, 1], q_branch[:, 1])
        torch.testing.assert_close(q_other[:, 0], q_branch[:, 0])
        torch.testing.assert_close(q_other[:, 2:], q_branch[:, 2:])
    # dQ/da at every recorded position flows through that frame's encoding.
    candidates = recorded.clone().requires_grad_(True)
    critic.q_branch(run, frames, valid, candidates)[0][valid].sum().backward()
    assert (candidates.grad[valid].abs().sum(-1) > 0).all()


def test_the_frame_history_cannot_learn_at_every_position():
    critic = Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=LENGTH)
    frames, valid, commands = _window()
    with pytest.raises(ValueError, match="history_kind='rlt'"):
        critic.run_sequence(frames, valid, torch.cat([commands, commands[:, :1]], dim=1))
    with pytest.raises(ValueError, match="Unknown history kind"):
        Critic(ObsSpec(POINTS), ACTION_DIM, 16, _cfg(), history_length=LENGTH, history_kind="gru")
