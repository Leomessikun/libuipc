"""CPU checks that the three execution schedules of the recurrent looped transformer agree."""

import pytest
import torch

from uipc_manip.rlt import RLTConfig, RecurrentLoopedHistory, RecurrentLoopedTransformer

IN, B, T = 5, 3, 7


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = RLTConfig(dim=16, layers=2, heads=2, window=kw.pop("window", 3), **kw)
    return RecurrentLoopedTransformer(IN, cfg).eval()


def _tokens(seed=1, rows=B, length=T):
    torch.manual_seed(seed)
    return torch.randn(rows, length, IN)


def _padded(valid_prefix):
    """Right-aligned prefixes of the given lengths, one per row."""
    return torch.stack([torch.arange(T) >= T - n for n in valid_prefix])


def _stream(model, tokens, max_len=T + 2):
    state = model.init_state(tokens.shape[0], max_len)
    return torch.stack([model.step(state, tokens[:, t]) for t in range(tokens.shape[1])], dim=1)


@pytest.mark.parametrize("window", [1, 3, T + 1])
@pytest.mark.parametrize("groups, tied", [(1, False), (2, False), (1, True)])
def test_streaming_reproduces_the_window_at_every_position(window, groups, tied):
    model = _model(window=window, groups=groups, tied=tied)
    tokens = _tokens()
    with torch.no_grad():
        window_out = model.readout(model.run(tokens, torch.ones(B, T, dtype=torch.bool)).s)
    torch.testing.assert_close(_stream(model, tokens), window_out, atol=1e-5, rtol=1e-5)


def test_a_padded_row_equals_the_shorter_sequence_and_starts_from_s_star():
    model = _model()
    tokens = _tokens()
    valid = _padded([T, 4, 1])
    with torch.no_grad():
        padded = model.run(tokens, valid)
        for row, n in enumerate([T, 4, 1]):
            alone = model.run(tokens[row : row + 1, T - n :], torch.ones(1, n, dtype=torch.bool))
            torch.testing.assert_close(padded.s[row, T - n :], alone.s[0], atol=1e-5, rtol=1e-5)
            streamed = _stream(model, tokens[row : row + 1, T - n :])
            torch.testing.assert_close(streamed[0], model.readout(alone.s[0]), atol=1e-5, rtol=1e-5)
        # The first valid position of a padded row is exactly the first step of a fresh stream.
        first_alone = model.run(tokens[1:2, T - 4 : T - 3], torch.ones(1, 1, dtype=torch.bool)).s[0, 0]
        torch.testing.assert_close(padded.s[1, T - 4], first_alone, atol=1e-5, rtol=1e-5)
        assert torch.isfinite(padded.s).all()


def test_the_readout_ignores_padding_content():
    model = _model()
    valid = _padded([T, 4, 1])
    a, b = _tokens(1), _tokens(2)
    mixed = torch.where(valid[..., None], a, b)
    with torch.no_grad():
        torch.testing.assert_close(model.run(a, valid).s[valid], model.run(mixed, valid).s[valid], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("window", [1, 3])
def test_branching_with_the_recorded_token_reproduces_the_recorded_state(window):
    model = _model(window=window)
    tokens, valid = _tokens(), _padded([T, 5, 2])
    with torch.no_grad():
        run = model.run(tokens, valid)
        again = model.branch(run, tokens)
    torch.testing.assert_close(again[valid], run.s[valid], atol=1e-5, rtol=1e-5)


def test_a_branch_token_changes_only_its_own_position_and_sees_only_the_prefix():
    model = _model()
    tokens, valid = _tokens(), torch.ones(B, T, dtype=torch.bool)
    other = _tokens(3)
    with torch.no_grad():
        run = model.run(tokens, valid)
        branched = model.branch(run, other)
        # Position t with the alternative token is the state a sequence with that token at t would
        # have reached, whatever comes after t.
        for t in (0, 2, T - 1):
            spliced = torch.cat([tokens[:, :t], other[:, t : t + 1], tokens[:, t + 1 :]], dim=1)
            torch.testing.assert_close(branched[:, t], model.run(spliced, valid).s[:, t], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(run.s, model.run(tokens, valid).s)


def test_gradients_reach_the_first_token_and_the_initial_state_through_the_recurrence():
    model = _model(window=1).train()
    tokens = _tokens().requires_grad_(True)
    model.run(tokens, torch.ones(B, T, dtype=torch.bool)).s[:, -1].sum().backward()
    assert tokens.grad[:, 0].abs().sum() > 0
    assert model.s_star.grad is not None and model.s_star.grad.abs().sum() > 0


def test_streams_of_different_ages_advance_together_and_reset_individually():
    model = _model()
    tokens = _tokens(rows=2)
    state = model.init_state(2, T)
    with torch.no_grad():
        model.step(state, tokens[:, 0])
        model.reset_state(state, [1])
        assert state.count.tolist() == [1, 0]
        out = torch.stack([model.step(state, tokens[:, t]) for t in range(1, T)], dim=1)
        long_run = model.readout(model.run(tokens[:1], torch.ones(1, T, dtype=torch.bool)).s)[0, 1:]
        short_run = model.readout(model.run(tokens[1:, 1:], torch.ones(1, T - 1, dtype=torch.bool)).s)[0]
    torch.testing.assert_close(out[0], long_run, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1], short_run, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="run past"):
        model.step(state, tokens[:, 0])


def test_config_is_validated():
    with pytest.raises(ValueError, match="groups"):
        RecurrentLoopedTransformer(IN, RLTConfig(dim=16, layers=3, heads=2, groups=2))
    with pytest.raises(ValueError, match="even"):
        RecurrentLoopedTransformer(IN, RLTConfig(dim=6, layers=1, heads=2))
    with pytest.raises(ValueError, match="window"):
        RecurrentLoopedTransformer(IN, RLTConfig(dim=16, layers=1, heads=2, window=0))
    with pytest.raises(ValueError, match="two frames"):
        RecurrentLoopedHistory(4, 2, 1, RLTConfig(dim=16, layers=1, heads=2))


def test_history_tokens_carry_the_previous_command_only_where_its_frame_is_recorded():
    torch.manual_seed(0)
    history = RecurrentLoopedHistory(4, 2, 3, RLTConfig(dim=16, layers=1, heads=2))
    latent = torch.full((2, 3, 4), float("nan"))
    latent[0, 1:], latent[1] = 1.0, 2.0
    valid = torch.tensor([[False, True, True], [True, True, True]])
    commands = torch.tensor([[[9.0, 9.0], [1.0, 1.0]], [[3.0, 3.0], [4.0, 4.0]]])
    tokens = history.tokens(latent, valid, commands)
    assert torch.isfinite(tokens).all()
    assert tokens[0, 0].tolist() == [0.0] * 6
    assert tokens[0, 1].tolist() == [1.0] * 4 + [0.0, 0.0]  # first recorded frame: no command before it
    assert tokens[0, 2].tolist() == [1.0] * 4 + [1.0, 1.0]
    assert tokens[1, 0].tolist() == [2.0] * 4 + [0.0, 0.0]
    assert tokens[1, 1].tolist() == [2.0] * 4 + [3.0, 3.0]
    with torch.no_grad():
        last = history(latent, valid, commands)
        every = history.sequence(latent, valid, commands)
    torch.testing.assert_close(last, every[:, -1])
    assert last.shape == (2, 16) and torch.isfinite(every).all()
