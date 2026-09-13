"""CPU checks that the three execution schedules of the recurrent looped transformer agree."""

import copy

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


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_streaming_cache_matches_model_dtype(dtype):
    model = _model().to(dtype=dtype)
    tokens = _tokens().to(dtype=dtype)
    state = model.init_state(B, T)
    for caches in (state.enc_k, state.enc_v, state.mem_k, state.mem_v, state.dec_k, state.dec_v):
        assert all(cache.dtype == dtype for cache in caches)
    with torch.no_grad():
        expected = model.readout(model.run(tokens, torch.ones(B, T, dtype=torch.bool)).s)
        actual = torch.stack([model.step(state, tokens[:, t]) for t in range(T)], dim=1)
    tolerance = 1e-10 if dtype == torch.float64 else 0.05
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


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


def test_branch_gradients_match_independent_prefix_interventions():
    model = _model().double()
    independent = copy.deepcopy(model)
    recorded = _tokens().double().requires_grad_()
    candidates = _tokens(3).double().requires_grad_()
    recorded_copy = recorded.detach().clone().requires_grad_()
    candidates_copy = candidates.detach().clone().requires_grad_()
    valid = torch.ones(B, T, dtype=torch.bool)
    branch = model.branch(model.run(recorded, valid), candidates)
    reference = torch.stack([
        independent.run(torch.cat([recorded_copy[:, :t], candidates_copy[:, t:t + 1]], dim=1), valid[:, :t + 1]).s[:, -1]
        for t in range(T)
    ], dim=1)
    coefficients = torch.randn_like(branch)
    (branch * coefficients).sum().backward()
    (reference * coefficients).sum().backward()
    torch.testing.assert_close(branch, reference, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(recorded.grad, recorded_copy.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(candidates.grad, candidates_copy.grad, atol=1e-10, rtol=1e-10)
    for (name, parameter), (_, other) in zip(model.named_parameters(), independent.named_parameters()):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=1e-7, rtol=1e-10, msg=name)


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


def test_the_pretraining_objective_reads_only_recorded_positions_and_fits_a_predictable_sequence():
    from uipc_manip.rlt import TrajectoryPretrainingHead

    torch.manual_seed(0)
    history = RecurrentLoopedHistory(4, 2, 3, RLTConfig(dim=16, layers=1, heads=2)).train()
    head = TrajectoryPretrainingHead(16, action_dim=2, priv_dim=3, reward=True)
    latent, valid = torch.randn(4, 6, 4), _padded_rows([[False, False, True, True, True, True], [True] * 6, [True] * 6, [False] * 5 + [True]])
    commands = torch.tanh(latent[:, :, :2])            # the command is a function of the frame: learnable
    next_priv, rewards = latent[:, :, :3] * 2.0, latent[:, :, :1].sum(-1, keepdim=True)
    with pytest.raises(ValueError, match="privileged"):
        head(history.sequence(latent, valid, commands[:, :-1]), valid, commands, None, rewards)
    # Padded content is unread: changing it changes nothing.
    other = torch.where(valid[..., None], latent, torch.randn_like(latent))
    with torch.no_grad():
        a = head(history.sequence(latent, valid, commands[:, :-1]), valid, commands, next_priv, rewards)
        b = head(history.sequence(other, valid, commands[:, :-1]), valid, commands, next_priv, rewards)
    torch.testing.assert_close(a["loss"], b["loss"])
    assert set(a) == {"next_command", "next_priv", "reward", "loss"}
    optim = torch.optim.Adam([*history.parameters(), *head.parameters()], lr=3e-3)
    first = None
    for _ in range(150):
        out = head(history.sequence(latent, valid, commands[:, :-1]), valid, commands, next_priv, rewards)
        optim.zero_grad()
        out["loss"].backward()
        optim.step()
        first = out["loss"].item() if first is None else first
    assert out["loss"].item() < 0.25 * first


def _padded_rows(rows):
    return torch.tensor(rows)


def test_the_pretraining_step_runs_an_actor_on_a_sequence_batch():
    from uipc_manip.models import Actor, EncoderConfig
    from uipc_manip.obs import ObsSpec
    from uipc_manip.replay import FlatReplayBuffer
    from uipc_manip.rlt import TrajectoryPretrainingHead, pretraining_step

    torch.manual_seed(0)
    spec, action_dim, priv_dim, length = ObsSpec(8), 2, 3, 4
    encoder = EncoderConfig(sa_mlp=[[8, 8], [8, 8], [8, 8]], fp_mlp=[[8, 8], [8, 8], [8, 8]], linear_mlp=[8], output_dim=5,
                            sa_neighbors=[2, 2], sa_ratio=[1.0, 1.0])
    actor = Actor(spec, action_dim, 16, encoder, history_length=length, history_kind="rlt", rlt=RLTConfig(dim=16, layers=1, heads=2))
    head = TrajectoryPretrainingHead(16, action_dim, priv_dim)
    replay = FlatReplayBuffer(spec.dim, action_dim, 64, 4, "cpu", priv_dim=priv_dim, sequence=True)
    rng = torch.Generator().manual_seed(1)
    for step in range(10):
        obs = torch.rand(spec.dim, generator=rng).numpy()
        obs[7 * 8 - 4] = 1.0  # a tool flag somewhere so the cloud has a valid point
        replay.add(obs, torch.rand(action_dim, generator=rng).numpy(), 0.5, obs, False, priv=torch.rand(priv_dim, generator=rng).numpy(),
                   next_priv=torch.rand(priv_dim, generator=rng).numpy(), stream_id=0, episode_id=0, episode_step=step, episode_end=step == 9)
    batch = replay.sample_sequences(length, 4, pad=True)
    out = pretraining_step(actor, head, batch, spec.unpack_torch)
    assert torch.isfinite(out["loss"]) and out["loss"].requires_grad
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.encoder.parameters())


def test_pretraining_fits_action_dependent_outcomes_at_the_same_history():
    from uipc_manip.rlt import TrajectoryPretrainingHead

    torch.manual_seed(92)
    head = TrajectoryPretrainingHead(4, 1, priv_dim=2)
    # Identical history with opposite actions: a state-only predictor cannot fit these outcomes.
    state = torch.zeros(2, 1, 4)
    commands = torch.tensor([[[-1.0]], [[1.0]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    next_priv = torch.cat([2 * commands, -commands], dim=-1)
    reward = 3 * commands
    optimizer = torch.optim.SGD(head.parameters(), lr=0.15)
    for _ in range(80):
        losses = head(state, valid, commands, next_priv, reward)
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()
    losses = head(state, valid, commands, next_priv, reward)
    assert losses["next_priv"].item() < 1e-6
    assert losses["reward"].item() < 1e-6
    assert not losses["next_command"].requires_grad
    assert all(parameter.grad is None for parameter in head.command.parameters())


def test_pretraining_masks_nonfinite_padding_before_arithmetic():
    from uipc_manip.rlt import TrajectoryPretrainingHead

    torch.manual_seed(94)
    head = TrajectoryPretrainingHead(4, 2, priv_dim=3, command_weight=0.5)
    reference = copy.deepcopy(head)
    state = torch.randn(1, 3, 4)
    state[:, :2] = float("nan")
    state.requires_grad_()
    valid = torch.tensor([[False, False, True]])
    commands, next_priv, rewards = torch.randn(1, 3, 2), torch.randn(1, 3, 3), torch.randn(1, 3, 1)
    for target in (commands, next_priv, rewards):
        target[:, :2] = float("nan")
    actual = head(state, valid, commands, next_priv, rewards)
    expected = reference(state.detach()[:, -1:], valid[:, -1:], commands[:, -1:], next_priv[:, -1:], rewards[:, -1:])
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    expected["loss"].backward()
    assert torch.isfinite(state.grad).all() and not state.grad[:, :2].any()
    for parameter, other in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad)


def test_pretraining_component_means_and_weights_are_explicit():
    from uipc_manip.rlt import TrajectoryPretrainingHead

    head = TrajectoryPretrainingHead(4, 2, priv_dim=3, command_weight=2, priv_weight=3, reward_weight=4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    state, valid = torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)
    losses = head(state, valid, torch.ones(1, 2, 2), torch.full((1, 2, 3), 2.0), torch.full((1, 2, 1), 3.0))
    assert losses["next_command"].item() == 1
    assert losses["next_priv"].item() == 4
    assert losses["reward"].item() == 9
    assert losses["loss"].item() == 2 * 1 + 3 * 4 + 4 * 9


def test_actor_state_never_consumes_its_current_or_future_command():
    history = RecurrentLoopedHistory(4, 2, 4, RLTConfig(dim=16, layers=1, heads=2))
    latent = torch.randn(1, 4, 4)
    commands = torch.randn(1, 4, 2, requires_grad=True)
    state = history.sequence(latent, torch.ones(1, 4, dtype=torch.bool), commands[:, :-1])
    state[:, 2].sum().backward()
    assert commands.grad[:, :2].abs().sum() > 0
    assert not commands.grad[:, 2:].any()
