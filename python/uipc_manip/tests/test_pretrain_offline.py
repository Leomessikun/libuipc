"""CPU checks of the offline representation pretraining and its transfer into a fresh agent."""

import json

import numpy as np
import pytest
import torch

from uipc_manip import pretrain_offline
from uipc_manip.obs import EXTRA_DIM, FLAG_TOOL, POINT_DIM, ObsSpec
from uipc_manip.replay import FlatReplayBuffer
from uipc_manip.sac import SACAgent, SACConfig

POINTS, ACTION_DIM, PRIV_DIM = 8, 2, 3
NETWORK = ["--hidden-dim", "16", "--point-budget", str(POINTS), "--batch-size", "4", "--device", "cpu", "--seed", "0"]
RLT = ["--history-length", "3", "--history-kind", "rlt", "--rlt-dim", "16", "--rlt-layers", "1", "--rlt-heads", "2"]


def _observation(rng):
    block = np.zeros((POINTS, POINT_DIM), dtype=np.float32)
    block[:, :3] = rng.standard_normal((POINTS, 3)) * 0.1
    block[:, 3] = 1.0
    block[0, 3], block[0, 3 + FLAG_TOOL] = 0.0, 1.0
    return np.concatenate([block.reshape(-1), np.zeros(EXTRA_DIM, dtype=np.float32)])


def _priv(obs):
    return obs.reshape(-1)[: POINTS * POINT_DIM].reshape(POINTS, POINT_DIM)[:, :3].mean(0).astype(np.float32)


def _snapshot(directory, episodes=4, steps=6, streams=2, priv=True, sequence=True, seed=0):
    """A small recorded corpus: ``streams`` slots, ``episodes`` episodes each, learnable targets."""
    spec = ObsSpec(POINTS)
    rng = np.random.default_rng(seed)
    replay = FlatReplayBuffer(spec.dim, ACTION_DIM, 512, 4, "cpu", priv_dim=PRIV_DIM if priv else 0, sequence=sequence)
    for episode in range(episodes):
        for stream in range(streams):
            obs = _observation(rng)
            for step in range(steps):
                action = rng.uniform(-1, 1, ACTION_DIM).astype(np.float32)
                next_obs = _observation(rng)
                reward = float(action.sum())
                kw = dict(stream_id=stream, episode_id=episode * streams + stream, episode_step=step, episode_end=step == steps - 1) if sequence else {}
                replay.add(obs, action, reward, next_obs, False, _priv(obs) if priv else None, _priv(next_obs) if priv else None, **kw)
                obs = next_obs
    replay.save(directory, metadata={"transitions": episodes * steps * streams})
    return directory


@pytest.mark.parametrize("history", [[], RLT])
def test_pretraining_trains_a_representation_and_a_fresh_agent_adopts_it(tmp_path, history):
    snapshot = _snapshot(tmp_path / "replay")
    out = tmp_path / "pretrain"
    final = pretrain_offline.main(["--replay", str(snapshot), "--out", str(out), "--steps", "6", "--eval-every", "3", "--eval-batches", "2",
                                   *NETWORK, *history])
    assert final == out / "checkpoints" / "pretrain_final.pt" and (out / "checkpoints" / "pretrain_best.pt").exists()
    payload = SACAgent.read_checkpoint(final)
    meta = payload["metadata"]["pretraining"]
    # Episodes are split, never rows, and validation never touches the training statistics.
    assert meta["split"]["train_episodes"] + meta["split"]["val_episodes"] == 8 and meta["split"]["val_episodes"] == 2
    assert meta["split"]["train_rows"] == 36 and meta["split"]["val_rows"] == 12
    assert len(meta["normalization"]["priv_mean"]) == PRIV_DIM
    assert meta["sources"][0]["sha256"] and meta["sources"][0]["rows"] == 48
    assert {"val_loss", "val_next_priv", "val_reward", "val_priv_baseline", "val_reward_baseline"} <= set(meta["val"])
    rows = list(__import__("csv").DictReader((out / "pretrain_log.csv").open()))
    assert len(rows) == 6 and rows[2]["val_loss"] != "" and rows[1]["val_loss"] == ""

    cfg = SACConfig.from_dict(payload["sac_config"])
    assert (cfg.history_kind == "rlt") == bool(history)
    # The receiving run may learn in either mode; the representation does not depend on it.
    cfg.rlt_learning_mode = "prefix" if history else "endpoint"
    torch.manual_seed(5)
    agent = SACAgent(ObsSpec(POINTS), ACTION_DIM, cfg, "cpu")
    trunk_before = {k: v.clone() for k, v in agent.actor.trunk.state_dict().items()}
    info = agent.initialize_representation(final)
    assert info["components"] == (["actor.encoder", "actor.history"] if history else ["actor.encoder"])
    assert info["pretraining"]["steps"] == 6 and info["sha256"]
    for name in ("encoder", "history"):
        module = getattr(agent.actor, name)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, payload["actor"][f"{name}.{key}"])
    for key, value in agent.actor.trunk.state_dict().items():
        torch.testing.assert_close(value, trunk_before[key])
    assert json.loads(final.with_suffix(".json").read_text())["metadata"]["pretraining"]["steps"] == 6


def test_the_transfer_refuses_what_it_cannot_vouch_for(tmp_path):
    snapshot = _snapshot(tmp_path / "replay")
    final = pretrain_offline.main(["--replay", str(snapshot), "--out", str(tmp_path / "p"), "--steps", "2", "--eval-every", "2", "--eval-batches", "1",
                                   *NETWORK, *RLT])
    cfg = SACConfig.from_dict(SACAgent.read_checkpoint(final)["sac_config"])
    spec = ObsSpec(POINTS)
    other = SACConfig.from_dict(cfg.to_dict())
    other.rlt.dim = 32
    with pytest.raises(ValueError, match="protocol"):
        SACAgent(spec, ACTION_DIM, other, "cpu").initialize_representation(final)
    plain = SACAgent(spec, ACTION_DIM, cfg, "cpu")
    online = plain.save(tmp_path / "online.pt", 0)
    with pytest.raises(ValueError, match="offline pretraining checkpoint"):
        SACAgent(spec, ACTION_DIM, cfg, "cpu").initialize_representation(online)
    used = SACAgent(spec, ACTION_DIM, cfg, "cpu")
    replay = FlatReplayBuffer(spec.dim, ACTION_DIM, 64, 4, "cpu", sequence=True)
    rng = np.random.default_rng(1)
    for step in range(12):
        obs = _observation(rng)
        replay.add(obs, rng.uniform(-1, 1, ACTION_DIM), 0.0, obs, False, stream_id=0, episode_id=0, episode_step=step, episode_end=step == 11)
    used.update(replay)
    with pytest.raises(ValueError, match="fresh"):
        used.initialize_representation(final)


def test_pretraining_refuses_flat_snapshots_frame_histories_and_missing_targets(tmp_path):
    flat = _snapshot(tmp_path / "flat", sequence=False)
    with pytest.raises(ValueError, match="flat replay"):
        pretrain_offline.main(["--replay", str(flat), "--out", str(tmp_path / "a"), "--steps", "1", *NETWORK])
    snapshot = _snapshot(tmp_path / "replay")
    with pytest.raises(ValueError, match="frame concatenation"):
        pretrain_offline.main(["--replay", str(snapshot), "--out", str(tmp_path / "b"), "--steps", "1", *NETWORK, "--history-length", "3"])
    without = _snapshot(tmp_path / "nopriv", priv=False)
    with pytest.raises(ValueError, match="privileged"):
        pretrain_offline.main(["--replay", str(without), "--out", str(tmp_path / "c"), "--steps", "1", *NETWORK])
    final = pretrain_offline.main(["--replay", str(without), "--out", str(tmp_path / "d"), "--steps", "2", "--eval-every", "2",
                                   "--eval-batches", "1", "--priv-weight", "0", *NETWORK])
    assert "priv_mean" not in SACAgent.read_checkpoint(final)["metadata"]["pretraining"]["normalization"]


def test_two_snapshots_keep_their_episodes_apart():
    a = pretrain_offline.read_snapshot(_snapshot(__import__("tempfile").mkdtemp(), episodes=2, seed=1))
    b = pretrain_offline.read_snapshot(_snapshot(__import__("tempfile").mkdtemp(), episodes=2, seed=2))
    sources = a + b
    train, val = pretrain_offline.split_episodes(sources, 0.25, 0)
    assert len(train | val) == 8 and not (train & val) and len(val) == 2
    train_buffer, val_buffer = pretrain_offline.fill_buffers(sources, train, val, 4, "cpu")
    # The same stream id in the two runs became two streams, so their episodes never link.
    assert train_buffer.size + val_buffer.size == 48
    assert set(train_buffer._stream_ids[: train_buffer.size]) <= {0, 1, 2, 3}
    batch = train_buffer.sample_sequences(3, 8, pad=True, strict_context=True)
    assert batch.valid[:, -1].all()
