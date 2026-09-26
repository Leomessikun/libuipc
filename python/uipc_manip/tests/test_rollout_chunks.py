"""Causal windows, body leakage, invalid demonstrations and flow checkpoint use."""
import json

import numpy as np
import pytest
import torch

from uipc_manip.fql import FQLConfig
from uipc_manip.obs import ObsSpec
from uipc_manip.rollout_chunks import ChunkDataset, prepare, read_episode
from uipc_manip.tests.test_sac_agent import _small_cfg
from uipc_manip.train_flow_bc import ChunkFlowPolicy, tensor_batch


def source_episode(root, body):
    job = root / f"job_{body}"
    episode = job / "episode"
    episode.mkdir(parents=True)
    config = dict(point_budget=10, obs=dict(mode="wang_static_arm"), max_translation=.008,
                  max_rotation=.08, dt=.02, action_repeat=5)
    (episode / "config.json").write_text(json.dumps(dict(config=config)))
    metadata = dict(body=body, accepted=True, stable_success=True, hold_complete=True, valid_grasp=True,
                    sim_error=None, max_translation_m=.008, max_rotation_rad=.08, decision_dt_s=.1,
                    tcp_rotation_convention="world relative to initial", collision_geometry="full_body",
                    checkpoint_sha256="test", success_geometry="physical_sleeve", stop_proximal_upper=.7)
    # Each observation and command identifies its source episode and time.
    obs = np.stack([np.full(ObsSpec(10).dim, body + t / 10, np.float32) for t in range(5)])
    actions = np.stack([np.full(6, t / 10, np.float32) for t in range(4)])
    path = episode / "baseline.npz"
    np.savez(path, obs=obs, actions=actions, grasp_valid=np.ones(4, bool),
             controller_id=np.array([0, 0, 0, 2]), metadata_json=json.dumps(metadata))
    return dict(log=str(job.with_suffix(".log")), path="episode/baseline.npz", accepted=True,
                body=body, garment="test", replica=0, seed=1, transitions=4)


def test_preparation_splits_bodies_and_removes_duplicate_training_arrays(tmp_path):
    source = tmp_path / "source"
    rows = [source_episode(source, body) for body in (1, 2, 3)]
    (source / "manifest.json").write_text(json.dumps(dict(accepted=[*rows, rows[0]])))
    result = prepare(source, tmp_path / "data", seed=7)
    assert len(result["episodes"]) == 3 and len(result["duplicates"]) == 1 and not result["rejected"]
    train = {e["body"] for e in result["episodes"] if e["split"] == "train"}
    validation = {e["body"] for e in result["episodes"] if e["split"] == "validation"}
    assert train and validation and not train & validation
    data = ChunkDataset(tmp_path / "data", "train", history=3, horizon=3)
    body = data.episodes[0]["body"]
    first, second, last = data[0], data[1], data[3]
    np.testing.assert_array_equal(first["history_valid"], [False, False, True])
    np.testing.assert_allclose(second["obs"][:, 0], [body, body, body + .1])
    np.testing.assert_allclose(second["actions"][:, 0], [.1, .2, .3])
    np.testing.assert_array_equal(last["action_valid"], [True, False, False])
    np.testing.assert_allclose(last["actions"][:, 0], [.3, 0, 0])
    # The next episode starts with a fresh history, never the previous ending.
    np.testing.assert_array_equal(data[4]["history_valid"], [False, False, True])
    assert data[4]["obs"][0, 0] == data.episodes[1]["body"]
    with pytest.raises(IndexError):
        data[len(data)]


@pytest.mark.parametrize("bad", ["alignment", "grasp", "nonfinite", "bounds"])
def test_unusable_episode_is_rejected(tmp_path, bad):
    row = source_episode(tmp_path, 1)
    path = tmp_path / "job_1/episode/baseline.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    if bad == "alignment":
        arrays["obs"] = arrays["obs"][:-1]
    elif bad == "grasp":
        arrays["grasp_valid"][2] = False
    else:
        arrays["actions"][0, 0] = np.nan if bad == "nonfinite" else 1.2
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        read_episode(row)


def test_flow_trains_encoder_and_chunk_head_and_reloads_for_inference(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    spec = ObsSpec(10)
    model = ChunkFlowPolicy(spec, FQLConfig(encoder=_small_cfg().encoder, hidden_dim=16, flow_steps=2), 3, 4)
    obs = spec.pack(np.array([[.02, 0, 0], [.03, .01, 0]]), np.array([False, True]),
                    np.array([.1, 0, 0]), np.zeros(3), True)
    batch = tensor_batch(dict(obs=np.tile(obs, (2, 3, 1)), history_valid=np.array([[False, True, True], [True] * 3]),
                              actions=np.full((2, 4, 6), .1, np.float32),
                              action_valid=np.array([[True, True, False, False], [True] * 4])), "cpu")
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    for prefix in ("encoder.", "trunk."):
        assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name.startswith(prefix))
    noise = torch.randn(2, 24)
    expected = model.eval().sample(batch["obs"], batch["history_valid"], noise=noise)
    assert expected.shape == (2, 4, 6) and expected.abs().max() <= 1
    path = tmp_path / "flow.pt"
    model.save(path, metadata=dict(test=True), step=1)
    loaded, metadata = ChunkFlowPolicy.load(path)
    assert metadata == dict(test=True)
    torch.testing.assert_close(expected, loaded.sample(batch["obs"], batch["history_valid"], noise=noise), rtol=0, atol=0)
