import json

import numpy as np
import pytest

from uipc_manip.distill import episode_ok, load_dataset


def dataset(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps(dict(obs_dim=21, action_dim=6, point_budget=2)))
    records = []
    for index, body in enumerate([10, 10, 20, 20]):
        path = f"episode_{index}.npz"
        np.savez(tmp_path / path, obs=np.full((5, 21), body, dtype=np.float32),
                 actions=np.zeros((5, 6), dtype=np.float32))
        records.append(dict(path=path, human=body, garment="shirt", kept=True))
    (tmp_path / "episode_metrics.json").write_text(json.dumps(records))
    return dict(source_dirs=[str(tmp_path)], val_ratio=.5, seed=3,
                max_train_transitions=0, min_upperarm_ratio=None)


def test_body_split_keeps_all_repeats_out_of_training_and_does_not_cap_validation(tmp_path):
    args = dataset(tmp_path)
    args["max_train_transitions"] = 3
    train, _, val, _, _, summary = load_dataset(**args, validation_bodies=[20])
    assert train.shape == (3, 21) and val.shape == (10, 21)
    assert np.all(train == 10) and np.all(val == 20)
    assert summary["train_cells"] == [("shirt", 10)]
    assert summary["val_cells"] == [("shirt", 20)]
    assert not set(summary["train_paths"]) & set(summary["val_paths"])


def test_body_split_rejects_missing_ids_or_empty_partition(tmp_path):
    args = dataset(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        load_dataset(**args, validation_bodies=[99])
    records = json.loads((tmp_path / "episode_metrics.json").read_text())
    records[0].pop("human")
    (tmp_path / "episode_metrics.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="human ID"):
        load_dataset(**args, validation_bodies=[20])


def test_dataset_rejects_misaligned_observations_actions(tmp_path):
    args = dataset(tmp_path)
    np.savez(tmp_path / "episode_0.npz", obs=np.zeros((5, 21)), actions=np.zeros((4, 6)))
    with pytest.raises(ValueError, match="alignment"):
        load_dataset(**args, validation_bodies=[20])


def test_unkept_reconstructed_failure_does_not_enter_training(tmp_path):
    args = dataset(tmp_path)
    records = json.loads((tmp_path / "episode_metrics.json").read_text())
    records[0]["kept"] = False
    (tmp_path / "episode_metrics.json").write_text(json.dumps(records))
    train, _, _, _, _, summary = load_dataset(**args, validation_bodies=[20])
    assert len(train) == 5 and summary["train_episodes"] == 1


def test_equal_dimensions_do_not_allow_mixing_different_physics(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    args = dataset(first)
    dataset(second)
    for directory, youngs in [(first, 6000), (second, 60000)]:
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["env"] = {"cloth_youngs": youngs}
        path.write_text(json.dumps(manifest))
    args["source_dirs"].append(str(second))
    with pytest.raises(ValueError, match="environment contracts"):
        load_dataset(**args, validation_bodies=[20])


def test_extra_coverage_threshold_only_tightens_the_explicit_admission_rule():
    row = dict(path="episode.npz", kept=False, final_upperarm_ratio=.9, early_turn=False)
    assert not episode_ok(row, min_upperarm_ratio=.8)
    row.update(kept=True, early_turn=True)
    assert episode_ok(row, min_upperarm_ratio=.8)
    assert not episode_ok(row, min_upperarm_ratio=.95)
