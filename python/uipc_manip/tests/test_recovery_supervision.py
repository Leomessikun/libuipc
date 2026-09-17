import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uipc_manip.obs import ObsSpec
from uipc_manip.recovery_supervision import load_recovery_supervision
from uipc_manip.sac import SACConfig


def source(tmp_path):
    env = {"clip_rotation_to_yz": True, "friction": .3}
    (tmp_path / "manifest.json").write_text(json.dumps(dict(
        env=env, obs_dim=28, action_dim=6, point_budget=3, completed=True,
        admission_rule="verified_full_continuation_v1")))
    (tmp_path / "result.json").write_text(json.dumps(dict(admitted=True)))
    np.savez(tmp_path / "good.npz", obs=np.zeros((5, 28), np.float32), actions=np.zeros((5, 6), np.float32))
    (tmp_path / "episode_metrics.json").write_text(json.dumps([
        dict(path="good.npz", kept=True, human=10, garment="shirt"),
        dict(path="absent_rejected.npz", kept=False, human=10, garment="shirt")]))
    agent = SimpleNamespace(cfg=SACConfig(), spec=ObsSpec(3), action_dim=6, device=torch.device("cpu"),
                            actor=torch.nn.Linear(1, 1), teachers={})
    return dict(source_dirs=[str(tmp_path)], agent=agent, env=env, eval_bodies={20}, weight=1, seed=1)


def test_only_verified_kept_training_rows_are_loaded(tmp_path):
    args = source(tmp_path)
    args["env"]["range"] = (0, 1)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["env"]["range"] = [0, 1]
    manifest_path.write_text(json.dumps(manifest))
    supervision, summary = load_recovery_supervision(**args)
    assert len(supervision.obs) == 5 and supervision.active_axes == (0, 1, 2, 4, 5)
    assert summary["train_cells"] == [("shirt", 10)] and summary["train_transitions"] == 5
    assert not any(k.startswith("val_") for k in summary)


@pytest.mark.parametrize("failure", ["unverified", "heldout", "physics"])
def test_unverified_or_incompatible_recoveries_are_refused(tmp_path, failure):
    args = source(tmp_path)
    if failure == "unverified":
        (tmp_path / "result.json").write_text(json.dumps(dict(admitted=False)))
    elif failure == "heldout":
        args["eval_bodies"] = {10}
    else:
        args["env"] = {**args["env"], "friction": .4}
    with pytest.raises(ValueError):
        load_recovery_supervision(**args)
