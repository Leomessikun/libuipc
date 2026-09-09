"""CPU checks for diagnostic trace identity and replay safeguards."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from uipc_manip.diagnose_dressing import ACTION_CONFIG_KEYS, cell_fingerprint, compare_traces, validate_replay


def _cell():
    names = ("cloth", "faces", "opening_idx", "grasp_idx", "picker_idx", "alignment_idx", "picker_pos",
             "arm_points", "arm_faces", "human_points", "finger", "elbow", "shoulder")
    return SimpleNamespace(**{name: np.arange(3, dtype=np.float64) for name in names})


def test_replay_rejects_different_geometry_and_action_timing():
    cell = _cell()
    cfg = dict(dt=1 / 60, action_repeat=6, max_ee_speed_m_s=0.15, max_rotation=0.087, clip_rotation_to_yz=True)
    metadata = {"cell_sha256": cell_fingerprint(cell), "config": cfg}
    validate_replay(metadata, cell, cfg)
    with pytest.raises(ValueError, match="action_repeat"):
        validate_replay(metadata, cell, {**cfg, "action_repeat": 1})
    cell.opening_idx = cell.opening_idx[::-1].copy()
    with pytest.raises(ValueError, match="geometry or semantic"):
        validate_replay(metadata, cell, cfg)


def test_comparison_distinguishes_equal_actions_from_equal_initial_states(tmp_path):
    paths = [tmp_path / "left.npz", tmp_path / "right.npz"]
    metadata = {"cell_sha256": "same", "config": {key: 1 for key in ACTION_CONFIG_KEYS}}
    for i, path in enumerate(paths):
        np.savez(path, actions=np.zeros((2, 6)), positions=np.ones((3, 4, 3)) * i, tcp=np.zeros((3, 3)))
        path.with_suffix(".json").write_text(json.dumps(metadata))
    result = compare_traces(*paths)
    assert result["same_actions"] and result["same_cell"] and result["same_action_scaling"]
    assert result["initial_cloth_max_difference_m"] == pytest.approx(np.sqrt(3))
    assert result["tcp_max_difference_m"] == 0.0
