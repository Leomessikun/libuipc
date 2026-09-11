"""GPU smoke test of Step 0: the scripted expert over two live configurations of a region."""

import json

import pytest

pytestmark = pytest.mark.cuda
pytest.importorskip("torch")


def test_the_expert_baseline_records_every_configuration_it_runs(tmp_path):
    from uipc_manip import expert_baseline

    expert_baseline.main([
        "--region", "13", "--poses", "heldout", "--garments", "tshirt_26", "--num-envs", "2",
        "--max-cells", "2", "--horizon", "3", "--work-dir", str(tmp_path), "--point-budget", "256",
    ])
    out = tmp_path / "expert_r13_heldout_s0"
    records = json.loads((out / "records.json").read_text())
    assert len(records) == 2
    for record in records:
        # The record carries what an evaluation round carries, so the bar and the teacher compare directly.
        assert {"slot", "success", "paper_filter", "final_upperarm_ratio", "max_upperarm_ratio", "path"} <= set(record)
        assert record["length"] == 3 and record["garment"] == "tshirt_26"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["summary"]["cell_count"] == 2 and manifest["region"] == 13
    # The demonstrations a teacher's replay is seeded from: privileged rows, actions and rewards.
    import numpy as np

    episode = np.load(out / records[0]["path"])
    assert episode["privileged"].shape == (3, 35) and episode["actions"].shape == (3, 6)
    assert episode["rewards"].shape == (3,) and "obs" not in episode
