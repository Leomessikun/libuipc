"""GPU smoke test of Wang's pretraining loop: a tiny live teacher over two world rotations and one evaluation world."""

import json

import pytest

pytestmark = pytest.mark.cuda
pytest.importorskip("torch")


def test_tiny_teacher_rotates_its_world_and_evaluates_held_out_poses(tmp_path):
    from uipc_manip import pretrain_wang

    pretrain_wang.main([
        "teacher", "--region", "13", "--garments", "tshirt_26", "tshirt_68", "--train-poses", "0", "1", "2", "3",
        "--eval-poses", "45", "--num-envs", "2", "--horizon", "3", "--transitions", "12", "--eval-every", "6",
        "--checkpoint-every", "6", "--replay-capacity", "64", "--batch-size", "4", "--work-dir", str(tmp_path),
        "--run-name", "smoke", "--log-interval", "1", "--hidden-dim", "32", "--point-budget", "256",
    ])
    run = tmp_path / "smoke"
    state = json.loads((run / "checkpoints" / "state.json").read_text())
    # Two episodes of six transitions: the first world, one rotation, and no rebuild after the budget is spent.
    assert state["transitions"] == 12 and state["counters"]["rotations"] == 2 and state["counters"]["sim_errors"] == 0
    rows = (run / "eval_log.csv").read_text().splitlines()
    assert len(rows) == 1 + 3  # before training, at six transitions, and at the end
    best = json.loads((run / "checkpoints" / "best.json").read_text())["metadata"]
    assert best["eval_cells"] == [["tshirt_26", 14045], ["tshirt_68", 14045]] and best["heldout_slots"] == []
