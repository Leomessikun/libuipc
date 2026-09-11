"""Step 0's episode tape: the records must match what ``train_sac.evaluate`` writes (CPU)."""

import json

import numpy as np

from uipc_manip.expert_baseline import EpisodeTape, build_parser

METRICS = ("upperarm_ratio", "forearm_ratio", "threaded", "task_reward")


def _tape(**kwargs) -> EpisodeTape:
    tape = EpisodeTape(METRICS, kwargs.pop("save_observations", False))
    for ratio, reward in kwargs.pop("steps", [(0.2, 1.0), (0.8, 2.0)]):
        tape.step(
            np.zeros(35), np.zeros(6), np.zeros(4), "last", reward,
            {"upperarm_ratio": ratio, "forearm_ratio": 1.0, "threaded": 1.0, "task_reward": reward,
             "tracking_error": 0.01, **kwargs},
        )
    return tape


def test_the_record_carries_the_evaluation_fields():
    record = _tape().record(2, {"success": True, "distance": 0.2, "upperarm_ratio": 0.8, "forearm_ratio": 1.0,
                                "threaded": 1.0, "task_reward": 2.0}, ("tshirt_26", 14045))
    for key in ("slot", "success", "distance", "return", "max_tracking_error", "sim_error", "early_turn", "paper_filter"):
        assert key in record
    for metric in METRICS:
        assert f"final_{metric}" in record and f"max_{metric}" in record
    assert record["slot"] == 2 and record["return"] == 3.0 and record["length"] == 2
    assert record["garment"] == "tshirt_26" and record["human"] == 14045


def test_the_paper_filter_needs_a_success_without_an_early_turn():
    kept = _tape().record(0, {"success": True, "upperarm_ratio": 0.8}, ("tshirt_4", 14046))
    assert kept["paper_filter"] is True
    turned = _tape(early_turn=True).record(0, {"success": True, "upperarm_ratio": 0.8}, ("tshirt_4", 14046))
    assert turned["early_turn"] is True and turned["paper_filter"] is False


def test_a_simulator_error_is_scored_at_the_last_completed_decision():
    # The final info of a trip carries no metrics, so the record falls back to what was last seen.
    record = _tape(steps=[(0.4, 1.0), (0.6, 1.0)]).record(0, {"sim_error": True, "success": False}, ("tshirt_68", 14047))
    assert record["sim_error"] is True
    assert record["final_upperarm_ratio"] == 0.6 and record["max_upperarm_ratio"] == 0.6


def test_the_tape_saves_what_a_demonstration_replay_needs(tmp_path):
    tape = _tape()
    record = tape.record(0, {"success": False, "upperarm_ratio": 0.8}, ("hospital_gown", 14048))
    tape.save(tmp_path / "episode_00000.npz", record)
    saved = np.load(tmp_path / "episode_00000.npz", allow_pickle=False)
    assert saved["privileged"].shape == (2, 35) and saved["actions"].shape == (2, 6)
    assert saved["rewards"].tolist() == [1.0, 2.0] and "obs" not in saved
    assert json.loads(str(saved["record"]))["length"] == 2


def test_observations_are_stored_only_when_asked(tmp_path):
    tape = _tape(save_observations=True)
    tape.save(tmp_path / "episode_00001.npz", tape.record(0, {"success": False}, ("tshirt_392", 14049)))
    assert np.load(tmp_path / "episode_00001.npz")["obs"].shape == (2, 4)


def test_the_held_out_poses_are_the_default():
    args, extra = build_parser().parse_known_args(["--region", "13", "--dt", "0.0166667"])
    assert (args.region, args.poses, args.num_envs) == (13, "heldout", 24)
    assert extra == ["--dt", "0.0166667"]
