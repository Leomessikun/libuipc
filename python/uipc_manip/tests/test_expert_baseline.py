"""Step 0's episode tape: the records must match what ``train_sac.evaluate`` writes (CPU)."""

import json

import numpy as np
import pytest

from uipc_manip.expert_baseline import EpisodeTape, build_parser, run_world

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


class _StubEnv:
    """Enough of the dressing environment to drive ``run_world`` on the CPU."""

    metric_keys = ("upperarm_ratio", "forearm_ratio")
    action_dim = 6
    num_envs = 2

    def __init__(self, horizon: int = 3) -> None:
        self.horizon, self.step_count, self.seen = horizon, 0, []

    def reset(self, seeds):
        self.step_count = 0
        self.seeds = list(seeds)
        return np.zeros((self.num_envs, 4), dtype=np.float32)

    def privileged(self):
        return np.full((self.num_envs, 35), float(self.step_count), dtype=np.float32)

    def scripted_actions(self):
        return np.full((self.num_envs, self.action_dim), 0.5, dtype=np.float32)

    def scripted_stage_names(self):
        return ["last"] * self.num_envs

    def step(self, actions):
        self.seen.append(np.asarray(actions, dtype=np.float32).copy())
        self.step_count += 1
        done = self.step_count >= self.horizon
        infos = [{"upperarm_ratio": 0.1 * self.step_count, "forearm_ratio": 1.0, "success": False,
                  "time_limit": done, "cell": f"cell_{i}"} for i in range(self.num_envs)]
        return (np.zeros((self.num_envs, 4), dtype=np.float32), np.ones(self.num_envs, dtype=np.float32),
                np.full(self.num_envs, done), infos)


def test_run_world_records_and_saves_every_slot(tmp_path):
    env = _StubEnv()
    records = run_world(env, [("tshirt_26", 14045), ("tshirt_4", 14046)], 3, 7, tmp_path, 0)
    assert [r["slot"] for r in records] == [0, 1]
    assert env.seeds == [7, 8]
    assert all(np.allclose(a, 0.5) for a in env.seen)  # the expert played
    for record in records:
        saved = np.load(tmp_path / "episodes" / f"episode_{record['slot']:05d}.npz")
        assert saved["privileged"].shape == (3, 35) and saved["actions"].shape == (3, 6)
        assert record["final_upperarm_ratio"] == pytest.approx(0.3) and record["length"] == 3


def test_a_policy_replaces_the_expert(tmp_path):
    env = _StubEnv()
    records = run_world(env, [("tshirt_26", 14045), ("tshirt_4", 14046)], 3, 0, tmp_path, 4,
                        policy=lambda obs: np.full((2, 6), -1.0, dtype=np.float32))
    assert all(np.allclose(a, -1.0) for a in env.seen)
    assert [r["slot"] for r in records] == [4, 5]
    assert records[0]["final_stage"] == "policy"


def test_a_checkpoint_can_be_played_instead_of_the_expert():
    args, _ = build_parser().parse_known_args(["--region", "13", "--checkpoint", "run/checkpoints/best.pt"])
    assert args.checkpoint == "run/checkpoints/best.pt"

