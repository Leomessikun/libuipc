"""CPU regression tests for experiment continuation and unbiased evaluation."""

import csv
from types import SimpleNamespace

import numpy as np
import torch
import pytest

pytest.importorskip("torch")

from uipc_manip.dressing_env import DressingConfig
from uipc_manip.obs import ObsSpec
from uipc_manip.sac import SACConfig
from uipc_manip.train_sac import (
    CsvLogger,
    build_parser,
    build_sac_config,
    evaluate,
    resolve_defaults,
    restore_env_config,
    restore_resume_args,
    validate_resume_replay,
)


def _checkpoint():
    env = DressingConfig(horizon=150, action_repeat=6, constraint_strength=100.0)
    env.reward.upper_w = 7.0
    env.obs.pose_jitter_m = 0.08
    return {
        "step": 20,
        "sac_config": SACConfig(discount=0.99, alpha_lr=3e-5, init_temperature=0.0167).to_dict(),
        "metadata": {
            "task": "dressing", "env": env.to_dict(), "reward_scale": 1.0,
            "training_args": {"garment_curriculum_interval": 50},
        },
    }


def test_evaluation_keeps_grasp_valid_success_separate_from_geometric_success():
    class Env:
        num_envs = 2
        grasp_tracking_tolerance_m = 0.02

        def reset(self, seeds):
            return np.zeros((2, 1))

        def step(self, actions):
            infos = [dict(success=True, distance=0, grasp_valid=False, valid_grasp_success=False),
                     dict(success=True, distance=0, grasp_valid=True, valid_grasp_success=True)]
            return np.zeros((2, 1)), np.zeros(2), np.ones(2, bool), infos

    result = evaluate(Env(), lambda obs, deterministic: obs, ObsSpec(3), SimpleNamespace(seed=0), 2)
    assert result["success_rate"] == 1
    assert result["valid_grasp_success_rate"] == result["grasp_valid_rate"] == 0.5


def test_resume_recovers_timing_reward_camera_and_temperature():
    args = build_parser().parse_args([])
    cfg = restore_resume_args(args, [], _checkpoint())
    resolve_defaults(args)
    assert (args.horizon, args.action_repeat) == (150, 6)
    assert args.garment_curriculum_interval == 50
    assert cfg.alpha_lr == 3e-5 and cfg.init_temperature == 0.0167
    env_cfg = restore_env_config(DressingConfig(), args)
    assert env_cfg.constraint_strength == 100.0
    assert env_cfg.reward.upper_w == 7.0
    assert env_cfg.obs.pose_jitter_m == 0.08
    assert env_cfg.cache.cache_path.exists()


def test_new_run_uses_dense_residual_but_resume_keeps_saved_architecture():
    args = build_parser().parse_args([])
    assert (args.critic_action_mode, args.trunk_style) == ("dense", "residual")
    saved = _checkpoint()
    saved["sac_config"].update(critic_action_mode="latent", trunk_style="plain")
    restore_resume_args(args, [], saved)
    assert (args.critic_action_mode, args.trunk_style) == ("latent", "plain")
    argv = ["--eval-only", "--trunk-style", "residual"]
    with pytest.raises(ValueError, match="trunk-style"):
        restore_resume_args(build_parser().parse_args(argv), argv, saved)


def test_sequence_replay_setting_restores_from_checkpoint():
    payload = _checkpoint()
    payload["metadata"]["training_args"]["sequence_replay"] = True
    args = build_parser().parse_args([])
    assert not args.sequence_replay
    restore_resume_args(args, [], payload)
    assert args.sequence_replay


def test_resume_rejects_silent_protocol_change_but_eval_allows_explicit_timing():
    argv = ["--horizon", "900"]
    with pytest.raises(ValueError, match="horizon"):
        restore_resume_args(build_parser().parse_args(argv), argv, _checkpoint())
    argv += ["--eval-only"]
    args = build_parser().parse_args(argv)
    restore_resume_args(args, argv, _checkpoint())
    assert restore_env_config(DressingConfig(horizon=args.horizon), args).horizon == 900
    argv += ["--encoder", "transformer"]
    with pytest.raises(ValueError, match="encoder"):
        restore_resume_args(build_parser().parse_args(argv), argv, _checkpoint())



def test_resume_keeps_the_critic_input():
    argv = ["--critic-input", "privileged"]
    with pytest.raises(ValueError, match="critic-input"):
        restore_resume_args(build_parser().parse_args(argv), argv, _checkpoint())
    payload = _checkpoint()
    payload["sac_config"].update(critic_input="privileged", privileged_dim=35)
    args = build_parser().parse_args(["--eval-only"])
    cfg = restore_resume_args(args, ["--eval-only"], payload)
    assert args.critic_input == "privileged" and cfg.privileged_dim == 35


def test_time_step_and_settle_reach_the_dressing_config_and_are_pinned_on_resume():
    from uipc_manip.train_sac import dressing_config

    args = build_parser().parse_args([])
    resolve_defaults(args)
    cfg, default = dressing_config(args), DressingConfig()
    assert (cfg.dt, cfg.settle_steps, cfg.constraint_strength) == (default.dt, default.settle_steps, 1.0e4)
    argv = ["--dt", repr(1.0 / 30.0), "--action-repeat", "3", "--cuff-strength", "4e4", "--settle-steps", "15"]
    args = build_parser().parse_args(argv)
    resolve_defaults(args)
    cfg = dressing_config(args)
    assert (cfg.dt, cfg.action_repeat, cfg.constraint_strength, cfg.settle_steps) == (1.0 / 30.0, 3, 4.0e4, 15)
    args = build_parser().parse_args(["--task", "cloth_drag"])
    resolve_defaults(args)
    assert args.settle_steps == 40
    payload = _checkpoint()
    payload["metadata"]["env"].update(dt=1.0 / 30.0, settle_steps=15)
    args = build_parser().parse_args([])
    restore_resume_args(args, [], payload)
    assert (args.dt, args.settle_steps) == (1.0 / 30.0, 15)
    argv = ["--dt", repr(1.0 / 60.0)]
    with pytest.raises(ValueError, match="dt"):
        restore_resume_args(build_parser().parse_args(argv), argv, payload)


def test_resume_keeps_the_obs_mode_and_playback_may_look_through_another_rig():
    payload = _checkpoint()
    payload["metadata"]["env"]["obs"]["mode"] = "stretch3_head_wrist"
    args = build_parser().parse_args([])
    restore_resume_args(args, [], payload)
    assert args.obs_mode == "stretch3_head_wrist"
    assert restore_env_config(DressingConfig(), args).obs.mode == "stretch3_head_wrist"
    argv = ["--obs-mode", "visible_dual"]
    with pytest.raises(ValueError, match="obs-mode"):
        restore_resume_args(build_parser().parse_args(argv), argv, payload)
    argv += ["--eval-only"]
    args = build_parser().parse_args(argv)
    restore_resume_args(args, argv, payload)
    assert args.obs_mode == "visible_dual"


def test_teachers_are_keyed_by_the_one_region_their_training_cells_share(tmp_path):
    from uipc_manip.models import EncoderConfig
    from uipc_manip.sac import SACAgent
    from uipc_manip.train_sac import load_teachers

    spec = ObsSpec(10)
    cfg = SACConfig(hidden_dim=16, actor_type="wang-flow", encoder=EncoderConfig(kind="pointnet2", sa_neighbors=[4, 4]))
    agent = SACAgent(spec, 3, cfg, "cpu")
    # Region 13 bodies train; the held-out slot may sit anywhere.
    agent.save(tmp_path / "r13.pt", 1, {"cells": [["tshirt_26", 14000], ["tshirt_68", 14001], ["tshirt_26", 3]], "heldout_slots": [2]})
    teachers = load_teachers([tmp_path / "r13.pt"], spec, 3, "cpu")
    assert list(teachers) == [13]
    agent.save(tmp_path / "mixed.pt", 1, {"cells": [["tshirt_26", 14000], ["tshirt_26", 11000]], "heldout_slots": []})
    with pytest.raises(ValueError, match="exactly one"):
        load_teachers([tmp_path / "mixed.pt"], spec, 3, "cpu")
    with pytest.raises(ValueError, match="Two teachers"):
        load_teachers([tmp_path / "r13.pt", tmp_path / "r13.pt"], spec, 3, "cpu")

def test_resume_checks_replay_step_and_reward_scale():
    validate_resume_replay(_checkpoint(), {"step": 20})  # legacy snapshots
    with pytest.raises(ValueError, match="step"):
        validate_resume_replay(_checkpoint(), {"step": 19})
    with pytest.raises(ValueError, match="reward scale"):
        validate_resume_replay(_checkpoint(), {"step": 20, "reward_scale": 1 / 6})


def test_csv_preserves_history_and_late_optimizer_metrics(tmp_path):
    path = tmp_path / "train.csv"
    logger = CsvLogger(path)
    logger.log({"step": 1, "transitions": 20})
    logger.log({"step": 2, "transitions": 40, "alpha": 0.1})
    CsvLogger(path).log({"step": 3, "transitions": 60, "actor_loss": 1.2})
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["step"] for row in rows] == ["1", "2", "3"]
    assert rows[1]["alpha"] == "0.1" and rows[2]["actor_loss"] == "1.2"


class UnequalEpisodeEnv:
    num_envs = 3
    metric_keys = ("forearm_ratio", "upperarm_ratio")

    def reset(self, seeds):
        self.step_count = 0
        return np.zeros((3, 1))

    def step(self, actions):
        self.step_count += 1
        # The failing first slot resets every step; it must not dominate the
        # estimate or crowd the harder/longer garment out of the evaluation.
        dones = np.array([True, self.step_count % 2 == 0, self.step_count % 3 == 0])
        infos = [
            {"success": i == 2, "distance": 1 - i / 2, "garment": f"garment_{i}",
             "forearm_ratio": i / 2, "upperarm_ratio": i / 2}
            for i in range(3)
        ]
        return np.zeros((3, 1)), np.zeros(3), dones, infos


class TrippingEnv:
    """Two slots whose third decision ends on a simulator error, as the decision watchdog does."""

    num_envs = 2
    metric_keys = ("forearm_ratio", "upperarm_ratio")

    def reset(self, seeds):
        self.step_count = 0
        return np.zeros((2, 1))

    def step(self, actions):
        self.step_count += 1
        if self.step_count == 3:
            infos = [{"sim_error": True, "error": "RuntimeError('Decision ran past its budget')", "success": False,
                      "distance": float("nan")} for _ in range(2)]
            return np.zeros((2, 1)), np.zeros(2), np.ones(2, dtype=bool), infos
        infos = [{"success": False, "distance": 0.5, "forearm_ratio": 0.4 * self.step_count,
                  "upperarm_ratio": 0.1 * self.step_count + 0.1 * i} for i in range(2)]
        return np.zeros((2, 1)), np.zeros(2), np.zeros(2, dtype=bool), infos


def test_an_episode_cut_by_a_simulator_error_is_scored_at_its_last_decision():
    result = evaluate(TrippingEnv(), lambda obs, deterministic: obs, ObsSpec(3), SimpleNamespace(seed=0), 2)
    assert result["episodes"] == 2 and result["sim_errors"] == 2 and result["success_rate"] == 0.0
    # The second decision is the last one completed: upper-arm 0.2 and 0.3, forearm 0.8 for both.
    assert result["mean_final_upperarm_ratio"] == pytest.approx(0.25)
    assert result["mean_final_forearm_ratio"] == pytest.approx(0.8)


@pytest.mark.parametrize("requested, actual", [(1, 3), (4, 6)])
def test_evaluation_covers_each_slot_equally(requested, actual):
    result = evaluate(UnequalEpisodeEnv(), lambda obs, deterministic: obs, ObsSpec(3), SimpleNamespace(seed=0), requested)
    assert result["requested_episodes"] == requested and result["episodes"] == actual
    assert result["success_rate"] == pytest.approx(1 / 3)
    assert result["mean_final_forearm_ratio"] == pytest.approx(0.5)
    for i in range(3):
        assert result[f"episodes_garment_{i}"] == actual // 3


def test_training_budget_counts_only_admitted_curriculum_samples(monkeypatch, tmp_path):
    from uipc_manip import sac, train_sac

    class Agent:
        def __init__(self, spec, action_dim, cfg, device):
            self.cfg, self.updates = cfg, 0

        def act(self, obs, deterministic):
            return np.zeros((3, 1))

        def train(self, training):
            pass

        def save(self, path, step, metadata):
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    class Env:
        num_envs, obs_dim, action_dim = 3, 1, 1
        descriptions = [{"garment": g, "config": {}, "build_seconds": 0, "settle_displacement_m": 0}
                        for g in ("tshirt_26", "tshirt_392", "tshirt_68")]
        count = 0

        def reset(self, seeds):
            return np.zeros((3, 1))

        def step(self, actions):
            self.count += 1
            return np.zeros((3, 1)), np.ones(3), np.ones(3, dtype=bool), [{"success": False}] * 3

        def close(self):
            pass

    env = Env()
    monkeypatch.setattr(sac, "SACAgent", Agent)
    monkeypatch.setattr(train_sac, "make_env", lambda args: env)
    monkeypatch.setattr(train_sac, "evaluate", lambda *a, **kw: {"success_rate": 0, "mean_final_distance": 1, "mean_return": 0})
    train_sac.main(["--num-envs", "3", "--total-transitions", "6", "--garment-curriculum-interval", "100",
                    "--point-budget", "3", "--replay-capacity", "10", "--device", "cpu", "--log-interval", "1",
                    "--eval-freq", "0", "--checkpoint-interval", "0", "--work-dir", str(tmp_path)])
    assert env.count == 6  # Only the first garment's slot writes to replay.
    with (tmp_path / "dressing_sac_seed1" / "train_log.csv").open() as handle:
        last = list(csv.DictReader(handle))[-1]
    assert int(last["transitions"]) == 6 and int(last["simulated_transitions"]) == 18


def test_resume_refuses_a_changed_garment_placement(capsys):
    from types import SimpleNamespace

    from uipc_manip.train_sac import reconcile_resume_placement

    new = {"pre_insertion": True, "hang_as_baked": False, "hang_as_baked_garments": ["hospital_gown", "tshirt_68"], "sleeve_outward_garments": ["tshirt_4", "tshirt_392"]}
    old = {"pre_insertion": True, "hang_as_baked": False}  # written before the per-garment lists existed
    reconcile_resume_placement(SimpleNamespace(_resume_cell_plan={"live": dict(new)}, eval_only=False), new)
    reconcile_resume_placement(SimpleNamespace(_resume_cell_plan={}, eval_only=False), new)
    with pytest.raises(ValueError, match="placed its garments differently"):
        reconcile_resume_placement(SimpleNamespace(_resume_cell_plan={"live": old}, eval_only=False), new)
    reconcile_resume_placement(SimpleNamespace(_resume_cell_plan={"live": old}, eval_only=True), new)
    assert "sleeve_outward_garments [] -> ['tshirt_392', 'tshirt_4']" in capsys.readouterr().out


class _ResetEnv:
    """Two slots; slot 0 finishes every second step, slot 1 every fourth, and finished slots auto-reset."""

    num_envs, obs_dim, action_dim = 2, 1, 1
    descriptions = [{"garment": "tshirt_26", "config": {}, "build_seconds": 0, "settle_displacement_m": 0}] * 2
    metric_keys = ()

    def __init__(self):
        self.step_count = 0

    def reset(self, seeds):
        return np.zeros((2, 1))

    def step(self, actions):
        self.step_count += 1
        dones = np.array([self.step_count % 2 == 0, self.step_count % 4 == 0])
        infos = [{"success": False, "distance": 0.5, "garment": "tshirt_26"} for _ in range(2)]
        return np.zeros((2, 1)), np.zeros(2), dones, infos


def test_evaluation_resets_only_the_rollout_state_of_a_finished_slot():
    from uipc_manip.history import RolloutHistory

    masks = []

    def policy(obs, deterministic, history):
        masks.append(history.window(obs)[2].copy())
        history.push(obs, np.zeros((2, 1)))
        return np.zeros((2, 1))

    history = RolloutHistory(2, 1, 1, 3)
    history.push(np.zeros((2, 1)), np.zeros((2, 1)))  # stale state from before this evaluation
    evaluate(_ResetEnv(), policy, ObsSpec(3), SimpleNamespace(seed=0), 3, history=history)
    # Decision 1 starts empty despite the stale push; slot 0 restarts after its step-2 finish while
    # slot 1 keeps growing; both restart after step 4.
    assert [m[0].tolist() for m in masks[:4]] == [[False, False, True], [False, True, True], [False, False, True], [False, True, True]]
    assert [m[1].tolist() for m in masks[:4]] == [[False, False, True], [False, True, True], [True, True, True], [True, True, True]]
    assert masks[4][0].tolist() == masks[4][1].tolist() == [False, False, True]


def test_a_history_policy_needs_sequence_replay_and_then_trains_through_its_state(monkeypatch, tmp_path):
    from uipc_manip import sac, train_sac
    from uipc_manip.history import RolloutHistory

    seen: list = []

    class Agent:
        def __init__(self, spec, action_dim, cfg, device):
            self.cfg, self.updates = cfg, 0

        def make_history(self, num_streams):
            return RolloutHistory(num_streams, 1, 1, self.cfg.history_length)

        def act(self, obs, deterministic, history=None):
            assert history is not None, "a history-aware agent must be handed its rollout state"
            seen.append(history.window(obs)[2].copy())
            action = np.zeros((3, 1))
            history.push(obs, action)
            return action

        def update(self, replay):
            self.updates += 1
            batch = replay.sample_sequences(self.cfg.history_length, 2, pad=True)
            assert batch.valid[:, -1].all()
            return {}

        def train(self, training):
            pass

        def save(self, path, step, metadata):
            path.parent.mkdir(parents=True, exist_ok=True)
            return path

    class Env:
        num_envs, obs_dim, action_dim = 3, 1, 1
        descriptions = [{"garment": "tshirt_26", "config": {}, "build_seconds": 0, "settle_displacement_m": 0}] * 3
        count = 0

        def reset(self, seeds):
            return np.zeros((3, 1))

        def step(self, actions):
            self.count += 1
            done = self.count % 3 == 0
            return np.zeros((3, 1)), np.ones(3), np.full(3, done), [{"success": False}] * 3

        def close(self):
            pass

    built: list = []
    monkeypatch.setattr(sac, "SACAgent", Agent)
    monkeypatch.setattr(train_sac, "make_env", lambda args: built.append(Env()) or built[-1])
    monkeypatch.setattr(train_sac, "evaluate", lambda *a, **kw: {"success_rate": 0, "mean_final_distance": 1, "mean_return": 0})
    argv = ["--num-envs", "3", "--total-transitions", "18", "--point-budget", "3", "--replay-capacity", "64", "--device", "cpu",
            "--log-interval", "1", "--eval-freq", "0", "--checkpoint-interval", "0", "--init-steps", "2", "--history-length", "2",
            "--work-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        train_sac.main(argv)
    assert not built, "the refusal must come before a world is built"
    train_sac.main(argv + ["--sequence-replay"])
    # Two warm-up commands entered the history: the first policy decision already sees a full prefix,
    # and the episode end after step 3 empties it again.
    assert seen[0][:, 0].tolist() == [True, True, True]
    assert seen[1][:, 0].tolist() == [False, False, False]


def test_rlt_history_knobs_reach_the_agent_config_and_old_checkpoints_default_them():
    from uipc_manip.sac import SACConfig

    horizon = ["--horizon", "150"]
    args = build_parser().parse_args([*horizon, "--history-length", "8", "--history-kind", "rlt", "--rlt-dim", "32", "--rlt-window", "4", "--rlt-tied"])
    cfg = build_sac_config(args)
    assert cfg.history_length == 8 and cfg.history_kind == "rlt"
    assert (cfg.rlt.dim, cfg.rlt.window, cfg.rlt.tied, cfg.rlt.layers) == (32, 4, True, 2)
    assert SACConfig.from_dict(cfg.to_dict()) == cfg
    old = build_sac_config(build_parser().parse_args(horizon)).to_dict()
    del old["rlt"], old["history_kind"]
    restored = SACConfig.from_dict(old)
    assert restored.history_kind == "frames" and restored.rlt == SACConfig().rlt


def test_the_representation_flag_initialises_a_fresh_run_and_refuses_a_resume(monkeypatch, tmp_path):
    from uipc_manip import sac, train_sac

    saved: list = []

    class Agent:
        def __init__(self, spec, action_dim, cfg, device):
            self.cfg, self.updates, self.initialized = cfg, 0, None

        def initialize_representation(self, path):
            self.initialized = str(path)
            return {"checkpoint": str(path), "sha256": "stub", "components": ["actor.encoder"], "pretraining": {"steps": 1}}

        def act(self, obs, deterministic, history=None):
            return np.zeros((3, 1))

        def update(self, replay):
            self.updates += 1
            return {}

        def train(self, training):
            pass

        def save(self, path, step, metadata):
            path.parent.mkdir(parents=True, exist_ok=True)
            saved.append(metadata)
            return path

    class Env:
        num_envs, obs_dim, action_dim = 3, 1, 1
        descriptions = [{"garment": "tshirt_26", "config": {}, "build_seconds": 0, "settle_displacement_m": 0}] * 3

        def reset(self, seeds):
            return np.zeros((3, 1))

        def step(self, actions):
            return np.zeros((3, 1)), np.ones(3), np.zeros(3, dtype=bool), [{"success": False}] * 3

        def close(self):
            pass

    built: list = []
    monkeypatch.setattr(sac, "SACAgent", Agent)
    monkeypatch.setattr(train_sac, "make_env", lambda args: built.append(Env()) or built[-1])
    monkeypatch.setattr(train_sac, "evaluate", lambda *a, **kw: {"success_rate": 0, "mean_final_distance": 1, "mean_return": 0})
    argv = ["--num-envs", "3", "--total-transitions", "6", "--point-budget", "3", "--replay-capacity", "64", "--device", "cpu",
            "--log-interval", "1", "--eval-freq", "0", "--checkpoint-interval", "0", "--init-steps", "2", "--work-dir", str(tmp_path)]
    with pytest.raises(SystemExit, match="no such checkpoint"):
        train_sac.main(argv + ["--init-representation", str(tmp_path / "typo.pt")])
    torch.save({"metadata": {}}, tmp_path / "online.pt")
    with pytest.raises(SystemExit, match="not an offline pretraining checkpoint"):
        train_sac.main(argv + ["--init-representation", str(tmp_path / "online.pt")])
    assert not built, "a bad checkpoint must be refused before a world is built"
    torch.save({"metadata": {"pretraining": {"steps": 1}}}, tmp_path / "rep.pt")
    train_sac.main(argv + ["--init-representation", str(tmp_path / "rep.pt")])
    assert saved and saved[-1]["representation_init"]["checkpoint"] == str(tmp_path / "rep.pt")
    with pytest.raises(ValueError, match="fresh run"):
        train_sac.main(argv + ["--init-representation", str(tmp_path / "rep.pt"), "--resume", str(tmp_path / "nothing.pt")])
