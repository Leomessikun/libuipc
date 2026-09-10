"""CPU tests of the one-policy-for-many-cells wiring in the SAC trainer."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from uipc_manip import train_sac
from uipc_manip.dressing_env import DressingConfig
from uipc_manip.sac import SACConfig
from uipc_manip.train_sac import build_parser, cell_bodies, plan_cells, resolve_defaults, restore_resume_args

GARMENTS = ("tshirt_26", "tshirt_392")
HELDOUT_BODY = 2
TRAIN_ARGV = [
    "--cell-source", "live", "--body-seeds", "0,1,2", "--heldout-bodies", "1", "--garments", *GARMENTS,
    "--num-envs", "6", "--total-transitions", "16", "--point-budget", "3", "--replay-capacity", "64",
    "--device", "cpu", "--log-interval", "1", "--eval-freq", "2", "--checkpoint-interval", "0",
]


def _args(argv):
    args = build_parser().parse_args(argv)
    resolve_defaults(args)
    return args


def _grid(args, cfg=None):
    return [(g, b) for g in args.garments for b in cell_bodies(args)]


class CellEnv:
    """Stub world built from ``DressingConfig.cells``: slot i observes i; the held-out body never dresses."""

    obs_dim, action_dim = 1, 1
    metric_keys = ("upperarm_ratio", "forearm_ratio")

    def __init__(self, cfg, num_envs):
        assert len(cfg.cells) == num_envs and cfg.cell_source == "live"
        self.num_envs, self.cells, self.t = num_envs, [tuple(c) for c in cfg.cells], 0
        self.descriptions = [
            {"garment": g, "human": b, "cell": f"{g}__human_{b}", "config": cfg.to_dict(),
             "build_seconds": 0.0, "settle_displacement_m": 0.0}
            for g, b in self.cells
        ]

    def _obs(self):
        return np.arange(self.num_envs, dtype=np.float32)[:, None]

    def reset(self, seeds=None):
        self.t = 0
        return self._obs()

    def step(self, actions):
        self.t += 1
        infos = []
        for g, b in self.cells:
            ratio = 0.1 if b == HELDOUT_BODY else 0.9
            infos.append({"success": ratio >= 0.7, "garment": g, "human": b, "cell": f"{g}__human_{b}",
                          "distance": 1.0 - ratio, "upperarm_ratio": ratio, "forearm_ratio": 1.0})
        return self._obs(), np.ones(self.num_envs, dtype=np.float32), np.full(self.num_envs, self.t % 2 == 0), infos

    def close(self):
        pass


@pytest.fixture
def stub_world(monkeypatch):
    """Real make_env/main over a stub world, agent and recording replay; yields the replayed slot ids and saves."""
    from uipc_manip import replay as replay_module
    from uipc_manip import sac

    record = {"replayed_slots": [], "saves": {}}

    class Agent:
        def __init__(self, spec, action_dim, cfg, device):
            self.cfg, self.updates = cfg, 0

        def act(self, obs, deterministic):
            return np.zeros((obs.shape[0], 1), dtype=np.float32)

        def update(self, replay):
            self.updates += 1
            return {}

        def train(self, training):
            pass

        def save(self, path, step, metadata):
            record["saves"][path.name] = {"step": step, "sac_config": self.cfg.to_dict(), "metadata": metadata}
            return path

    class RecordingReplay(replay_module.FlatReplayBuffer):
        def add(self, obs, action, reward, next_obs, done):
            record["replayed_slots"].append(int(np.asarray(obs).reshape(-1)[0]))
            super().add(obs, action, reward, next_obs, done)

    monkeypatch.setattr(sac, "SACAgent", Agent)
    monkeypatch.setattr(replay_module, "FlatReplayBuffer", RecordingReplay)
    monkeypatch.setattr(train_sac, "GenesisIPCDressingEnv", CellEnv)
    monkeypatch.setattr(train_sac, "library_cells", _grid)
    return record


def test_heldout_slots_never_reach_replay_and_evaluation_reports_both_subsets(stub_world, tmp_path, capsys):
    train_sac.main([*TRAIN_ARGV, "--work-dir", str(tmp_path), "--run-name", "cells"])
    out = capsys.readouterr().out
    assert ("[uipc-manip distribution] source=live slots=6 unique_cells=6 training_cells=4 heldout_cells=2 "
            "garments=['tshirt_26', 'tshirt_392'] bodies=[0, 1, 2] heldout_bodies=[2]") in out

    # Slots 0-1 hold body 2 and step every time, yet not one of their transitions was replayed.
    assert sorted(set(stub_world["replayed_slots"])) == [2, 3, 4, 5]
    assert len(stub_world["replayed_slots"]) == 16  # the budget counts training transitions only
    run = tmp_path / "cells"
    with (run / "train_log.csv").open() as handle:
        last = list(csv.DictReader(handle))[-1]
    assert int(last["transitions"]) == 16 and int(last["simulated_transitions"]) == 24
    assert float(last["episode_success"]) == 1.0 and float(last["heldout_episode_success"]) == 0.0

    with (run / "eval_log.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["step"]) for r in rows] == [2, 4]
    row = rows[-1]
    assert int(row["episodes"]) == 6 and int(row["cell_count"]) == 6  # one deterministic episode per cell
    assert int(row["heldout_episode_count"]) == 2 and int(row["training_episode_count"]) == 4
    assert float(row["heldout_success_rate"]) == 0.0 and float(row["training_success_rate"]) == 1.0
    assert int(row["heldout_zero_cell_count"]) == 2 and float(row["training_worst_cell_success_rate"]) == 1.0
    assert float(row["heldout_mean_final_upperarm_ratio"]) == pytest.approx(0.1)
    assert float(row["success_rate_tshirt_392|human=2"]) == 0.0 and float(row["success_rate_tshirt_26|human=0"]) == 1.0
    assert "mean_final_upperarm_ratio_tshirt_26|human=1" in row and "success_rate_tshirt_26" in row

    best = stub_world["saves"]["best.pt"]["metadata"]
    assert best["cells"][:2] == [["tshirt_26", 2], ["tshirt_392", 2]] and len(best["cells"]) == 6
    assert best["heldout_cells"] == [["tshirt_26", 2], ["tshirt_392", 2]] and best["heldout_bodies"] == [2]
    assert best["heldout_slots"] == [0, 1] and best["cell_source"] == "live" and isinstance(best["live"], dict)
    assert best["training_args"]["body_seeds"] == [0, 1, 2] and best["training_args"]["heldout_bodies"] == 1
    assert best["curriculum_order"] == ["tshirt_26", "tshirt_392"]
    assert best["eval"]["heldout_success_rate"] == 0.0
    assert json.loads((run / "config.json").read_text())["heldout_bodies"] == [2]


@pytest.mark.parametrize("argv, heldout", [
    (["--cell-source", "live", "--body-seeds", "0,1,2,3"], [2, 3]),
    (["--cell-source", "live", "--body-seeds", "0,1"], [1]),  # the default always leaves a training body
    (["--cell-source", "cache", "--body-seeds", "0,1,2,3"], [3]),
    (["--cell-source", "cache", "--human", "3"], []),  # a regional teacher holds nothing out
    (["--cell-source", "live", "--body-seeds", "0,1,2,3", "--heldout-body-seeds", "0"], [0]),
])
def test_heldout_defaults_and_explicit_seeds(argv, heldout):
    args = _args([*argv, "--garments", *GARMENTS, "--num-envs", "8"])
    plan = plan_cells(args, _grid(args))
    assert plan["heldout_bodies"] == heldout
    pinned = [tuple(c) for c in plan["cells"][: len(plan["heldout_slots"])]]
    assert pinned == sorted((g, b) for g in GARMENTS for b in heldout)


def test_launcher_fails_fast_on_partial_coverage_unless_allowed():
    library = [(g, b) for g in GARMENTS for b in range(4)]  # 8 cells
    argv = ["--cell-source", "live", "--body-seeds", "0,1,2,3", "--garments", *GARMENTS]
    with pytest.raises(ValueError, match="8 cells do not fit in 6 slots"):
        plan_cells(_args([*argv, "--num-envs", "6"]), library)
    with pytest.raises(ValueError, match="4 evaluation episodes cannot score 8 cells"):
        plan_cells(_args([*argv, "--num-envs", "8", "--num-eval-episodes", "4"]), library)
    with pytest.raises(ValueError, match="no cell for garments"):
        plan_cells(_args([*argv, "tshirt_4", "--num-envs", "8"]), library)
    with pytest.raises(ValueError, match="not complete rows"):
        plan_cells(_args([*argv, "--num-envs", "8", "--heldout-body-seeds", "3"]), library[:-1])
    assert len(plan_cells(_args([*argv, "--num-envs", "6", "--allow-partial-cell-coverage"]), library)["cells"]) == 6
    # The single-body path keeps cycling its garments over every slot without a flag, and --vis
    # (which forces one slot) implies the escape hatch.
    single = [(g, 0) for g in GARMENTS]
    plan = plan_cells(_args(["--cell-source", "cache", "--garments", *GARMENTS, "--num-envs", "16"]), single)
    assert plan["heldout_slots"] == [] and [tuple(c) for c in plan["cells"]] == single * 8
    assert plan_cells(_args(["--garments", *GARMENTS, "--num-envs", "1", "--vis"]), single)["cells"] == [["tshirt_26", 0]]


def _trained_payload(stub_world, tmp_path):
    train_sac.main([*TRAIN_ARGV, "--work-dir", str(tmp_path), "--run-name", "cells"])
    return stub_world["saves"]["best.pt"]


def test_training_resume_keeps_its_plan_and_refuses_another(stub_world, tmp_path):
    payload = _trained_payload(stub_world, tmp_path)
    args = build_parser().parse_args([])
    restore_resume_args(args, [], payload)
    resolve_defaults(args)
    assert (args.cell_source, args.body_seeds, args.heldout_bodies, args.num_envs) == ("live", [0, 1, 2], 1, 6)
    train_sac.make_env(args)
    assert args._cell_plan["cells"] == payload["metadata"]["cells"] and args._cell_plan["heldout_bodies"] == [2]

    # An explicit change of the body axis is refused before planning ...
    argv = ["--body-seeds", "0,1"]
    with pytest.raises(ValueError, match="body-seeds"):
        restore_resume_args(build_parser().parse_args(argv), argv, payload)
    # ... and so is a plan that differs although the options match (another held-out row).
    payload["metadata"] = {**payload["metadata"], "heldout_bodies": [0]}
    args = build_parser().parse_args([])
    restore_resume_args(args, [], payload)
    resolve_defaults(args)
    with pytest.raises(ValueError, match="another cell plan.*held-out bodies \\[0\\] -> \\[2\\]"):
        train_sac.make_env(args)


def test_eval_only_on_other_bodies_prints_the_difference_and_scores_unseen_cells_as_heldout(stub_world, tmp_path, capsys):
    payload = _trained_payload(stub_world, tmp_path)
    capsys.readouterr()
    # Bodies 1 (trained), 2 (held out in training) and 3 (never seen); the saved
    # --heldout-bodies belongs to the old axis and gives way to this axis's defaults.
    argv = ["--eval-only", "--body-seeds", "1,2,3"]
    args = build_parser().parse_args(argv)
    restore_resume_args(args, argv, payload)
    resolve_defaults(args)
    train_sac.make_env(args)
    assert "eval-only cell plan differs from the checkpoint" in capsys.readouterr().out
    plan = args._cell_plan
    cells = [tuple(c) for c in plan["cells"]]
    assert {cells[i][1] for i in plan["heldout_slots"]} == {2, 3}
    assert {b for i, (_, b) in enumerate(cells) if i not in plan["heldout_slots"]} == {1}
    assert plan["heldout_bodies"] == [2, 3]


def test_legacy_checkpoint_resumes_on_the_bake_cache():
    env = DressingConfig(horizon=150, action_repeat=6).to_dict()
    for key in ("cell_source", "cells", "live"):
        env.pop(key)
    payload = {"step": 20, "sac_config": SACConfig().to_dict(), "metadata": {"task": "dressing", "env": env}}
    args = build_parser().parse_args([])
    restore_resume_args(args, [], payload)
    assert args.cell_source == "cache" and args._resume_cell_plan == {}
