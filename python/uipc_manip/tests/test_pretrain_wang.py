"""CPU tests of Wang's pretraining protocol: distribution, rotation, replay split, temperatures, and the run loop."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from uipc_manip import pretrain_wang, sac  # noqa: E402
from uipc_manip.dressing_body import pose_region  # noqa: E402
from uipc_manip.dressing_live import NoClearPlacement  # noqa: E402
from uipc_manip.models import EncoderConfig  # noqa: E402
from uipc_manip.obs import ObsSpec  # noqa: E402
from uipc_manip.pretrain_wang import ConfigPool, matched_physics, prepare, region_body, region_configs  # noqa: E402
from uipc_manip.replay import FlatReplayBuffer, ReplaySet  # noqa: E402
from uipc_manip.sac import SACAgent, SACConfig  # noqa: E402

GARMENTS = ("tshirt_26", "tshirt_68")
FIVE = ["hospital_gown", "tshirt_26", "tshirt_68", "tshirt_4", "tshirt_392"]


def test_region_bodies_and_wangs_pose_split():
    assert region_body(13, 0) == 14000 and region_body(26, 49) == 27049
    assert all(pose_region(region_body(r, p)) == r for r in range(27) for p in range(50))
    with pytest.raises(ValueError):
        region_body(27, 0)
    with pytest.raises(ValueError):
        region_body(0, 50)
    _, _, plan = prepare(["teacher", "--region", "13"])
    assert plan["garments"] == FIVE and plan["train_poses"] == list(range(45)) and plan["eval_poses"] == list(range(45, 50))
    assert len(plan["eval_configs"]) == 25 and {b for _, b in plan["eval_configs"]} == set(range(14045, 14050))
    assert region_configs([13], ["tshirt_26"], [0, 1]) == [("tshirt_26", 14000), ("tshirt_26", 14001)]
    with pytest.raises(ValueError, match="both trained on and held out"):
        prepare(["teacher", "--region", "13", "--train-poses", "0", "45"])


def test_rotation_draw_deals_groups_over_slots_and_poses_without_replacement():
    pool = ConfigPool([13], GARMENTS)
    rng = np.random.default_rng(0)
    counts: dict[tuple[str, int], int] = {}
    for _ in range(200):
        cells = pool.draw(24, rng)
        for garment in GARMENTS:
            bodies = [b for g, b in cells if g == garment]
            assert len(bodies) == 12 and len(set(bodies)) == 12 and all(14000 <= b < 14045 for b in bodies)
        for cell in cells:
            counts[cell] = counts.get(cell, 0) + 1
    # Every one of the 90 training configurations recurs at about Wang's uniform rate (4800 / 90 = 53).
    assert len(counts) == 90 and min(counts.values()) > 30 and max(counts.values()) < 80
    assert {g for g, _ in pool.draw(24, rng, garments=["tshirt_68"])} == {"tshirt_68"}
    student = ConfigPool([4, 13, 22], GARMENTS).draw(24, rng)
    assert {pose_region(b) for _, b in student} == {4, 13, 22}
    assert all(sum(1 for g, b in student if (pose_region(b), g) == key) == 4 for key in ConfigPool([4, 13, 22], GARMENTS).groups)
    pool.drop("tshirt_26", 14003, "refused")
    assert ("tshirt_26", 14003) not in pool.configs() and pool.dropped == [{"garment": "tshirt_26", "body": 14003, "reason": "refused"}]
    single = ConfigPool([13], ["tshirt_26"], poses=[0])
    with pytest.raises(RuntimeError, match="No pose of region 13"):
        single.drop("tshirt_26", 14000, "refused")


def test_matched_physics_keeps_the_decision_the_hold_and_the_settle():
    assert matched_physics(1.0 / 60.0) == {"action_repeat": 6, "cuff_strength": pytest.approx(1.0e4), "settle_steps": 30}
    assert matched_physics(1.0 / 30.0) == {"action_repeat": 3, "cuff_strength": pytest.approx(4.0e4), "settle_steps": 15}
    with pytest.raises(ValueError, match="does not divide"):
        matched_physics(0.03)


def test_stage_command_lines_generate_the_trainer_settings():
    args, targs, plan = prepare(["teacher", "--region", "13", "--hidden-dim", "32", "--encoder-precision", "bf16"])
    assert (targs.task, targs.cell_source, list(targs.garments)) == ("dressing", "live", FIVE)
    assert (targs.horizon, targs.action_repeat, targs.dt, targs.cuff_strength) == (300, 6, 1.0 / 60.0, 1.0e4)
    assert (targs.replay_capacity, targs.batch_size, targs.updates_per_step, targs.num_envs) == (400_000, 64, 0, 24)
    assert targs.settle_steps is None and targs.teacher_checkpoints is None
    assert (targs.hidden_dim, targs.encoder_precision, targs.run_name) == (32, "bf16", "wang_teacher_r13_s1")
    assert (plan["replay_split"], plan["buffer_keys"], plan["temperatures"], plan["temperature_count"]) == ("none", [0], "shared", 1)
    _, targs, _ = prepare(["teacher", "--region", "4", "--dt", repr(1.0 / 30.0)])
    assert (targs.action_repeat, targs.cuff_strength, targs.settle_steps) == (3, pytest.approx(4.0e4), 15)
    _, targs, _ = prepare(["teacher", "--region", "4", "--dt", repr(1.0 / 30.0), "--cuff-strength", "2e4"])
    assert targs.cuff_strength == 2.0e4  # an explicit pass-through flag wins
    _, targs, plan = prepare(["student", "--regions", "22", "4", "13", "--teacher-checkpoints", "a.pt", "b.pt", "c.pt"])
    assert plan["regions"] == [4, 13, 22] and plan["buffer_keys"] == [4, 13, 22] and plan["temperature_count"] == 1
    assert targs.teacher_checkpoints == ["a.pt", "b.pt", "c.pt"] and targs.distill_weight == 0.01
    assert targs.run_name == "wang_student_r4-13-22_s1" and len(plan["eval_configs"]) == 75
    _, _, plan = prepare(["student", "--regions", "4", "13", "--teacher-checkpoints", "a.pt", "b.pt", "--temperatures", "per-buffer"])
    assert plan["temperature_count"] == 2
    with pytest.raises(ValueError, match="per-buffer needs"):
        prepare(["teacher", "--region", "13", "--temperatures", "per-buffer"])
    for flag in ("--num-eval-episodes", "--total-transitions", "--body-seeds", "--eval-freq"):
        with pytest.raises(ValueError, match="sets"):
            prepare(["teacher", "--region", "13", flag, "5"])


def test_replay_set_routes_splits_capacity_and_reports_the_buffer(tmp_path):
    replay = ReplaySet(GARMENTS, 1, 1, 10, 2, "cpu")
    assert [b.capacity for b in replay.buffers] == [5, 5] and replay.indexed
    for value in (0.0, 1.0, 2.0):
        replay.add("tshirt_26", [value], [0.0], 0.0, [value], False)
    replay.add("tshirt_68", [100.0], [0.0], 0.0, [100.0], False)
    for _ in range(20):  # only the first buffer holds more than a batch
        batch = replay.sample()
        assert batch[-1] == 0 and float(batch[0].max()) < 100.0
    for _ in range(2):
        replay.add("tshirt_68", [100.0], [0.0], 0.0, [100.0], False)
    seen = set()
    for _ in range(60):
        batch = replay.sample()
        seen.add(batch[-1])
        assert (float(batch[0].min()) >= 100.0) == (batch[-1] == 1)  # a batch is one buffer's
    assert seen == {0, 1} and replay.total_added == 6 and replay.size == 6 and replay.sizes() == [3, 3]
    replay.save(tmp_path / "set", metadata={"transitions": 6})
    again = ReplaySet(GARMENTS, 1, 1, 10, 2, "cpu")
    assert again.load(tmp_path / "set") == {"transitions": 6} and again.sizes() == [3, 3]
    with pytest.raises(ValueError, match="holds buffers"):
        ReplaySet(("tshirt_68", "tshirt_26"), 1, 1, 10, 2, "cpu").load(tmp_path / "set")
    with pytest.raises(RuntimeError, match="more than"):
        ReplaySet(GARMENTS, 1, 1, 10, 2, "cpu").sample()
    with pytest.raises(ValueError, match="distinct"):
        ReplaySet(["a", "a"], 1, 1, 10, 2, "cpu")


def _tiny_cfg(**kw) -> SACConfig:
    cfg = SACConfig(hidden_dim=32, batch_size=8, actor_update_freq=1, actor_type="flat", **kw)
    cfg.encoder = EncoderConfig(
        kind="pointnet2", sa_mlp=[[16, 16], [16, 16], [16, 32]], fp_mlp=[[16, 16], [16, 8], [8, 8]], linear_mlp=[16],
        output_dim=8, sa_neighbors=[6, 6], transformer_dim=16, transformer_heads=2, transformer_layers=1,
    )
    return cfg


def _obs(spec: ObsSpec, rng) -> np.ndarray:
    marker = np.array([True, True, False, False, False, False])
    return spec.pack(rng.normal(scale=0.01, size=(6, 3)), marker, rng.uniform(-0.05, 0.05, 3), np.zeros(3), True)


def test_per_buffer_temperatures_follow_the_buffer_a_batch_came_from(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    spec = ObsSpec(10)
    agent = SACAgent(spec, 3, _tiny_cfg(temperature_count=2), "cpu")
    assert tuple(agent.log_alpha.shape) == (2,)
    replay = ReplaySet(["a", "b"], spec.dim, 3, 64, 8, "cpu")
    for _ in range(12):
        replay.add("b", _obs(spec, rng), rng.uniform(-1, 1, 3), 0.1, _obs(spec, rng), False)
    before = agent.log_alpha.detach().clone()
    stats = agent.update(replay)  # buffer "a" is empty, so the batch and the temperature are buffer 1's
    after = agent.log_alpha.detach().clone()
    assert after[0] == before[0] and after[1] != before[1]
    assert stats["alpha_1"] == pytest.approx(float(after[1].exp())) and stats["alpha"] == stats["alpha_1"]
    agent.save(tmp_path / "agent.pt", 1)
    again = SACAgent(spec, 3, _tiny_cfg(temperature_count=2), "cpu")
    again.load(tmp_path / "agent.pt")
    assert torch.equal(again.log_alpha.detach(), after)
    flat = FlatReplayBuffer(spec.dim, 3, 64, 8, "cpu")
    for _ in range(12):
        flat.add(_obs(spec, rng), rng.uniform(-1, 1, 3), 0.1, _obs(spec, rng), False)
    with pytest.raises(ValueError, match="replay set"):
        agent.update(flat)
    # One shared temperature, Wang's student, reads a replay set's batches and keeps the 0-d shape.
    shared = SACAgent(spec, 3, _tiny_cfg(), "cpu")
    shared.update(replay)
    assert shared.log_alpha.dim() == 0 and "alpha_1" not in shared.update(replay)
    legacy = {k: v for k, v in SACConfig().to_dict().items() if k != "temperature_count"}
    assert SACConfig.from_dict(legacy).temperature_count == 1


# ------------------------------------------------------------------ the run loop on a stub world
REFUSED = {("tshirt_68", 14001), ("tshirt_26", 14046)}
FAIL_BUILDS: list = []  # each entry makes the next stub world refuse to build, as libuipc refuses an invalid one
TRIPS: list = []  # each entry ends one training episode on a simulator error at its last step, as the watchdog does
RUN_ARGV = [
    "teacher", "--region", "13", "--garments", *GARMENTS, "--train-poses", "0", "1", "--eval-poses", "45", "46",
    "--num-envs", "4", "--horizon", "2", "--eval-every", "8", "--checkpoint-every", "8", "--replay-capacity", "64",
    "--batch-size", "2", "--replay-split", "garment", "--device", "cpu", "--log-interval", "1", "--point-budget", "3",
]


@pytest.fixture
def stub_run(monkeypatch):
    """The real run loop over stub worlds, placement and agent; yields every world built."""
    TRIPS.clear()
    built: list = []
    agents: list = []

    class Factory:
        def __init__(self, cfg=None):
            self.cfg = cfg

        def clearance_for(self, garment, body):
            if (garment, int(body)) in REFUSED:
                raise NoClearPlacement(f"{garment} on body {body}: refused by the test")
            return 0.09

    class World:
        obs_dim, action_dim, privileged_dim = 1, 1, 0
        metric_keys = ("upperarm_ratio", "forearm_ratio")

        def __init__(self, cfg, num_envs, cell_factory=None):
            assert isinstance(cell_factory, Factory) and len(cfg.cells) == num_envs
            if FAIL_BUILDS:
                FAIL_BUILDS.pop()
                raise RuntimeError("IPC world is invalid after build")
            self.cells = [(str(g), int(b)) for g, b in cfg.cells]
            self.num_envs, self.horizon, self.t, self.closed = num_envs, cfg.horizon, 0, False
            self.descriptions = [
                {"garment": g, "human": b, "pose_region": pose_region(b), "config": cfg.to_dict(), "build_seconds": 0.0,
                 "settle_displacement_m": 0.0}
                for g, b in self.cells
            ]
            built.append(self)

        def _obs(self):
            return np.array([[GARMENTS.index(g) * 1000 + b % 1000] for g, b in self.cells], dtype=np.float32)

        def reset(self, seeds=None):
            self.t = 0
            return self._obs()

        def step(self, actions):
            assert not self.closed
            self.t += 1
            if TRIPS and self.t >= self.horizon and all(b % 1000 < 45 for _, b in self.cells):
                TRIPS.pop()
                self.t = 0
                tripped = [{"sim_error": True, "error": "RuntimeError('Decision ran past its budget')", "success": False,
                            "distance": float("nan")} for _ in self.cells]
                return self._obs(), np.zeros(self.num_envs, dtype=np.float32), np.ones(self.num_envs, dtype=bool), tripped
            infos = []
            for g, b in self.cells:
                ratio = 0.9 if g == "tshirt_26" else 0.2
                infos.append({"success": ratio >= 0.7, "garment": g, "human": b, "cell": f"{g}__human_{b}",
                              "distance": 1.0 - ratio, "upperarm_ratio": ratio, "forearm_ratio": 1.0})
            done = self.t >= self.horizon
            if done:
                self.t = 0
            return self._obs(), np.ones(self.num_envs, dtype=np.float32), np.full(self.num_envs, done), infos

        def close(self):
            self.closed = True

    class Agent:
        def __init__(self, spec, action_dim, cfg, device):
            self.cfg, self.updates, self.alpha, self.batches, self.teachers = cfg, 0, torch.zeros(()), [], {}
            agents.append(self)

        def set_teachers(self, teachers):
            self.teachers = dict(teachers)

        def act(self, obs, deterministic):
            return np.zeros((obs.shape[0], 1), dtype=np.float32)

        def update(self, replay):
            batch = replay.sample(self.cfg.batch_size)
            # (buffer index, garments of the batch's rows, their region labels when the replay carries them)
            labels = sorted({int(v) for v in batch[-2].tolist()}) if replay.labelled else None
            self.batches.append((batch[-1], sorted({int(v) // 1000 for v in batch[0].reshape(-1).tolist()}), labels))
            self.updates += 1
            return {}

        def train(self, training):
            pass

        def save(self, path, step, metadata=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"step": step, "metadata": metadata or {}, "sac_config": self.cfg.to_dict()}, default=str))
            return path

        def load(self, path, load_optimizers=True):
            payload = json.loads(Path(path).read_text())
            assert load_optimizers and payload["sac_config"] == json.loads(json.dumps(self.cfg.to_dict()))
            return payload

    monkeypatch.setattr(pretrain_wang, "_ensure_genesis", lambda level: None)
    monkeypatch.setattr(pretrain_wang, "available_garments", lambda cfg: list(GARMENTS))
    monkeypatch.setattr(pretrain_wang, "LiveCellFactory", Factory)
    monkeypatch.setattr(pretrain_wang, "GenesisIPCDressingEnv", World)
    monkeypatch.setattr(sac, "SACAgent", Agent)
    return built, agents


def _training_worlds(built):
    return [w for w in built if all(b % 1000 < 45 for _, b in w.cells)]


def test_run_rotates_worlds_evaluates_held_out_poses_and_resumes_the_same_draws(stub_run, tmp_path):
    built, agents = stub_run
    pretrain_wang.main(RUN_ARGV + ["--transitions", "24", "--work-dir", str(tmp_path), "--run-name", "full"])
    worlds = _training_worlds(built)
    held = [w for w in built if w not in worlds]
    # Three episodes of 8 transitions: the first world and one rotation after each episode but the last.
    assert len(worlds) == 3 and all(w.closed for w in built)
    for w in worlds:
        # Two poses per garment: the refused one is dropped the first time it is drawn, so tshirt_68 repeats its other pose.
        assert sorted(w.cells) == [("tshirt_26", 14000), ("tshirt_26", 14001), ("tshirt_68", 14000), ("tshirt_68", 14000)]
    assert len(held) == 1 and held[0].cells == [("tshirt_26", 14045), ("tshirt_68", 14045), ("tshirt_68", 14046)]
    # Each update's batch comes from one garment's buffer, and both buffers are drawn.
    batches = agents[0].batches
    assert batches and all(garments == [index] for index, garments, _ in batches) and {i for i, _, _ in batches} == {0, 1}
    run = tmp_path / "full"
    rows = list((run / "eval_log.csv").read_text().splitlines())
    header = rows[0].split(",")
    evals = [dict(zip(header, r.split(","), strict=False)) for r in rows[1:]]
    assert [int(e["transitions"]) for e in evals] == [0, 8, 16, 24]
    assert float(evals[-1]["heldout_mean_final_upperarm_ratio"]) == pytest.approx((0.9 + 0.2 + 0.2) / 3)
    # Checkpoints name the training pool with no held-out slot, which load_teachers reads a teacher's region from.
    best = json.loads((run / "checkpoints" / "best.pt").read_text())["metadata"]
    last = json.loads((run / "checkpoints" / "checkpoint_00000024.pt").read_text())["metadata"]
    assert best["heldout_slots"] == [] and "eval" in best and best["cells"] == last["cells"]
    assert last["cells"] == [["tshirt_26", 14000], ["tshirt_26", 14001], ["tshirt_68", 14000]]
    assert [d["body"] for d in last["dropped_configs"]] == [14001] and last["dropped_eval_configs"][0]["body"] == 14046
    state = json.loads((run / "checkpoints" / "state.json").read_text())
    assert state["transitions"] == 24 and state["counters"]["rotations"] == 3 and state["counters"]["episodes"] == 3
    assert (run / "checkpoints" / "replay_latest" / "replay_set.json").exists()

    # Stopped at 16 and resumed to 24, the run builds the same worlds as the uninterrupted one.
    full = [w.cells for w in worlds]
    built.clear()
    pretrain_wang.main(RUN_ARGV + ["--transitions", "16", "--work-dir", str(tmp_path), "--run-name", "part"])
    assert [w.cells for w in _training_worlds(built)] == full[:2]
    built.clear()
    pretrain_wang.main(["resume", str(tmp_path / "part"), "--transitions", "24"])
    assert [w.cells for w in _training_worlds(built)] == full[2:]
    state_path = tmp_path / "part" / "checkpoints" / "state.json"
    state = json.loads(state_path.read_text())
    assert state["transitions"] == 24 and state["counters"]["rotations"] == 3
    state["plan"]["num_envs"] = 8
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="num_envs"):
        pretrain_wang.main(["resume", str(tmp_path / "part")])


def test_episodes_that_all_end_on_a_simulator_error_still_evaluate_and_checkpoint(stub_run, tmp_path):
    # Every episode trips at its last step, so each world adds one step of 4 transitions and none finishes.
    TRIPS.extend([1] * 8)
    pretrain_wang.main(RUN_ARGV + ["--transitions", "16", "--work-dir", str(tmp_path), "--run-name", "tripped"])
    run = tmp_path / "tripped"
    rows = (run / "eval_log.csv").read_text().splitlines()
    header = rows[0].split(",")
    assert [int(dict(zip(header, r.split(","), strict=False))["transitions"]) for r in rows[1:]] == [0, 8, 16]
    assert (run / "checkpoints" / "checkpoint_00000008.pt").exists()
    state = json.loads((run / "checkpoints" / "state.json").read_text())
    # The fourth world reaches the budget on its first step, before the step that would trip.
    assert state["counters"]["sim_errors"] == 3 and state["counters"]["episodes"] == 0


def test_student_deals_every_region_keeps_one_buffer_each_and_needs_every_teacher(stub_run, tmp_path, monkeypatch):
    from uipc_manip import train_sac

    built, agents = stub_run
    argv = [
        "student", "--regions", "13", "4", "--teacher-checkpoints", "a.pt", "b.pt", "--garments", *GARMENTS,
        "--train-poses", "0", "1", "--eval-poses", "45", "--num-envs", "4", "--horizon", "2", "--transitions", "16",
        "--eval-every", "8", "--checkpoint-every", "0", "--replay-capacity", "64", "--batch-size", "2", "--device", "cpu",
        "--log-interval", "1", "--point-budget", "3", "--work-dir", str(tmp_path),
    ]
    monkeypatch.setattr(train_sac, "load_teachers", lambda paths, spec, action_dim, device: {4: "teacher_4"})
    with pytest.raises(ValueError, match=r"Regions \[13\] have no teacher"):
        pretrain_wang.main(argv + ["--run-name", "missing"])
    assert built == []  # refused before any world is built
    monkeypatch.setattr(train_sac, "load_teachers", lambda paths, spec, action_dim, device: {4: "teacher_4", 13: "teacher_13"})
    pretrain_wang.main(argv + ["--run-name", "student"])
    agent = agents[-1]
    assert agent.teachers == {4: "teacher_4", 13: "teacher_13"}
    worlds = _training_worlds(built)
    for w in worlds:
        assert sorted((pose_region(b), g) for g, b in w.cells) == [(4, "tshirt_26"), (4, "tshirt_68"), (13, "tshirt_26"), (13, "tshirt_68")]
    held = [w for w in built if w not in worlds]
    assert held[0].cells == [("tshirt_26", 5045), ("tshirt_68", 5045), ("tshirt_26", 14045), ("tshirt_68", 14045)]
    # Buffer 0 is region 4 and buffer 1 region 13: each batch is one region's rows, labelled for that region's teacher.
    assert {i for i, _, _ in agent.batches} == {0, 1}
    assert all(labels == [(4, 13)[index]] for index, _, labels in agent.batches)


def test_a_world_that_fails_to_build_is_drawn_again(stub_run, tmp_path):
    FAIL_BUILDS.append(1)
    pretrain_wang.main(RUN_ARGV + ["--transitions", "8", "--work-dir", str(tmp_path), "--run-name", "retry"])
    state = json.loads((tmp_path / "retry" / "checkpoints" / "state.json").read_text())
    assert not FAIL_BUILDS and state["counters"]["build_failures"] == 1 and state["counters"]["rotations"] == 1
