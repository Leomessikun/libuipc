"""CPU checks for route geometry and experiment validity; no physical success claim."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from uipc_manip.dressing_heuristic import HeuristicDressingPolicy
from uipc_manip.dressing_route_pilot import build_plan, execute, summarize_trial, write_plan
from uipc_manip.expert_baseline import build_parser, selected_poses
from uipc_manip.pretrain_wang import prepare, region_configs


def make_policy(offset, rotation=None, straight=False):
    rotation = np.eye(3) if rotation is None else rotation
    cell = SimpleNamespace(
        finger=rotation @ np.array([0.0, 0.0, 0.0]),
        elbow=rotation @ np.array([0.3, 0.0, 0.0]),
        shoulder=rotation @ np.array([0.6, 0.0, 0.0] if straight else [0.3, 0.3, 0.0]),
    )
    return HeuristicDressingPolicy(SimpleNamespace(num_envs=1, cells=[cell]), outward_offset=offset)


def test_route_moves_away_from_inside_and_preserves_later_stages():
    baseline, route = make_policy(0), make_policy(0.04)
    for key in ("approach", "finger", "middle"):
        np.testing.assert_allclose(route._targets[0][key] - baseline._targets[0][key], [0, -0.04, 0])
    for key in ("elbow_hook", "last", "forearm_dir", "upperarm_dir"):
        np.testing.assert_array_equal(route._targets[0][key], baseline._targets[0][key])
    route.reset()
    np.testing.assert_allclose(route._targets[0]["middle"], [0.38, -0.04, 0])


def test_route_displacement_rotates_with_arm():
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    baseline, route = make_policy(0, rotation), make_policy(0.04, rotation)
    np.testing.assert_allclose(route._targets[0]["middle"] - baseline._targets[0]["middle"],
                               rotation @ np.array([0, -0.04, 0]))


def test_straight_arm_does_not_invent_an_outside():
    baseline, route = make_policy(0, straight=True), make_policy(0.04, straight=True)
    for key in baseline._targets[0]:
        np.testing.assert_array_equal(baseline._targets[0][key], route._targets[0][key])


@pytest.mark.parametrize("offset", [-0.1, float("nan"), float("inf")])
def test_invalid_offset_rejected(offset):
    with pytest.raises(ValueError):
        make_policy(offset)


def test_training_pose_boundary_and_explicit_selection():
    assert selected_poses("train", [2, 0, 2]) == [2, 0]
    with pytest.raises(ValueError):
        selected_poses("train", [45])
    with pytest.raises(ValueError):
        selected_poses("heldout", [0])


def test_plan_is_matched_and_does_not_launch_anything(tmp_path):
    out = tmp_path / "pilot"
    plan = build_plan(out)
    assert plan["expected_cells"] == [list(c) for c in region_configs([13], ["tshirt_26"], [0, 1, 2])]
    assert plan["max_decisions_per_arm"] == 2700
    assert len(plan["trials"]) == 9
    for b, c in zip(plan["trials"][3:6], plan["trials"][6:]):
        assert c["params"] == {**b["params"], "outward_offset": 0.04}
    write_plan(plan, out)
    assert len(list((out / "parameters").glob("*.json"))) == 9
    assert not (out / "results.json").exists()
    with pytest.raises(FileExistsError):
        write_plan(plan, out)


def test_planned_command_matches_existing_trainer_protocol(tmp_path):
    command = build_plan(tmp_path)["trials"][0]["command"]
    args, extra = build_parser().parse_known_args(command[3:])
    assert selected_poses(args.poses, args.pose_ids) == [0, 1, 2]
    _, trainer, _ = prepare(["teacher", "--region", str(args.region),
                             "--garments", *args.garments, "--num-envs", str(args.num_envs), *extra])
    assert trainer.horizon == 300
    assert trainer.obs_mode == "wang_static_arm"


@pytest.mark.parametrize("kwargs", [{"poses": [45]}, {"poses": []}, {"poses": [1, 1]},
                                   {"region": 27}, {"horizon": 0}, {"timeout_s": float("nan")}])
def test_invalid_search_plan_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_plan(tmp_path, **kwargs)


def test_dropped_cells_and_sim_errors_invalidate_results(tmp_path):
    record = {"garment": "tshirt_26", "human": 14000, "final_upperarm_ratio": 0.9,
              "success": True, "sim_error": False, "paper_filter": False, "length": 300}
    manifest = {"dropped": [], "env": {"dt": 0.0166667}}
    (tmp_path / "records.json").write_text(json.dumps([record]))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert summarize_trial(tmp_path, [["tshirt_26", 14000]])["valid"]
    assert not summarize_trial(tmp_path, [["tshirt_26", 14000], ["tshirt_26", 14001]])["valid"]
    record["sim_error"] = True
    (tmp_path / "records.json").write_text(json.dumps([record]))
    assert not summarize_trial(tmp_path, [["tshirt_26", 14000]])["valid"]


@pytest.mark.parametrize("timeout", [False, True])
def test_runner_records_failure_and_stops_instead_of_launching_remaining_trials(tmp_path, monkeypatch, timeout):
    calls = []

    def fail(command, log, timeout_s):
        calls.append(command)
        if timeout:
            raise subprocess.TimeoutExpired(command, timeout_s)
        return 7

    monkeypatch.setattr("uipc_manip.dressing_route_pilot.run_trial", fail)
    results = execute(build_plan(tmp_path), tmp_path)
    assert len(calls) == 1
    assert results[0]["status"] == ("timeout" if timeout else "process_failed")
    assert json.loads((tmp_path / "results.json").read_text()) == results
