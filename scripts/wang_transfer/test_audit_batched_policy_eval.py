"""CPU regressions for incomplete, mismatched and falsely accepted evaluations."""
import json
from pathlib import Path

import numpy as np

from audit_batched_policy_eval import audit, digest, read_rows


def fixture(root, *, flow_displacement=0., flow_hold_action=0., illegal_flow=False):
    run = root / "evaluation"
    run.mkdir()
    index = root / "index.json"
    index.write_text(json.dumps(dict(episodes=[dict(body=1, split="test"), dict(body=2, split="train")])))
    (run / "run.json").write_text(json.dumps(dict(policies={"r1": "r1.pt", "flow": "flow.pt"},
        bodies=[1, 3], garments=["shirt"], split="test", index=str(index))))
    # Body 3 is included in the split but has no result yet.
    index.write_text(json.dumps(dict(episodes=[dict(body=1, split="test"), dict(body=3, split="test"),
                                               dict(body=2, split="train")])))
    rows = []
    for policy in ("r1", "flow"):
        checkpoint = root / f"{policy}.pt"
        checkpoint.write_text(policy)
        if policy == "flow" and illegal_flow:
            rows.append(dict(policy=policy, garment="shirt", body=1, no_legal_start=True))
            continue
        batch = run / policy / "shirt" / "batch"
        body = batch / "body_1_seed_7"
        body.mkdir(parents=True)
        (batch / "run.json").write_text(json.dumps(dict(hold=2, stop_proximal_upper=.7, seed=7)))
        record = dict(body=1, seed=7, accepted=True, transitions=3, checkpoint_sha256=digest(checkpoint), success_state=1)
        positions = np.zeros((4, 5, 3))
        if policy == "flow":
            positions[..., 0] += flow_displacement
        actions = np.zeros((3, 6))
        if policy == "flow":
            actions[-1, 0] = flow_hold_action
        path = Path("body_1_seed_7/episode.npz")
        np.savez_compressed(batch / path, positions=positions, tcp=np.zeros((4, 3)), obs=np.zeros((4, 8)),
            human_vertices=np.zeros((6, 3)), actions=actions, grasp_valid=np.ones(3, bool),
            sleeve_wrapped=np.ones(4, bool), sleeve_proximal_upper_fraction=np.ones(4),
            metadata_json=json.dumps(record))
        rows.append(dict(record, policy=policy, garment="shirt", log=str(batch.with_suffix(".log")), path=str(path)))
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run, rows


def test_planned_denominator_retains_pending_and_checks_real_starts(tmp_path):
    run, _ = fixture(tmp_path, flow_displacement=.004)
    result = audit(run, tmp_path)
    assert result["state"] == "incomplete"
    assert result["expected_results"] == 4 and result["pending_results"] == 2
    assert result["counts"]["flow"]["shirt"]["reported_accepted"] == 1
    assert result["pair_counts"] == {"initial_state_mismatch": 1}
    assert result["pairs"][0]["initial_difference"]["flow"]["positions"] == .004
    assert not result["errors"]


def test_accepted_label_does_not_override_failed_hold(tmp_path):
    run, _ = fixture(tmp_path, flow_hold_action=.1)
    result = audit(run, tmp_path)
    assert result["pair_counts"] == {"unverified_artifact": 1}
    assert any("zero-command hold" in e for e in result["errors"])


def test_missing_start_is_not_counted_as_attempted_or_matched(tmp_path):
    run, _ = fixture(tmp_path, illegal_flow=True)
    result = audit(run, tmp_path)
    assert result["counts"]["flow"]["shirt"]["no_legal_start"] == 1
    assert result["counts"]["flow"]["shirt"]["attempted"] == 0
    assert result["pair_counts"] == {"unmatched_eligibility": 1}


def test_duplicate_results_and_body_leak_are_visible(tmp_path):
    run, rows = fixture(tmp_path)
    with (run / "results.jsonl").open("a") as f:
        f.write(json.dumps(rows[0]) + "\n")
    index = tmp_path / "index.json"
    payload = json.loads(index.read_text())
    payload["episodes"].append(dict(body=1, split="train"))
    index.write_text(json.dumps(payload))
    result = audit(run, tmp_path)
    assert any("Duplicate" in e for e in result["errors"])
    assert any("dataset split" in e for e in result["errors"])
    assert result["pending_results"] == 3


def test_only_final_unterminated_json_can_be_in_progress(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"body": 1}\n{"body":')
    rows, errors, partial = read_rows(path)
    assert rows == [dict(body=1)] and not errors and partial
    path.write_text('{broken}\n{"body": 1}\n')
    rows, errors, partial = read_rows(path)
    assert len(errors) == 1 and not partial


def test_matching_geometry_with_different_seed_is_not_paired(tmp_path):
    run, _ = fixture(tmp_path)
    path = run / "flow/shirt/batch/run.json"
    settings = json.loads(path.read_text())
    settings["seed"] = 8
    path.write_text(json.dumps(settings))
    result = audit(run, tmp_path)
    assert result["pair_counts"] == {"settings_mismatch": 1}


def test_nan_initial_state_is_rejected(tmp_path):
    run, _ = fixture(tmp_path, flow_displacement=float("nan"))
    result = audit(run, tmp_path)
    assert result["pair_counts"] == {"unverified_artifact": 1}
    assert any("Non-finite" in e for e in result["errors"])
