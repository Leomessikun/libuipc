"""Prevent mismatched holdouts and silently changed reproduction protocols."""
import json

import pytest
import torch

from eval_static_flow import digest, make_cases


def inputs(root, *, aligned=False):
    package = root / "package"
    (package / "uipc_manip").mkdir(parents=True)
    environment = package / "uipc_manip/dressing_env.py"
    environment.write_text("# fixture\n")
    source = root / "job/episode/rollout.npz"
    source.parent.mkdir(parents=True)
    (source.parent / "config.json").write_text("{}")
    hang, actor = root / "hang.npz", root / "actor.pt"
    hang.write_bytes(b"hang")
    actor.write_bytes(b"actor")
    run = dict(package_root=str(package), environment_sha256=digest(environment),
               hang=str(hang), checkpoint=str(actor), hang_sha256=digest(hang), hang_key="k300",
               placement_offset_mm=[0, 5, 0], armhole_endpoint=True, align_armhole_axis=aligned,
               hold=20, success_geometry="physical_sleeve", stop_proximal_upper=.7)
    (source.parent.parent / "run.json").write_text(json.dumps(run))
    row = dict(body=7, seed=10, garment="test", split="validation", source_path=str(source),
               checkpoint_sha256=digest(actor), training_arrays_sha256="fixture",
               audit=dict(common_rule="interior_sections_and_armhole_0.7_hold20_v1",
                          run_sha256=digest(source.parent.parent / "run.json"),
                          config_sha256=digest(source.parent / "config.json")))
    data = root / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps(dict(episodes=[row, dict(body=8, split="train")])))
    checkpoint = root / "flow.pt"
    torch.save(dict(format="dressing_flow_bc_chunks_v1",
                    metadata=dict(data_manifest_sha256=digest(data / "manifest.json"))), checkpoint)
    return data, checkpoint


def test_common_rule_is_shared_and_checkpoint_holdout_is_verified(tmp_path):
    data, checkpoint = inputs(tmp_path)
    cases = make_cases(data, checkpoint, tmp_path / "out", tmp_path, ["test"], 750, "random")
    assert len(cases) == 1 and "--sections-wrap" in cases[0]["command"]
    assert cases[0]["command"][cases[0]["command"].index("--variants") + 1:][:2] == ["baseline", "flow"]
    with (data / "manifest.json").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Checkpoint was not trained"):
        make_cases(data, checkpoint, tmp_path / "out", tmp_path, ["test"], 750, "random")


def test_unimplemented_placement_cannot_silently_use_old_start(tmp_path):
    data, checkpoint = inputs(tmp_path, aligned=True)
    with pytest.raises(ValueError, match="select a v4 start"):
        make_cases(data, checkpoint, tmp_path / "out", tmp_path, ["test"], 750, "random")
