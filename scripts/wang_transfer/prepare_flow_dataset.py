"""Freeze and merge FMVP recordings after a common terminal sleeve audit.

Recompute three interior sleeve sections and armhole progress through the
recorded zero-command hold. Preserve collection flags and prior body splits.
This does not independently validate cloth collisions, forces or grasp physics.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from physical_sleeve import DEFAULT_OBJ, SleeveSections, measure, read_obj
from uipc_manip.rollout_chunks import prepare


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TerminalSleeveAudit:
    scope = ("Recorded acceptance, array/grasp/command contract checks; independently recomputed "
             "interior-section wrapping and armhole >= 0.7 through 20 stationary terminal decisions. "
             "Not a collision, force or grasp-physics revalidation. Source placement protocols retained.")

    def __init__(self, template_dir=DEFAULT_OBJ.parent):
        self.template_dir = Path(template_dir)

    @lru_cache(maxsize=32)
    def template(self, garment, opening):
        path = self.template_dir / f"{garment}.obj"
        vertices, faces = read_obj(path)
        return SleeveSections(vertices * 4., faces, np.asarray(opening)), sha256(path)

    def __call__(self, record, actions):
        path = Path(record["source_path"])
        run_path, config_path = path.parent.parent / "run.json", path.parent / "config.json"
        run = json.loads(run_path.read_text())
        config = json.loads(config_path.read_text())
        if (run.get("success_geometry") != "physical_sleeve" or not run.get("armhole_endpoint")
                or run.get("stop_proximal_upper") != .7 or run.get("hold") != 20
                or run.get("collision_geometry") != "full_body" or run.get("autonomous_hold")
                or config["config"]["obs"]["mode"] != "wang_static_arm"):
            raise ValueError("Common audit requires static full-body armhole-0.7 episodes with 20 external hold commands")
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            hold_start = int(metadata["success_state"])
            if len(actions) != hold_start + 20 or np.max(np.abs(actions[hold_start:])) > 1e-8:
                raise ValueError("Recorded terminal hold is incomplete or contains nonzero commands")
            sections, template_hash = self.template(record["garment"], tuple(data["opening_idx"].tolist()))
            faces = data["faces"]
            if not np.array_equal(faces, sections.faces):
                actual = set(map(tuple, np.sort(faces, axis=1)))
                expected = set(map(tuple, np.sort(sections.faces, axis=1)))
                if actual != expected:
                    raise ValueError("Recording topology differs from sleeve template")
            cloth = data["positions"]
            landmarks = np.stack([data[k] for k in ("finger", "elbow", "shoulder")])
            if (cloth.shape != (len(actions) + 1, len(sections.vertices), 3)
                    or not np.isfinite(cloth).all() or not np.isfinite(landmarks).all()
                    or np.any(np.linalg.norm(np.diff(landmarks, axis=0), axis=1) < 1e-6)):
                raise ValueError("Invalid cloth states or arm landmarks")
            measurements = [measure(sections, state, landmarks) for state in cloth[hold_start:]]
        wrapped = [all(r["wrapped"] for r in m["rings"][1:]) for m in measurements]
        progress = np.asarray([m["armhole_upper_fraction"] for m in measurements])
        if not all(wrapped) or not np.isfinite(progress).all() or progress.min() < .7:
            raise ValueError("Terminal mesh fails common interior-wrap/armhole-0.7 geometry")
        protocol = {key: run.get(key, False) for key in
                    ("align_armhole_axis", "sections_wrap", "armhole_endpoint", "arm_frame_placement")}
        protocol.update(hold=20, threshold=.7, placement=config["placement"],
                        environment_sha256=run["environment_sha256"], collector_sha256=run["collector_sha256"])
        return dict(common_rule="interior_sections_and_armhole_0.7_hold20_v1", hold_start=hold_start,
                    held_states=len(measurements), min_armhole_fraction=float(progress.min()),
                    cuff_also_wrapped=all(m["sleeve_wrapped"] for m in measurements),
                    template_sha256=template_hash, run_sha256=sha256(run_path), config_sha256=sha256(config_path),
                    auditor_sha256=sha256(__file__), geometry_sha256=sha256(Path(__file__).with_name("physical_sleeve.py")),
                    collection_protocol=protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    prepare(args.source, args.out, val_fraction=args.val_fraction, seed=args.seed,
            split_manifest=args.split_manifest, episode_audit=TerminalSleeveAudit())


if __name__ == "__main__":
    main()
