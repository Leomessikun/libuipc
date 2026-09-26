"""Search wider gripper offsets for (body, garment) units that had no legal gravity-hung start.

CPU only. Reproduces the collector's placement (hang scaled to the body by --fit-sleeve-ratio, gripper at
the FMVP-PyBullet offset from the fingertip plus an extra offset in the model frame, turned about the
vertical, legal when the garment clears the whole body by 3 mm) and reports, per unit, which offsets on a
grid admit a legal turn and how far the opening then sits from the calibrated one.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / ".claude/worktrees/residual-rl/python"))

BASE = np.array([-0.033, 0.106, -0.003])
DESIRED_OPENING = np.array([-0.085, -0.135, -0.050])
HANGS = {"tshirt_26": "output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz"}
for _g in ("tshirt_68", "tshirt_4", "tshirt_392", "hospital_gown"):
    HANGS[_g] = f"output/uipc_manip/garment_demos_20260925/hang_{_g}.npz"
CLOTH3D = Path("/home/ge47gax/kun/fmvp_pb/dressing_pb/assistive_gym/assets/data/cloth3d/train/Tshirt")


def probe(job):
    body, garment, grid, ratio, turn_step, arm_frame = job
    from uipc_manip import dressing_live
    from uipc_manip.dressing_body import BodyConfig, smplx_faces
    from uipc_manip.wang_bridge import up_axis_rotation
    from physical_sleeve import SleeveSections, read_obj

    rotation = up_axis_rotation(267.0)
    cfg = dressing_live.LiveCellConfig()
    cfg = replace(cfg, body=replace(cfg.body, device="cpu", fit_filter_m=0.18))
    cell = dressing_live.LiveCellFactory(cfg).build(garment, body)
    faces = smplx_faces()
    hang = np.load(ROOT / HANGS[garment])["k300"]
    rest, rest_faces = read_obj(CLOTH3D / f"{garment}.obj")
    sections = SleeveSections(rest * 4.0, rest_faces, cell.opening_idx)
    sleeve_r = float(np.mean([np.linalg.norm(q - q.mean(0), axis=1).mean() for q in sections.points(hang)[1:]]))
    elbow, shoulder = np.asarray(cell.elbow, float), np.asarray(cell.shoulder, float)
    u = (shoulder - elbow) / np.linalg.norm(shoulder - elbow)
    rel = np.asarray(cell.arm_points, float) - (elbow + shoulder) / 2
    t = rel @ u
    radial = np.linalg.norm(rel[np.abs(t) < 0.01] - np.outer(t[np.abs(t) < 0.01], u), axis=1)
    scale = max(1.0, ratio * float(np.median(radial[radial < 0.12])) / sleeve_r)
    hang = hang * scale
    finger = np.asarray(cell.finger, float)
    frame = rotation
    if arm_frame:
        forward = np.asarray(cell.elbow, float) - finger
        forward[2] = 0.
        forward /= np.linalg.norm(forward)
        up = rotation[1] / np.linalg.norm(rotation[1])
        side = np.cross(forward, up)
        if np.sign(np.linalg.det(np.stack([forward, up, side]))) != np.sign(np.linalg.det(rotation)):
            side = -side
        frame = np.stack([forward, up, side])
    legal = []
    for offset in grid:
        target = finger + (BASE + np.asarray(offset) / 1000.0) @ frame
        best = None
        for degrees in range(0, 360, turn_step):
            c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
            cloth = hang @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]).T + target
            if dressing_live.garment_arm_gap(cloth, cell.faces, cell.human_points, faces) < 0.003:
                continue
            opening = (cloth[cell.opening_idx].mean(0) - finger) @ frame.T
            err = float(np.linalg.norm(opening - DESIRED_OPENING))
            if best is None or err < best[1]:
                best = (degrees, err)
        if best is not None:
            legal.append(dict(offset_mm=list(offset), turn=best[0], opening_err_cm=round(best[1] * 100, 1)))
    return dict(body=body, garment=garment, scale=round(scale, 2), legal=legal)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--units", type=Path, required=True, help="no_legal_start.jsonl")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--turn-step", type=int, default=15)
    p.add_argument("--arm-frame", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    units = [json.loads(l) for l in a.units.read_text().splitlines() if l.strip()]
    rng = np.random.default_rng(0)
    units = [units[i] for i in rng.choice(len(units), size=min(a.limit, len(units)), replace=False)]
    # model frame: x out along the forearm (negative = away from the hand), y up, z lateral
    grid = [(dx, dy, dz) for dx, dy, dz in itertools.product((0, -40, -80, -120), (5, 40, 80), (-80, -40, 0, 40, 80))]
    jobs = [(u["body"], u["garment"], grid, 1.2, a.turn_step, a.arm_frame) for u in units]
    with ProcessPoolExecutor(a.workers) as pool:
        results = list(pool.map(probe, jobs))
    a.out.write_text(json.dumps(results, indent=1))
    for r in results:
        best = min(r["legal"], key=lambda x: x["opening_err_cm"]) if r["legal"] else None
        print(r["body"], r["garment"], "scale", r["scale"], "legal offsets", len(r["legal"]), "of", len(grid), "best", best, flush=True)


if __name__ == "__main__":
    main()
