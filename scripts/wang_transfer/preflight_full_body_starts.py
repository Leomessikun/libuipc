"""Screen gravity-hung FMVP starts against the complete SMPL-X body, without IPC.

Use the same mesh, yaw, 5-degree cloth rotations, and exact surface-gap test as
collect_better_rollouts.py. This only checks the initial geometry; it does not
predict whether the checkpoint will dress the arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE_PLACEMENT = np.array([-0.033, 0.106, -0.003])
DESIRED_OPENING = np.array([-0.085, -0.135, -0.050])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package-root", type=Path, default=ROOT / ".claude/worktrees/residual-rl/python")
    p.add_argument("--hang", type=Path, required=True)
    p.add_argument("--hang-key", default="k300")
    p.add_argument("--reference-episode", type=Path, default=None,
                   help="Optionally verify topology against a recorded tshirt_26 episode.")
    p.add_argument("--bodies", type=int, nargs="+", required=True)
    p.add_argument("--placement-offset-mm", type=float, nargs=3, default=[0., 5., 0.])
    p.add_argument("--yaw", type=float, default=267.)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error(f"output already exists: {args.out}")

    sys.path.insert(0, str(args.package_root.resolve()))
    from uipc_manip.dressing_body import BodyConfig, generate_body
    from uipc_manip.dressing_live import garment_arm_gap
    from uipc_manip.wang_bridge import up_axis_rotation

    with np.load(args.hang, allow_pickle=False) as source:
        hang = np.asarray(source[args.hang_key], dtype=np.float64)
        hang_faces = np.asarray(source["faces"], dtype=np.int32)
        opening_idx = np.asarray(source["opening_idx"], dtype=np.int32)
    if args.reference_episode is not None:
        with np.load(args.reference_episode, allow_pickle=False) as reference:
            if not np.array_equal(hang_faces, reference["faces"]) or not np.array_equal(opening_idx, reference["opening_idx"]):
                raise ValueError("Hang mesh topology/opening does not match reference episode")

    rotation = up_axis_rotation(args.yaw)
    turns = []
    for degrees in range(0, 360, 5):
        c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
        turn = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        turns.append((degrees, hang @ turn.T))

    rows = []
    for body_id in args.bodies:
        body = generate_body(body_id, BodyConfig(device="cpu"))
        finger = np.asarray(body.finger, dtype=np.float64)
        target = finger + (BASE_PLACEMENT + np.asarray(args.placement_offset_mm) / 1000.) @ rotation
        best = None
        legal_turns = 0
        for degrees, rotated in turns:
            cloth = rotated + target
            gap = garment_arm_gap(cloth, hang_faces, body.vertices, body.faces)
            if gap < 0.003:
                continue
            legal_turns += 1
            opening = (cloth[opening_idx].mean(axis=0) - finger) @ rotation.T
            error = float(np.linalg.norm(opening - DESIRED_OPENING))
            if best is None or error < best["opening_error_m"]:
                best = {"turn_degrees": degrees, "gap_m": gap,
                        "opening_model": opening.tolist(), "opening_error_m": error}
        row = {"body": body_id, "legal": best is not None, "legal_turns": legal_turns,
               "height_m": body.height_m, **(best or {})}
        rows.append(row)
        print(f"[preflight] {json.dumps(row)}", flush=True)

    manifest = {
        "collision_geometry": "full_body", "garment": "tshirt_26",
        "bodies": args.bodies, "legal_bodies": [r["body"] for r in rows if r["legal"]],
        "hang": str(args.hang.resolve()), "hang_sha256": hashlib.sha256(args.hang.read_bytes()).hexdigest(),
        "hang_key": args.hang_key,
        "reference_episode": str(args.reference_episode.resolve()) if args.reference_episode is not None else None,
        "placement_offset_mm": args.placement_offset_mm, "yaw": args.yaw,
        "min_gap_m": 0.003, "turn_increment_degrees": 5,
        "note": "Geometry-only screen; full IPC rollout still required.", "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[complete] {len(manifest['legal_bodies'])}/{len(rows)} legal starts -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
