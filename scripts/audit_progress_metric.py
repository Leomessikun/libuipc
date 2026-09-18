"""How often does the dressing progress metric read zero while the sleeve is still on the arm?

The project's progress metric casts one ray along the forearm and one along the upper
arm and asks whether they hit the garment opening, which is stored as a six-vertex
polygon triangulated into four triangles. This recomputes it from saved geometry and
compares it with two tests that do not depend on hitting those four triangles:

* the winding test already in the codebase, which asks whether the finger-to-shoulder
  axis passes through the opening ring;
* the distance from the ring's centroid to the arm's centreline.

A decision where the rays read zero while the ring still encircles the arm is a
measurement artifact, not a lost sleeve.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from uipc_manip.dressing_reward import WangRewardConfig, opening_threaded, wang_progress  # noqa: E402


def centreline_distance(point: np.ndarray, finger: np.ndarray, elbow: np.ndarray, shoulder: np.ndarray) -> float:
    best = np.inf
    for a, b in ((finger, elbow), (elbow, shoulder)):
        d = np.asarray(b) - np.asarray(a)
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            continue
        lam = float(np.clip((point - a) @ (d / n), 0.0, n))
        best = min(best, float(np.linalg.norm(point - (a + lam * d / n))))
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path, help="A directory with positions_*.npz and static_*.npz")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = WangRewardConfig()
    rows, summary = [], []
    for positions_path in sorted(args.capture.glob("positions_*.npz")):
        index = positions_path.stem.split("_")[-1]
        static = np.load(args.capture / f"static_{index}.npz")
        positions = np.load(positions_path)["positions"]
        finger, elbow, shoulder = static["finger"], static["elbow"], static["shoulder"]
        opening_idx, triangles = static["opening_idx"], static["opening_triangles"]
        arm = static["arm_vertices"]
        per_decision = []
        for t, cloth in enumerate(positions):
            progress = wang_progress(cloth, polygon_idx=opening_idx, triangle_idx=triangles,
                                     cuff_idx=opening_idx, finger=finger, elbow=elbow, shoulder=shoulder,
                                     human_points=arm, cfg=cfg)
            threaded, along = opening_threaded(cloth, opening_idx, finger, shoulder)
            centre = cloth[opening_idx].mean(axis=0)
            per_decision.append(dict(decision=t, upperarm_ratio=progress.upperarm_ratio,
                                     forearm_ratio=progress.forearm_ratio, on_forearm=progress.on_forearm,
                                     on_upperarm=progress.on_upperarm, threaded=bool(threaded),
                                     along_axis=float(along),
                                     centre_distance=centreline_distance(centre, finger, elbow, shoulder),
                                     ring_radius=float(np.linalg.norm(cloth[opening_idx] - centre, axis=1).mean())))
        zero = np.asarray([not (r["on_forearm"] or r["on_upperarm"]) for r in per_decision])
        threaded = np.asarray([r["threaded"] for r in per_decision])
        distance = np.asarray([r["centre_distance"] for r in per_decision])
        radius = np.asarray([r["ring_radius"] for r in per_decision])
        # The ring is around the arm when the axis passes through it, or when the centre sits
        # within a ring radius of the centreline.
        encircling = threaded | (distance < radius)
        artifact = zero & encircling
        flicker = zero[1:-1] & ~zero[:-2] & ~zero[2:]
        summary.append(dict(capture=positions_path.name, decisions=len(per_decision),
                            zero_decisions=int(zero.sum()), zero_while_encircling=int(artifact.sum()),
                            isolated_zero_decisions=int(flicker.sum()),
                            first_zero=int(np.argmax(zero)) if zero.any() else -1,
                            final_upperarm=per_decision[-1]["upperarm_ratio"],
                            final_threaded=per_decision[-1]["threaded"],
                            final_centre_distance=per_decision[-1]["centre_distance"],
                            final_ring_radius=per_decision[-1]["ring_radius"]))
        rows.append(per_decision)
        s = summary[-1]
        print(f"{s['capture']}: {s['decisions']} decisions, rays read zero at {s['zero_decisions']}, "
              f"of which {s['zero_while_encircling']} still encircle the arm; "
              f"{s['isolated_zero_decisions']} isolated one-decision zeros; "
              f"final upper-arm {s['final_upperarm']:.3f}, threaded {s['final_threaded']}, "
              f"centre {100 * s['final_centre_distance']:.1f} cm from the arm against a ring radius of "
              f"{100 * s['final_ring_radius']:.1f} cm")
    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, decisions=rows), indent=1, default=float) + "\n")
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
