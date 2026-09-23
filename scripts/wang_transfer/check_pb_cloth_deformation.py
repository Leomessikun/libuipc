"""Report saved FMVP PyBullet shirt edge lengths against the source OBJ mesh.

The mesh parser preserves OBJ vertex indices; no plot or simulator replay is
needed. This is a geometry diagnostic, not a calibrated textile strain test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(v.split("/")[0]) - 1 for v in line.split()[1:]]
            if len(face) < 3:
                raise ValueError(f"Invalid OBJ face: {path}")
            faces.extend([[face[0], face[i], face[i + 1]]
                          for i in range(1, len(face) - 1)])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def distribution(ratio: np.ndarray) -> dict:
    return {"median": float(np.median(ratio)),
            "p90": float(np.percentile(ratio, 90)),
            "p99": float(np.percentile(ratio, 99)),
            "max": float(ratio.max()),
            "fraction_gt_2": float(np.mean(ratio > 2.))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--obj", type=Path, required=True)
    p.add_argument("--scale", type=float, default=3.2,
                   help="FMVP tshirt_26 uses cloth_scales=4 times 0.8 in dressing.py")
    p.add_argument("--states", type=int, nargs="*", default=[0, -1])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rest, faces = read_obj(args.obj)
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]])), axis=1), axis=0)

    def lengths(vertices):
        return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)

    rest_lengths = lengths(rest * args.scale)
    if np.min(rest_lengths) <= 1e-9:
        raise ValueError("Degenerate source OBJ edge")
    rows = []
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as data:
            cloth = data["cloth"]
            if cloth.shape[1] != len(rest):
                raise ValueError(f"OBJ and saved cloth vertex counts differ: {path}")
            initial_lengths = lengths(cloth[0])
            states = {}
            for requested in args.states:
                k = requested if requested >= 0 else len(cloth) + requested
                if k < 0 or k >= len(cloth):
                    raise IndexError(f"State {requested} out of range for {path}")
                current = lengths(cloth[k])
                states[str(k)] = {"against_obj_rest": distribution(current / rest_lengths),
                                  "against_first_recorded": distribution(current / initial_lengths)}
            rows.append({"path": str(path.resolve()), "states": len(cloth),
                         "cloth_vertices": len(rest), "mesh_edges": len(edges),
                         "sampled_states": states})
    report = {"obj": str(args.obj.resolve()), "scale": args.scale,
              "note": "Edge ratios are geometric diagnostics; large values reject cloth realism, small values alone do not prove realism.",
              "episodes": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(f"{Path(row['path']).name}: " + ", ".join(
            f"state {k} rest p99={v['against_obj_rest']['p99']:.2f}x"
            for k, v in row["sampled_states"].items()))
    print(f"[complete] {args.out}")


if __name__ == "__main__":
    main()
