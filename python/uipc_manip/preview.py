"""Render saved evaluation trajectories to a GIF and a final PNG with matplotlib.

Usage::

    PYTHONPATH=python python -m uipc_manip.preview runs/uipc_manip/<run>/trajectories/episode_000.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def render(path: Path, output: Path | None = None, stride: int = 3) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = np.load(path, allow_pickle=False)
    positions, tcp, goal, marker = data["positions"], data["tcp"], data["goal"], data["marker_centroid"]
    faces, edges = data["faces"], data["edges"]
    kind, task = str(data["deformable"]), str(data["task"])
    frames = list(range(0, len(positions), max(1, stride)))
    if frames[-1] != len(positions) - 1:
        frames.append(len(positions) - 1)
    static_v = data["static_vertices"] if "static_vertices" in data.files else None
    static_f = data["static_faces"] if "static_faces" in data.files else None
    lo = positions.min(axis=(0, 1)) - 0.1
    hi = positions.max(axis=(0, 1)) + 0.1
    if static_v is not None:
        lo = np.minimum(lo, static_v.min(axis=0) - 0.05)
        hi = np.maximum(hi, static_v.max(axis=0) + 0.05)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    def draw(i: int) -> None:
        ax.clear()
        p = positions[i]
        if static_v is not None and static_f is not None:
            ax.add_collection3d(Poly3DCollection(static_v[static_f], facecolor="#e0c3a0", edgecolor="none", alpha=0.95))
        if kind == "cloth" and len(faces):
            ax.add_collection3d(Poly3DCollection(p[faces], facecolor="#4f8fd6", edgecolor="#23507f", linewidth=0.15, alpha=0.9))
        elif len(edges):
            ax.plot(*p.T, color="#e07b2a", linewidth=4)
        ax.scatter(*tcp[i], color="#222222", s=40, label="tool")
        ax.scatter(*goal[i], color="#2fb35a", s=60, label="goal")
        ax.scatter(*marker[i], color="#d63b3b", s=40, label="marker centroid")
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2] if static_v is not None else 0.0, max(0.3, hi[2])), xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
        ax.set_title(f"{task}: decision {i} / {len(positions) - 1}")
        ax.view_init(elev=28, azim=-60)
        ax.legend(loc="upper left")

    output = path.with_suffix("") if output is None else output
    animation = FuncAnimation(fig, draw, frames=frames, interval=60)
    animation.save(output.with_suffix(".gif"), writer=PillowWriter(fps=15))
    draw(len(positions) - 1)
    fig.savefig(output.with_suffix(".png"), dpi=140)
    plt.close(fig)
    return output.with_suffix(".gif")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, nargs="+")
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args(argv)
    for path in args.trajectory:
        print(render(path, stride=args.stride))


if __name__ == "__main__":
    main()
