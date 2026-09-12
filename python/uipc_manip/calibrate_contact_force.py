"""Calibrate exported model contact forces against gravity and momentum balance.

Run each configuration in a fresh process, for example::

    python -m uipc_manip.calibrate_contact_force --dt 0.016666666666666666 --gx 2 --tol 1e-7 --out /tmp/contact-tight

Compare ``--gx 0`` (normal support only) and ``--gx 2`` (loaded friction), at
``--dt 0.016666666666666666`` and ``--dt 0.008333333333333333``, with the
backend's default tolerance (omit ``--tol``) and ``--tol 1e-7``. The report
records measured errors rather than declaring all solver settings calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


def _positive(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _finite(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise argparse.ArgumentTypeError("must be finite")
    return result


def _nonnegative_integer(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dt", type=_positive, default=1.0 / 60.0, help="solver integration step in seconds")
    result.add_argument("--gx", type=_finite, default=2.0, help="tangential gravity in m/s²; normal gravity is -9.8")
    result.add_argument("--tol", type=_positive, help="Newton velocity tolerance; omission keeps backend default")
    result.add_argument("--min-iter", type=_nonnegative_integer, help="Newton iteration-index floor; 1 allows exit after 2 assemblies")
    result.add_argument("--semi", type=int, choices=(0, 1), default=0, help="semi-implicit termination enabled")
    result.add_argument("--seconds", type=_positive, default=2.0, help="total simulated duration")
    result.add_argument("--tail-seconds", type=_positive, default=0.5, help="final measurement window")
    result.add_argument("--out", type=Path, required=True, help="new output directory for result.json and world files")
    return result


def _provenance(uipc) -> dict:
    native = Path(uipc.__file__).resolve().parent / "_native"
    binaries = sorted(native.glob("pyuipc*.so")) + sorted(native.glob("libuipc_backend_cuda.so"))
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "uipc_version": str(uipc.__version__),
        "uipc_module": str(Path(uipc.__file__).resolve()),
        "binaries": [
            {"path": str(path), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
            for path in binaries
        ],
        "gradient_state": "last assembled Newton iterate, before its accepted position correction",
    }


def calibrate(args: argparse.Namespace) -> dict:
    import uipc
    from uipc import Engine, Logger, Scene, World, view
    from uipc.constitution import Particle
    from uipc.geometry import ground, label_surface, pointcloud

    from .contact_force import find_contact_feature, geometry_vertex_block, vertex_forces

    steps = int(round(args.seconds / args.dt))
    tail_steps = int(round(args.tail_seconds / args.dt))
    if steps < 3 or tail_steps < 1 or tail_steps > steps - 2:
        raise ValueError("duration must provide at least three frames and a tail after the first two frames")
    # This test assumes friction can support the tangential load below the Coulomb limit.
    mu, normal_gravity, density, radius = 0.5, -9.8, 1000.0, 0.01
    if abs(args.gx) >= mu * abs(normal_gravity):
        raise ValueError("abs(gx) must be below 4.9 m/s² for this sticking-force calibration")
    args.out.mkdir(parents=True, exist_ok=False)
    Logger.set_level(Logger.Level.Error)
    engine = Engine("cuda", str(args.out / "world"))
    world = World(engine)
    config = Scene.default_config()
    config["dt"] = args.dt
    gravity = np.array([args.gx, normal_gravity, 0.0])
    config["gravity"] = gravity.reshape(3, 1).tolist()
    config["newton"]["semi_implicit"]["enable"] = args.semi
    if args.tol is not None:
        config["newton"]["velocity_tol"] = args.tol
    if args.min_iter is not None:
        config["newton"]["min_iter"] = args.min_iter
    scene = Scene(config)
    scene.contact_tabular().default_model(mu, 1e9)
    mesh = pointcloud(np.array([[0.0, 0.03, 0.0]]))
    label_surface(mesh)
    Particle().apply_to(mesh, density, radius)
    scene.contact_tabular().default_element().apply_to(mesh)
    obj = scene.objects().create("calibration_particle")
    slot = obj.geometries().create(mesh)[0]
    obj.geometries().create(ground(0.0))
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("calibration world failed initialization")
    feature = find_contact_feature(world)
    if feature is None or not {"PH+N", "PH+F"}.issubset(feature.contact_primitive_types()):
        raise RuntimeError("calibration requires IPC half-plane normal and friction exporters")
    first, count = geometry_vertex_block(slot.geometry())
    mass = density * 4.0 / 3.0 * math.pi * radius**3
    positions, rows = [], []
    for frame in range(1, steps + 1):
        world.advance()
        world.retrieve()
        if not world.is_valid():
            raise RuntimeError(f"invalid calibration world at frame {frame}")
        pos = np.asarray(view(slot.geometry().positions())).reshape(-1, 3).copy()[0]
        normal, friction = vertex_forces(feature, args.dt, count, first_vertex=first)
        if not np.isfinite(np.concatenate([pos, normal.reshape(-1), friction.reshape(-1)])).all():
            raise RuntimeError(f"non-finite state or force at frame {frame}")
        stats = engine.frame_stats() if hasattr(engine, "frame_stats") else {}
        if stats.get("completed") is False:
            raise RuntimeError(f"incomplete calibration frame {frame}: {stats}")
        positions.append(pos)
        velocity = (positions[-1] - positions[-2]) / args.dt if len(positions) >= 2 else None
        residual = None
        if len(positions) >= 3:
            acceleration = (positions[-1] - 2 * positions[-2] + positions[-3]) / args.dt**2
            residual = mass * (acceleration - gravity) - normal[0] - friction[0]
        rows.append({
            "frame": frame,
            "position": pos.tolist(),
            "velocity": velocity.tolist() if velocity is not None else None,
            "normal": normal[0].tolist(),
            "friction": friction[0].tolist(),
            "force_balance_residual": residual.tolist() if residual is not None else None,
            "newton_iterations": stats.get("newton_iterations"),
            "stats": stats,
        })
    tail = rows[-tail_steps:]
    summary = {
        "dt": args.dt,
        "gx": args.gx,
        "velocity_tol": config["newton"]["velocity_tol"],
        "min_iter": config["newton"]["min_iter"],
        "semi": args.semi,
        "simulated_seconds": steps * args.dt,
        "tail_frames": tail_steps,
        "mass_kg": mass,
        "expected_normal_n": mass * abs(normal_gravity),
        "expected_friction_n": -mass * args.gx,
        "normal_mean_n": np.mean([row["normal"] for row in tail], axis=0).tolist(),
        "friction_mean_n": np.mean([row["friction"] for row in tail], axis=0).tolist(),
        "friction_zero_frames": int(sum(np.linalg.norm(row["friction"]) == 0 for row in tail)),
        "balance_residual_max_n": float(max(np.linalg.norm(row["force_balance_residual"]) for row in tail)),
        "newton_iterations_tail": [row["newton_iterations"] for row in tail],
        "final_velocity_m_s": rows[-1]["velocity"],
    }
    report = {
        "config": config,
        "particle": {"density_kg_m3": density, "radius_m": radius, "friction_coefficient": mu},
        "global_vertex_block": [first, count],
        "contact_primitive_types": list(feature.contact_primitive_types()),
        "provenance": _provenance(uipc),
        "summary": summary,
        "rows": rows,
    }
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parser().parse_args()
    report = calibrate(args)
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
