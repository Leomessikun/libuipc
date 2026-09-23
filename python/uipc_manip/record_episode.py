"""Record one dressing episode through Genesis's own renderer.

``uipc_manip.preview`` draws the saved trajectory with matplotlib, which shows the cloth as a
wireframe and nothing of the scene. This plays the episode again inside the simulator and renders
it with a Genesis camera, so the video is the shaded cloth and arm the viewer shows.

The cloth and arm are debug meshes (``DressingEnv._draw``), not scene entities, because the arm is
a native libuipc affine body rather than a Genesis mesh. Only a camera built with ``debug=True``
rasterises markers (``genesis/vis/rasterizer.py``), so that flag is not optional here.

Usage::

    PYTHONPATH=python python -m uipc_manip.record_episode \
        --checkpoint out/keep/best.pt --garment tshirt_26 --body 14045 --out demo/clip.mp4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="A pretrain_wang teacher checkpoint; its actor acts deterministically.")
    p.add_argument("--garment", required=True)
    p.add_argument("--body", type=int, required=True)
    p.add_argument("--out", required=True, help="Output .mp4.")
    p.add_argument("--region", type=int, default=13)
    p.add_argument("--seed", type=int, default=1, help="Run seed; the reset seed follows the evaluation's own rule.")
    p.add_argument("--eval-round", type=int, default=0, help="Which evaluation seed block to replay.")
    p.add_argument("--slot", type=int, default=0, help="The cell's slot in that round, which sets its reset seed.")
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--res", type=int, nargs=2, default=[1280, 960])
    p.add_argument("--fov", type=float, default=45.0)
    p.add_argument("--elevation", type=float, default=20.0, help="Camera elevation above the arm's plane, in degrees.")
    p.add_argument("--azimuth", type=float, default=0.0,
                   help="Rotation about the arm's own axis, in degrees; 0 looks at the arm broadside.")
    p.add_argument("--no-body", action="store_true",
                   help="Draw only the simulated right arm. By default the rest of the SMPL-X body is drawn "
                        "as well, greyed, to show where the person is; it is NOT in the simulation.")
    p.add_argument("--cam-margin", type=float, default=0.85,
                   help="Camera distance as a multiple of what just contains the reset bounding sphere; the sphere is the box diagonal, so under 1.0 still fits a scene that is wider than it is tall.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--critic-action-mode", default="dense")
    p.add_argument("--trunk-style", default="residual")
    p.add_argument("--obs-mode", default="wang_static_arm")
    return p


def _body_mesh(cell) -> tuple[np.ndarray, np.ndarray] | None:
    """The rest of the person, for arm-only recordings' picture.

    An arm-only world adds ``cell.arm_points`` as its fixed affine body. The cell
    still carries the full SMPL-X vertices, so those can be drawn for context.
    A full-body collision world already draws its complete simulated collider.
    """
    points = getattr(cell, "human_points", None)
    if points is None:
        return None
    try:
        from .dressing_body import smplx_faces
    except ImportError:
        return None
    try:
        return np.asarray(points, float), smplx_faces()
    except Exception as exc:
        print(f"[record] the rest of the body is not drawn: {exc}", flush=True)
        return None


def _draw_body(env, body) -> None:
    """Draw the un-simulated body once per frame, behind the arm the environment draws."""
    import trimesh

    vertices, faces = body
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    mesh.visual.vertex_colors = np.tile([150, 150, 158, 150], (len(vertices), 1))
    env._debug_objects.append(env.scene.draw_debug_mesh(mesh))


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import genesis as gs

    from . import pretrain_wang, train_sac
    from .dressing_env import GenesisIPCDressingEnv
    from .obs import ObsSpec
    from .sac import SACAgent, SACConfig

    _, targs, _ = pretrain_wang.prepare([
        "teacher", "--region", str(a.region), "--seed", str(a.seed), "--obs-mode", a.obs_mode, "--no-obs-augment",
        "--critic-action-mode", a.critic_action_mode, "--trunk-style", a.trunk_style,
    ])
    cfg = replace(train_sac.dressing_config(targs), cells=((a.garment, int(a.body)),), cell_source="live",
                  horizon=a.horizon, seed=a.seed, show_viewer=False)

    # The camera has to exist before the scene is built, and the environment builds in __init__.
    camera = {}
    original = gs.Scene.build

    def build_with_camera(scene, *args, **kw):
        camera["cam"] = scene.add_camera(res=tuple(a.res), GUI=False, fov=a.fov, debug=True)
        return original(scene, *args, **kw)

    gs.Scene.build = build_with_camera
    try:
        env = GenesisIPCDressingEnv(cfg, num_envs=1)
    finally:
        gs.Scene.build = original
    cam = camera["cam"]

    payload = SACAgent.read_checkpoint(a.checkpoint)
    agent = SACAgent(ObsSpec(targs.point_budget), env.action_dim, SACConfig.from_dict(payload["sac_config"]), targs.device)
    agent.load(a.checkpoint, load_optimizers=False)
    agent.train(False)

    # train_sac.evaluate's rule, so the episode is the one that round scored.
    seed = a.seed * 1000 + 97 * int(a.eval_round) + int(a.slot)
    obs = env.reset([seed])
    body = None if a.no_body or env.cfg.collision_geometry == "full_body" else _body_mesh(env.cells[0])

    # Look at the arm broadside. A direction fixed in world axes shows a differently posed arm
    # end-on, and the sleeve's progress up the forearm is exactly what an end-on view hides.
    cell = env.cells[0]
    finger, shoulder = np.asarray(cell.finger, float), np.asarray(cell.shoulder, float)
    axis = shoulder - finger
    axis /= max(np.linalg.norm(axis), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    side = np.cross(axis, up)
    side /= max(np.linalg.norm(side), 1e-9)
    theta, phi = np.radians(a.azimuth), np.radians(a.elevation)
    direction = np.cos(phi) * (np.cos(theta) * side + np.sin(theta) * axis) + np.sin(phi) * up

    shown = [env.positions()[0], env.arm_meshes[0][0]] + ([body[0]] if body is not None else [])
    points = np.concatenate(shown, axis=0)
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = 0.5 * (lo + hi)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    distance = a.cam_margin * radius / float(np.tan(np.radians(a.fov) / 2.0))
    # World z has to be given explicitly: without it the camera keeps whatever roll add_camera
    # left it with, and a seated body comes out lying on its side.
    cam.set_pose(pos=tuple(centre + direction * distance), lookat=tuple(centre), up=(0.0, 0.0, 1.0))
    print(f"[record] {a.garment} on body {a.body}, reset seed {seed}, {a.horizon} decisions; "
          f"scene radius {radius:.3f} m, camera {distance:.2f} m out, fov {a.fov:.0f}, "
          f"elevation {a.elevation:.0f} deg, {'body drawn' if body is not None else 'arm only'}", flush=True)

    def frame() -> None:
        env._draw()                       # clears the previous debug objects, then cloth and arm
        if body is not None:
            _draw_body(env, body)
        cam.render()

    cam.start_recording()
    frame()
    t0, ratio = time.time(), 0.0
    for k in range(a.horizon):
        obs, _, done, infos = env.step(agent.act(obs, deterministic=True).astype(np.float32))
        ratio = float(infos[0].get("upperarm_ratio", 0.0))
        frame()
        if k % 50 == 0:
            print(f"[record] decision {k:3d}/{a.horizon} upper-arm {ratio:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if done.any():
            break
    cam.stop_recording(save_to_filename=str(out), fps=a.fps)
    env.close()
    print(f"[record] {out} — final upper-arm ratio {ratio:.3f}, {'success' if ratio >= 0.7 else 'not a success'} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
