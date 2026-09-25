"""Render recorded IPC rollouts to mp4 offscreen (pyrender over EGL), without resimulating.

Each input is a collector ``.npz`` (cloth positions per state, the person's mesh, the gripper path).
The person, the garment and the gripper are drawn as in ``view_rollouts_genesis.py`` (Genesis's
debug meshes do not reach its offscreen cameras, so stock pyrender draws them), seen from the elbow; a caption strip with garment, body, step, upper-arm ratio and
gripper load is burned into each frame. ``--stride`` keeps every n-th state; the video plays at
``--fps``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([0, 0, image.width, 34], fill=(20, 20, 20))
    draw.text((10, 6), text, fill=(240, 240, 240), font=font)
    return np.asarray(image)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world pose for pyrender (camera looks down its -z)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, np.cross(z, x), z, eye
    return pose


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--names", nargs="*", default=None, help="Output stems, one per episode.")
    p.add_argument("--labels", nargs="*", default=None, help="Caption prefixes, one per episode.")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--res", type=int, nargs=2, default=[1280, 960])
    p.add_argument("--camera-offset", type=float, nargs=3, default=[1.0, -1.3, 0.6])
    args = p.parse_args()
    import os

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import imageio.v2 as imageio
    import pyrender
    import trimesh

    args.out_dir.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(*args.res)
    skin = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.87, 0.72, 0.59, 1.0], roughnessFactor=0.8,
                                             doubleSided=True)
    fabric = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.27, 0.53, 0.88, 1.0], roughnessFactor=0.9,
                                               doubleSided=True)
    dark = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 1.0])
    for i, path in enumerate(args.episodes):
        with np.load(path, allow_pickle=False) as source:
            d = {k: source[k] for k in ("positions", "faces", "tcp", "elbow", "upperarm_ratio", "gripper_force")}
            person = (source["human_vertices"], source["human_faces"]) if "human_vertices" in source \
                else (source["arm_vertices"], source["arm_faces"])
        target = np.asarray(d["elbow"]) + np.array([0.0, 0.0, -0.15])
        camera_pose = look_at(target + np.asarray(args.camera_offset), target)
        label = args.labels[i] if args.labels else path.stem
        name = args.names[i] if args.names else path.stem
        body = pyrender.Mesh.from_trimesh(trimesh.Trimesh(*person, process=False), material=skin, smooth=True)
        gripper = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(radius=0.012), material=dark)
        states = list(range(0, len(d["positions"]), args.stride))
        if states[-1] != len(d["positions"]) - 1:
            states.append(len(d["positions"]) - 1)
        frames = []
        for k in states:
            scene = pyrender.Scene(bg_color=[0.93, 0.94, 0.95, 1.0], ambient_light=[0.35, 0.35, 0.35])
            scene.add(body)
            cloth = trimesh.Trimesh(d["positions"][k], d["faces"], process=False)
            scene.add(pyrender.Mesh.from_trimesh(cloth, material=fabric, smooth=True))
            pose = np.eye(4); pose[:3, 3] = d["tcp"][k]
            scene.add(gripper, pose=pose)
            scene.add(pyrender.PerspectiveCamera(yfov=np.radians(40)), pose=camera_pose)
            scene.add(pyrender.DirectionalLight(intensity=3.0), pose=camera_pose)
            scene.add(pyrender.DirectionalLight(intensity=1.5), pose=look_at(target + np.array([-1.0, 1.0, 1.5]), target))
            rgb, _ = renderer.render(scene)
            text = (f"{label} | step {k}/{len(d['positions']) - 1} | upper arm {d['upperarm_ratio'][k]:.2f} | "
                    f"grip {np.linalg.norm(d['gripper_force'][k]):.0f} N")
            frames.append(caption(rgb, text))
        frames.extend([frames[-1]] * args.fps)          # hold the last frame for a second
        out = args.out_dir / f"{name}.mp4"
        imageio.mimsave(out, frames, fps=args.fps, macro_block_size=8)
        print(f"[render] {out} ({len(frames)} frames)", flush=True)
    renderer.delete()


if __name__ == "__main__":
    main()
