"""Replay saved FMVP PyBullet cloth trajectories in the native Genesis viewer.

The PyBullet recorder saves only arm landmarks, not a human body mesh. The two
drawn arm segments are visual guides; they are not the original collision body.
Space pauses, arrows step, Tab switches episodes, and Home restarts. No plots.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
DEFAULT_OBJ = (SCRIPT.parents[3] / "fmvp_pb/dressing_pb/assistive_gym/assets/data/"
               "cloth3d/train/Tshirt/tshirt_26.obj")
DEFAULT_SEMANTICS = (SCRIPT.parents[3] / "newton/exts/newton_isaaclab_tasks/"
                     "newton_isaaclab_tasks/dressing/data/garment_semantics/"
                     "tshirt_26__s4.0000_auto_semantics.npz")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episodes", type=Path, nargs="+")
    ap.add_argument("--obj", type=Path, default=DEFAULT_OBJ)
    ap.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--camera-offset", type=float, nargs=3, default=[1., -1.3, .6],
                    metavar=("DX", "DY", "DZ"))
    args = ap.parse_args()

    import genesis as gs
    import trimesh
    from genesis.vis.keybindings import Key, Keybind

    template = trimesh.load(args.obj, process=False)
    with np.load(args.semantics, allow_pickle=False) as source:
        rings = [source[f"right_sleeve_ring_{i:02d}"].astype(np.int64)
                 for i in range(int(source["right_ring_count"][0]))]
    data = []
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as source:
            d = {key: source[key] for key in ("cloth", "line", "tcp", "upperarm_ratio")}
            d["meta"] = json.loads(str(source["metadata_json"]))
        if d["cloth"].shape[1] != len(template.vertices):
            raise ValueError(f"Garment vertex count does not match OBJ: {path}")
        if d["line"].shape != (len(d["cloth"]), 3, 3):
            raise ValueError(f"Arm landmarks are not aligned: {path}")
        d["label"] = path.stem
        data.append(d)

    lookat = np.asarray(data[0]["line"][0, 1], dtype=float)
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.1, gravity=(0, 0, 0)),
        viewer_options=gs.options.ViewerOptions(
            res=(1200, 900), camera_pos=tuple(lookat + np.asarray(args.camera_offset)),
            camera_lookat=tuple(lookat), camera_fov=40, refresh_rate=30,
            run_in_thread=False, realtime_factor=None),
        vis_options=gs.options.VisOptions(ambient_light=(.35, .35, .35),
                                         show_world_frame=False),
        show_viewer=True)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(),
                     surface=gs.surfaces.Default(color=(.82, .84, .86)))
    scene.build()
    frame = args.start_frame if args.start_frame >= 0 else len(data[0]["cloth"]) + args.start_frame
    state = {"episode": 0, "frame": int(np.clip(frame, 0, len(data[0]["cloth"]) - 1)),
             "paused": args.paused, "changed": True}
    drawings = []

    def pause():
        state["paused"] = not state["paused"]

    def move(delta):
        state["paused"] = True
        state["frame"] = int(np.clip(state["frame"] + delta, 0,
                                     len(data[state["episode"]]["cloth"]) - 1))
        state["changed"] = True

    def next_episode():
        state["episode"] = (state["episode"] + 1) % len(data)
        state["frame"] = 0
        state["changed"] = True

    def restart():
        state["frame"] = 0
        state["changed"] = True

    scene.viewer.register_keybinds(
        Keybind("pb_pause", Key.SPACE, callback=pause),
        Keybind("pb_previous", Key.LEFT, callback=move, args=(-1,)),
        Keybind("pb_next", Key.RIGHT, callback=move, args=(1,)),
        Keybind("pb_episode", Key.TAB, callback=next_episode),
        Keybind("pb_restart", Key.HOME, callback=restart), overwrite=True)
    print("[viewer] READY: PyBullet cloth, arm landmarks as guides; Space/arrows/Tab/Home", flush=True)
    next_frame_at = time.monotonic()
    end_at = None
    try:
        while scene.viewer.is_alive():
            now = time.monotonic()
            if state["changed"]:
                d, k = data[state["episode"]], state["frame"]
                with scene.viewer.lock:
                    for drawing in drawings:
                        scene.clear_debug_object(drawing)
                    drawings.clear()
                    for start, end, radius in ((d["line"][k, 0], d["line"][k, 1], .04),
                                               (d["line"][k, 1], d["line"][k, 2], .055)):
                        arm = trimesh.creation.cylinder(radius=radius,
                                                        segment=np.stack((start, end)), sections=12)
                        arm.visual.vertex_colors = np.tile([222, 184, 150, 255], (len(arm.vertices), 1))
                        drawings.append(scene.draw_debug_mesh(arm))
                    cloth = trimesh.Trimesh(vertices=d["cloth"][k], faces=template.faces,
                                            process=False)
                    cloth.visual.vertex_colors = np.tile([70, 135, 225, 255], (len(cloth.vertices), 1))
                    drawing = scene.draw_debug_mesh(cloth)
                    for primitive in drawing.primitives:
                        primitive.material.doubleSided = True
                    drawings.append(drawing)
                    drawings.append(scene.draw_debug_sphere(d["tcp"][k], radius=.013,
                                                            color=(.15, .15, .15, 1)))
                    for i, ring in enumerate(rings):
                        center = d["cloth"][k, ring].mean(axis=0)
                        frac = i / max(1, len(rings) - 1)
                        drawings.append(scene.draw_debug_sphere(
                            center, radius=.007,
                            color=(1. - .65 * frac, .12 + .65 * frac, .15 + .65 * frac, 1.)))
                scene.viewer._pyrender_viewer.set_caption(
                    f"Genesis | PyBullet {d['label']} | {k}/{len(d['cloth'])-1} | "
                    f"upper {d['upperarm_ratio'][k]:.3f} | arm proxy, not collision mesh | "
                    "Space/arrows/Tab/Home")
                state["changed"] = False
            scene.viewer.update(force=True)
            if not state["paused"] and now >= next_frame_at:
                last = len(data[state["episode"]]["cloth"]) - 1
                if state["frame"] < last:
                    state["frame"] += 1
                    state["changed"] = True
                    end_at = None
                elif end_at is None:
                    end_at = now + 2.
                elif now >= end_at:
                    next_episode()
                    end_at = None
                next_frame_at = now + 1. / args.fps
            time.sleep(.005)
    except KeyboardInterrupt:
        pass
    finally:
        scene.destroy()


if __name__ == "__main__":
    main()
