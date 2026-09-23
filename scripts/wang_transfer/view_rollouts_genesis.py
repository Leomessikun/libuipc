"""Inspect recorded IPC meshes in the native Genesis viewer, without resimulating.

Space: pause/play; Left/Right: step; Tab: next trajectory; Home: restart.
The viewer loops through the supplied episodes and preserves their measured states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path, nargs="+")
    p.add_argument("--fps", type=float, default=10.)
    p.add_argument("--paused", action="store_true")
    p.add_argument("--start-frame", type=int, default=0,
                   help="Initial frame of the first episode; -1 opens at the last recorded state.")
    p.add_argument("--camera-offset", type=float, nargs=3, default=[1., -1.3, .6],
                   metavar=("DX", "DY", "DZ"),
                   help="Genesis viewer camera position relative to the elbow-centered look-at point.")
    p.add_argument("--show-ring-centers", action="store_true",
                   help="Overlay the seven semantic sleeve-ring centers in the Genesis viewer.")
    p.add_argument("--semantics", type=Path, default=(Path(__file__).resolve().parents[3] / "newton/"
                   "exts/newton_isaaclab_tasks/newton_isaaclab_tasks/dressing/data/"
                   "garment_semantics/tshirt_26__s4.0000_auto_semantics.npz"))
    args = p.parse_args()
    import genesis as gs
    import trimesh
    from genesis.vis.keybindings import Key, Keybind

    rings = []
    if args.show_ring_centers:
        with np.load(args.semantics, allow_pickle=False) as source:
            rings = [source[f"right_sleeve_ring_{i:02d}"].astype(np.int64)
                     for i in range(int(source["right_ring_count"][0]))]

    data = []
    for path in args.episodes:
        with np.load(path, allow_pickle=False) as source:
            episode = {k: source[k] for k in ["positions", "faces", "arm_vertices", "arm_faces",
                       "tcp", "shoulder", "elbow", "upperarm_ratio", "gripper_force", "metadata_json"]}
            if "human_vertices" in source and "human_faces" in source:
                episode["human_vertices"] = source["human_vertices"]
                episode["human_faces"] = source["human_faces"]
        episode["meta"] = json.loads(str(episode.pop("metadata_json")))
        if rings and max(int(r.max()) for r in rings) >= episode["positions"].shape[1]:
            raise ValueError(f"Sleeve-ring semantics do not match cloth mesh: {path}")
        episode["label"] = f"{path.parent.parent.name}/{path.stem}"
        data.append(episode)
    first = data[0]
    lookat = np.asarray(first["elbow"]) + np.array([0., 0., -.15])
    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=.1, gravity=(0, 0, 0)),
        viewer_options=gs.options.ViewerOptions(
            res=(1200, 900), camera_pos=tuple(lookat + np.asarray(args.camera_offset)),
            camera_lookat=tuple(lookat), camera_fov=40, refresh_rate=30,
            run_in_thread=False, realtime_factor=None),
        vis_options=gs.options.VisOptions(ambient_light=(.35, .35, .35), show_world_frame=False),
        show_viewer=True)
    scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(), surface=gs.surfaces.Default(color=(.82, .84, .86)))
    scene.build()
    initial = args.start_frame if args.start_frame >= 0 else len(first["positions"]) + args.start_frame
    state = dict(episode=0, frame=int(np.clip(initial, 0, len(first["positions"]) - 1)),
                 paused=args.paused, changed=True)
    meshes = []

    def toggle():
        state["paused"] = not state["paused"]

    def move(delta):
        state["paused"] = True
        state["frame"] = int(np.clip(state["frame"] + delta, 0, len(data[state["episode"]]["positions"]) - 1))
        state["changed"] = True

    def next_episode():
        state["episode"] = (state["episode"] + 1) % len(data)
        state["frame"] = 0
        state["changed"] = True

    def restart():
        state["frame"] = 0
        state["changed"] = True

    scene.viewer.register_keybinds(
        Keybind("replay_pause", Key.SPACE, callback=toggle),
        Keybind("replay_previous_frame", Key.LEFT, callback=move, args=(-1,)),
        Keybind("replay_next_frame", Key.RIGHT, callback=move, args=(1,)),
        Keybind("replay_next_episode", Key.TAB, callback=next_episode),
        Keybind("replay_restart", Key.HOME, callback=restart), overwrite=True)
    print("[viewer] READY. Space pause/play; arrows step; Tab next episode; Home restart.", flush=True)
    next_frame = time.monotonic()
    end_hold_until = None
    try:
        while scene.viewer.is_alive():
            now = time.monotonic()
            if state["changed"]:
                d, k = data[state["episode"]], state["frame"]
                with scene.viewer.lock:
                    for mesh in meshes:
                        scene.clear_debug_object(mesh)
                    meshes.clear()
                    person = (d.get("human_vertices", d["arm_vertices"]),
                              d.get("human_faces", d["arm_faces"]), [222, 184, 150, 255])
                    for vertices, faces, color in [person,
                                                   (d["positions"][k], d["faces"], [70, 135, 225, 255])]:
                        mesh = trimesh.Trimesh(vertices, faces, process=False)
                        mesh.visual.vertex_colors = np.tile(color, (len(vertices), 1))
                        drawing = scene.draw_debug_mesh(mesh)
                        for primitive in drawing.primitives:
                            primitive.material.doubleSided = True
                        meshes.append(drawing)
                    meshes.append(scene.draw_debug_sphere(d["tcp"][k], radius=.012, color=(.15, .15, .15, 1)))
                    meshes.append(scene.draw_debug_sphere(d["shoulder"], radius=.01, color=(.2, .85, .3, 1)))
                    for i, ring in enumerate(rings):
                        center = d["positions"][k, ring].mean(axis=0)
                        frac = i / max(1, len(rings) - 1)
                        color = (1. - .65 * frac, .12 + .65 * frac, .15 + .65 * frac, 1.)
                        meshes.append(scene.draw_debug_sphere(center, radius=.007, color=color))
                m = d["meta"]
                if "stage" in m:
                    phase = f"{m['stage'].upper()} PREFIX / {m.get('quality_class', 'TOPOLOGY UNVERIFIED')}"
                    if k == m["milestone_state"]:
                        phase += " / MILESTONE"
                else:
                    phase = ("RATIO HOLD / SLEEVE UNVERIFIED"
                             if m["success_state"] is not None and k >= m["success_state"] else "DRESS")
                caption = (f"Genesis | {d['label']} | body {m['body']} | {k}/{len(d['positions'])-1} {phase} | "
                           f"{m.get('collision_geometry', 'arm')} collision | "
                           f"upper {d['upperarm_ratio'][k]:.3f} | grip {np.linalg.norm(d['gripper_force'][k]):.1f} N | "
                           "Space pause, arrows step, Tab next")
                scene.viewer._pyrender_viewer.set_caption(caption)
                state["changed"] = False
            scene.viewer.update(force=True)
            if not state["paused"] and now >= next_frame:
                last = len(data[state["episode"]]["positions"]) - 1
                if state["frame"] < last:
                    state["frame"] += 1
                    end_hold_until = None
                    state["changed"] = True
                elif end_hold_until is None:
                    end_hold_until = now + 2.
                elif now >= end_hold_until:
                    next_episode()
                    end_hold_until = None
                next_frame = now + 1. / args.fps
            time.sleep(.005)
    except KeyboardInterrupt:
        pass
    finally:
        scene.destroy()


if __name__ == "__main__":
    main()
