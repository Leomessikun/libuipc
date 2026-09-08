"""Cloth and cable with movable soft grips; headless validation or Polyscope GUI."""

import argparse
import json
from pathlib import Path

import numpy as np
from uipc import Logger, builtin, view
from uipc.constitution import (
    DiscreteShellBending,
    ElasticModuli2D,
    HookeanSpring,
    KirchhoffRodBending,
    NeoHookeanShell,
    SoftPositionConstraint,
)
from uipc.core import Engine, Scene, World
from uipc.geometry import ground, label_surface, linemesh, trimesh


def build(output):
    output.mkdir(parents=True, exist_ok=True)
    Logger.set_level(Logger.Level.Warn)
    engine = Engine("cuda", str(output))
    world = World(engine)
    config = Scene.default_config()
    config["dt"] = 0.01
    config["contact"]["d_hat"] = 0.003
    scene = Scene(config)
    scene.contact_tabular().default_model(0.3, 1e8)
    contact = scene.contact_tabular().default_element()
    n = 17
    vertices = np.array(
        [
            [-0.95 + i * 0.6 / (n - 1), 1.15, -0.3 + j * 0.6 / (n - 1)]
            for j in range(n)
            for i in range(n)
        ]
    )
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            faces.extend([[a, a + n, a + 1], [a + 1, a + n, a + n + 1]])
    faces = np.array(faces, dtype=np.int32)
    cloth = trimesh(vertices, faces)
    label_surface(cloth)
    NeoHookeanShell().apply_to(
        cloth, ElasticModuli2D.youngs_poisson(5e4, 0.35), 200.0, 0.001
    )
    DiscreteShellBending().apply_to(cloth, 5e4, 0.35)
    t = np.linspace(0, 1, 41)
    points = np.column_stack(
        (0.2 + 0.9 * t, 1.0 - 0.2 * np.sin(np.pi * t), np.zeros_like(t))
    )
    edges = np.column_stack((np.arange(40), np.arange(1, 41))).astype(np.int32)
    cable = linemesh(points, edges)
    label_surface(cable)
    HookeanSpring().apply_to(cable, 2e6, 1000.0, 0.006)
    KirchhoffRodBending().apply_to(cable, 1e5)
    state = {
        "engine": engine,
        "world": world,
        "scene": scene,
        "faces": faces,
        "edges": edges,
        "offsets": {"cloth": np.zeros(3), "cable": np.zeros(3)},
        "automatic": True,
        "slots": {},
        "rest": {},
        "grips": {"cloth": [0, n - 1], "cable": [0, len(t) - 1]},
        "targets": {},
    }
    for name, mesh in [("cloth", cloth), ("cable", cable)]:
        SoftPositionConstraint().apply_to(mesh, 1000.0)
        contact.apply_to(mesh)
        obj = scene.objects().create(name)
        state["rest"][name] = np.array(view(mesh.positions())).reshape(-1, 3).copy()
        state["slots"][name] = obj.geometries().create(mesh)[0]

        def animate(info, name=name):
            geo = info.geo_slots()[0].geometry()
            grips = state["grips"][name]
            flags = view(geo.vertices().find(builtin.is_constrained)).reshape(-1)
            flags[:] = 0
            flags[grips] = 1
            targets = state["rest"][name][grips].copy()
            if state["automatic"]:
                phase = min(info.frame() / 120, 1.0) * np.pi
                delta = np.array([0, 0.16 * np.sin(phase / 2), 0.14 * np.sin(phase)])
                targets[-1] += delta
            targets[-1] += state["offsets"][name]
            view(geo.vertices().find(builtin.aim_position)).reshape(-1, 3)[grips] = (
                targets
            )
            state["targets"][name] = targets

        scene.animator().insert(obj, animate)
    floor = ground(0.0)
    contact.apply_to(floor)
    scene.objects().create("floor").geometries().create(floor)
    world.init(scene)
    if not world.is_valid():
        raise RuntimeError("Scene initialization failed")
    return state


def positions(state):
    return {
        name: np.array(view(slot.geometry().positions())).reshape(-1, 3).copy()
        for name, slot in state["slots"].items()
    }


def step(state):
    state["world"].advance()
    state["world"].retrieve()
    if not state["world"].is_valid():
        raise RuntimeError(f"Invalid simulation at frame {state['world'].frame()}")
    data = positions(state)
    if not all(np.isfinite(p).all() for p in data.values()):
        raise RuntimeError("Non-finite simulated positions")
    return data


def headless(state, frames, output):
    history = [positions(state)]
    max_error = {"cloth": 0.0, "cable": 0.0}
    for _ in range(frames):
        data = step(state)
        history.append(data)
        for name in data:
            error = np.linalg.norm(
                data[name][state["grips"][name]] - state["targets"][name], axis=1
            ).max()
            max_error[name] = max(max_error[name], float(error))
    metrics = {"frames": state["world"].frame(), "dt_seconds": 0.01, "objects": {}}
    for name, grip_error in max_error.items():
        final = history[-1][name]
        displacement = np.linalg.norm(final - state["rest"][name], axis=1)
        metrics["objects"][name] = {
            "vertices": len(final),
            "max_grip_error_m": grip_error,
            "max_displacement_m": float(displacement.max()),
            "moving_grip_displacement_m": float(displacement[state["grips"][name][-1]]),
            "min_height_m": float(final[:, 1].min()),
        }
    np.savez_compressed(
        output / "trajectory.npz",
        **{name: np.stack([p[name] for p in history]) for name in max_error},
        faces=state["faces"],
        edges=state["edges"],
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    if any(error > 0.03 for error in max_error.values()):
        raise RuntimeError("A grip missed its target by more than 3 cm")
    if frames >= 120 and any(
        m["moving_grip_displacement_m"] < 0.1 for m in metrics["objects"].values()
    ):
        raise RuntimeError("Manipulation did not move both grips by at least 10 cm")
    return history


def preview(state, history, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(projection="3d")

    def draw(k):
        ax.clear()
        p = history[k]
        # Plot with simulator Y as the vertical axis.
        c = p["cloth"][:, [0, 2, 1]]
        ax.add_collection3d(
            Poly3DCollection(
                c[state["faces"]],
                facecolor="#52a9da",
                edgecolor="#23618a",
                linewidth=0.2,
                alpha=0.9,
            )
        )
        r = p["cable"][:, [0, 2, 1]]
        ax.plot(*r.T, color="#eb8c32", linewidth=4)
        for name in p:
            grips = p[name][state["grips"][name]][:, [0, 2, 1]]
            ax.scatter(*grips.T, color="#c73549", s=35)
        ax.set(
            xlim=(-1.1, 1.25),
            ylim=(-0.65, 0.65),
            zlim=(0, 1.5),
            xlabel="X (m)",
            ylabel="Z (m)",
            zlabel="Y (m)",
            title=f"Cloth + cable: moving soft grips | t = {k * 0.01:.2f} s",
        )
        ax.view_init(elev=25, azim=-65)

    selected = list(range(0, len(history), 4))
    if selected[-1] != len(history) - 1:
        selected.append(len(history) - 1)
    anim = FuncAnimation(fig, draw, frames=selected, interval=40)
    anim.save(output / "manipulation.gif", writer=PillowWriter(fps=25))
    draw(len(history) - 1)
    fig.savefig(output / "final.png", dpi=150)
    plt.close(fig)


def gui(state):
    import polyscope as ps
    import polyscope.imgui as ui

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_height(0)
    p = positions(state)
    cloth = ps.register_surface_mesh(
        "cloth", p["cloth"], state["faces"], color=(0.2, 0.6, 0.85)
    )
    cable = ps.register_curve_network(
        "cable",
        p["cable"],
        state["edges"],
        color=(0.95, 0.5, 0.1),
        radius=0.006,
        relative=False,
    )
    running = False

    def callback():
        nonlocal running
        _, running = ui.Checkbox("Run", running)
        _, state["automatic"] = ui.Checkbox("Scripted motion", state["automatic"])
        ui.TextUnformatted("Move the second grip on each object (meters).")
        for name in state["offsets"]:
            for axis, label in enumerate("XYZ"):
                _, value = ui.SliderFloat(
                    f"{name} {label}", float(state["offsets"][name][axis]), -0.2, 0.2
                )
                state["offsets"][name][axis] = value
        single = ui.Button("Step")
        if running or single:
            current = step(state)
            cloth.update_vertex_positions(current["cloth"])
            cable.update_node_positions(current["cable"])
        ui.TextUnformatted(f"Frame: {state['world'].frame()}")

    ps.set_user_callback(callback)
    ps.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/cloth_cable"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    state = build(args.output)
    if args.gui:
        gui(state)
    else:
        history = headless(state, args.frames, args.output)
        if args.preview:
            preview(state, history, args.output)


if __name__ == "__main__":
    main()
