"""Genesis scene with a Franka Panda whose deformable objects are solved by libuipc.

The robot, table plane, and rendering come from Genesis. The cloth or cable is a
native libuipc geometry inserted through the Genesis IPC coupler, so contact
between the finger links and the deformable is resolved by the IPC solver in
the same Newton iterations as the deformable's own elasticity.

Grasping uses the *picker* convention of SoftGym, Wang RSS 2023, and the
Newton dressing teacher: a fixed set of vertices is attached to the tool centre
point with a soft position constraint and follows the commanded tool motion.
The finger pads stay open and still collide with the rest of the deformable.
Everything the coupler needs is reached through Genesis 1.1.2 private
attributes, which is the same access pattern as the upstream IPC examples.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .assets import Deformable, build_cable, build_cloth
from .obs import ObsSpec
from .tasks import TaskSpec, get_task

_GENESIS_INITIALISED = False
_OBS_SUBSET_SEED = 12345


@dataclass
class EnvConfig:
    """Everything that defines the MDP and the simulator settings."""

    task: str = "cloth_drag"
    horizon: int = 150
    action_repeat: int = 5
    """Simulation steps per decision. Five was validated; three leaves the cloth lagging the tool
    so a scripted drag stalls about 4 cm from its goal."""
    max_translation: float = 0.006
    """Per-axis tool displacement for a unit action, in metres per decision."""
    point_budget: int = 256
    seed: int = 0
    dt: float = 0.01
    friction: float = 0.6
    contact_resistance: float = 1e8
    d_hat: float = 0.001
    newton_tolerance: float = 0.01
    constraint_strength: float = 1000.0
    settle_steps: int = 40
    hold_steps: int = 3
    workspace_min: tuple[float, float, float] = (0.30, -0.35, 0.03)
    workspace_max: tuple[float, float, float] = (0.75, 0.35, 0.50)
    max_joint_delta: float = 0.05
    """Per simulation step clamp on the arm joint target, in radians."""
    tcp_offset: float = 0.1034
    """Distance from the hand frame origin to the tool centre point along the hand z axis."""
    progress_scale: float = 10.0
    success_bonus: float = 0.1
    show_viewer: bool = False
    logging_level: str = "warning"

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_genesis(logging_level: str):
    global _GENESIS_INITIALISED
    import genesis as gs

    if not _GENESIS_INITIALISED:
        gs.init(backend=gs.gpu, logging_level=logging_level)
        _GENESIS_INITIALISED = True
    return gs


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the unit quaternion ``q`` given as ``(w, x, y, z)``."""
    w, x, y, z = (float(c) for c in q)
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class GenesisIPCManipEnv:
    """Single-environment goal-reaching task on a cloth sheet or cable."""

    action_dim = 3

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.task: TaskSpec = get_task(cfg.task)
        self.spec = ObsSpec(cfg.point_budget)
        self.rng = np.random.default_rng(cfg.seed)
        self._gs = _ensure_genesis(cfg.logging_level)
        self._torch = __import__("torch")
        self._uipc = __import__("uipc")
        self._device = self._gs.device
        self._debug_objects: list = []
        self._build_scene()
        self._prepare_grasp()
        self.goal = np.zeros(3)
        self._prev_distance = 0.0
        self._episode_step = 0
        self._last_ik_error = 0.0

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        gs = self._gs
        cfg = self.cfg
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=1),
            coupler_options=gs.options.IPCCouplerOptions(
                constraint_strength_translation=10.0,
                constraint_strength_rotation=10.0,
                newton_semi_implicit_enable=False,
                newton_tolerance=cfg.newton_tolerance,
                contact_d_hat=cfg.d_hat,
                contact_resistance=cfg.contact_resistance,
                enable_rigid_rigid_contact=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(900, 800),
                camera_pos=(1.55, -1.35, 1.05),
                camera_lookat=(0.5, 0.0, 0.15),
                camera_fov=40,
                refresh_rate=60,
            ),
            show_viewer=cfg.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only"))
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
            material=gs.materials.Rigid(
                coup_type="two_way_soft_constraint",
                coup_links=("left_finger", "right_finger"),
                coup_friction=cfg.friction,
            ),
        )
        coupler = self.scene.sim.coupler
        self.coupler = coupler
        if self.task.deformable == "cloth":
            self.deformable: Deformable = build_cloth()
        elif self.task.deformable == "cable":
            self.deformable = build_cable()
        else:
            raise ValueError(f"Unsupported deformable {self.task.deformable!r}")
        from uipc.constitution import SoftPositionConstraint

        mesh = self.deformable.mesh
        SoftPositionConstraint().apply_to(mesh, cfg.constraint_strength)
        coupler._ipc_contact_tabular.default_model(cfg.friction, cfg.contact_resistance)
        coupler._ipc_contact_tabular.default_element().apply_to(mesh)
        ipc_object = coupler._ipc_objects.create(self.deformable.kind)
        self.slot = ipc_object.geometries().create(mesh)[0]

        rest = self.deformable.rest
        self.grasp_idx = np.asarray(self.task.grasp_vertices(rest), dtype=np.int64)
        self.marker_idx = np.asarray(self.task.marker_vertices(rest), dtype=np.int64)
        self._picker = {"active": False, "targets": rest[self.grasp_idx].copy()}
        grasp_idx = self.grasp_idx
        picker = self._picker
        uipc = self._uipc

        def animate(info):
            geo = info.geo_slots()[0].geometry()
            flags = uipc.view(geo.vertices().find(uipc.builtin.is_constrained)).reshape(-1)
            flags[:] = 0
            if picker["active"]:
                flags[grasp_idx] = 1
                aim = uipc.view(geo.vertices().find(uipc.builtin.aim_position)).reshape(-1, 3)
                aim[grasp_idx] = picker["targets"]

        coupler._ipc_animator.insert(ipc_object, animate)
        t0 = time.time()
        self.scene.build()
        self.build_seconds = time.time() - t0
        if not coupler._ipc_world.is_valid():
            raise RuntimeError("IPC world is invalid right after build; check for initial intersections")
        self._world = coupler._ipc_world

        # Fixed observation subset: markers first, then a deterministic random
        # sample of the remaining vertices, so every process sees the same layout.
        budget = self.spec.deformable_budget
        n = self.deformable.vertex_count
        if n <= budget:
            subset = np.arange(n, dtype=np.int64)
        else:
            markers = np.unique(self.marker_idx)
            others = np.setdiff1d(np.arange(n, dtype=np.int64), markers)
            sub_rng = np.random.default_rng(_OBS_SUBSET_SEED)
            # Markers are subsampled at random rather than truncated by index.
            # Vertex indices run along the mesh, so keeping the first ones
            # would show the policy one region of the sheet while the reward
            # measures the centroid of all of it.
            if markers.size > budget:
                keep_markers = np.sort(sub_rng.choice(markers, size=budget, replace=False))
            else:
                keep_markers = markers
            remaining = budget - keep_markers.size
            extra = sub_rng.choice(others, size=min(remaining, others.size), replace=False) if remaining > 0 else []
            subset = np.sort(np.concatenate([keep_markers, np.asarray(extra, dtype=np.int64)]))
        self._obs_subset = subset
        marker_set = np.zeros(n, dtype=bool)
        marker_set[self.marker_idx] = True
        self._obs_marker_mask = marker_set[subset]

    def _prepare_grasp(self) -> None:
        cfg = self.cfg
        robot = self.robot
        self.hand = robot.get_link("hand")
        self._arm_dofs = list(range(7))
        self._finger_dofs = [7, 8]
        robot.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000, 500, 500])
        robot.set_dofs_kv([100, 100], self._finger_dofs)
        self.quat_down = np.array([0.0, 1.0, 0.0, 0.0])
        rest = self.deformable.rest
        centroid = rest[self.grasp_idx].mean(axis=0)
        offsets = rest[self.grasp_idx] - centroid[None, :]
        offsets[:, 2] = 0.0
        self._grasp_offsets = offsets
        self._tcp_start = np.array([centroid[0], centroid[1], self.task.grasp_height])
        q, err = self._ik(self._tcp_start, seed=None, samples=50)
        q[7:] = 0.04
        robot.set_dofs_position(q)
        robot.control_dofs_position(q)
        self._q_target = q.copy()
        self._tcp_cmd = self._tcp_start.copy()
        self._picker["active"] = True
        self._picker["targets"] = self._tcp_cmd[None, :] + self._grasp_offsets
        before = self.positions()
        for _ in range(cfg.settle_steps):
            self._sim_step()
        self._check_world()
        after = self.positions()
        self.settle_displacement = float(np.linalg.norm(after - before, axis=1).max())
        self._snapshot_frame = int(self._world.frame())
        if not self._world.dump():
            raise RuntimeError("Failed to dump the settled IPC snapshot")
        self._snapshot_positions = after.copy()
        self._snapshot_q = robot.get_qpos().cpu().numpy().copy()
        self._snapshot_tracking = float(np.linalg.norm(self.tcp_measured() - self._tcp_cmd))

    # ------------------------------------------------------------------
    # Low level helpers
    # ------------------------------------------------------------------
    def _ik(self, tcp: np.ndarray, seed: np.ndarray | None, samples: int) -> tuple[np.ndarray, float]:
        torch = self._torch
        kwargs = dict(
            link=self.hand,
            pos=torch.as_tensor(tcp, dtype=torch.float32, device=self._device),
            quat=torch.as_tensor(self.quat_down, dtype=torch.float32, device=self._device),
            local_point=torch.as_tensor([0.0, 0.0, self.cfg.tcp_offset], dtype=torch.float32, device=self._device),
            max_samples=int(samples),
            return_error=True,
        )
        if seed is not None:
            kwargs["init_qpos"] = torch.as_tensor(seed, dtype=torch.float32, device=self._device)
        q, err = self.robot.inverse_kinematics(**kwargs)
        err = np.asarray(err.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        return q.detach().cpu().numpy().astype(np.float64), float(np.linalg.norm(err[:3]))

    def _command(self, tcp: np.ndarray) -> None:
        q, err = self._ik(tcp, seed=self._q_target, samples=1)
        self._last_ik_error = err
        lo = self._q_target[:7] - self.cfg.max_joint_delta
        hi = self._q_target[:7] + self.cfg.max_joint_delta
        self._q_target[:7] = np.clip(q[:7], lo, hi)
        self._tcp_cmd = np.asarray(tcp, dtype=np.float64).copy()
        self._picker["targets"] = self._tcp_cmd[None, :] + self._grasp_offsets

    def _sim_step(self) -> None:
        self.robot.control_dofs_position(self._q_target[:7], self._arm_dofs)
        self.robot.control_dofs_position([0.04, 0.04], self._finger_dofs)
        self.scene.step()
        if self.cfg.show_viewer:
            self._draw()

    def _check_world(self) -> None:
        if not self._world.is_valid():
            raise RuntimeError(f"Invalid IPC world at frame {self._world.frame()}")
        p = self.positions()
        if not np.isfinite(p).all():
            raise RuntimeError("Non-finite deformable positions")

    def positions(self) -> np.ndarray:
        return np.asarray(self._uipc.view(self.slot.geometry().positions())).reshape(-1, 3).copy()

    def tcp_measured(self) -> np.ndarray:
        pos = self.hand.get_pos().detach().cpu().numpy().reshape(3).astype(np.float64)
        quat = self.hand.get_quat().detach().cpu().numpy().reshape(4).astype(np.float64)
        return pos + quat_rotate(quat, np.array([0.0, 0.0, self.cfg.tcp_offset]))

    def marker_centroid(self, positions: np.ndarray | None = None) -> np.ndarray:
        p = self.positions() if positions is None else positions
        return p[self.marker_idx].mean(axis=0)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scene.reset()
        if not self._world.recover(self._snapshot_frame):
            raise RuntimeError(f"Failed to recover IPC frame {self._snapshot_frame}")
        self._world.retrieve()
        self.robot.set_dofs_position(self._snapshot_q)
        self.robot.control_dofs_position(self._snapshot_q)
        self._q_target = self._snapshot_q.copy()
        self._tcp_cmd = self._tcp_start.copy()
        self._picker["targets"] = self._tcp_cmd[None, :] + self._grasp_offsets
        restored = self.positions()
        self.reset_restore_error = float(np.abs(restored - self._snapshot_positions).max())
        for _ in range(self.cfg.hold_steps):
            self._sim_step()
        self._check_world()
        p = self.positions()
        centroid = self.marker_centroid(p)
        goal = np.asarray(self.task.sample_goal(self.rng, self.deformable.rest, centroid), dtype=np.float64)
        goal[:2] = np.clip(goal[:2], np.asarray(self.cfg.workspace_min[:2]), np.asarray(self.cfg.workspace_max[:2]))
        self.goal = goal
        self._prev_distance = float(np.linalg.norm(centroid - goal))
        self._episode_step = 0
        return self.observation(p)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        cfg = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.action_dim), -1.0, 1.0)
        start = self._tcp_cmd.copy()
        target = np.clip(start + a * cfg.max_translation, np.asarray(cfg.workspace_min), np.asarray(cfg.workspace_max))
        ik_errors = []
        for k in range(cfg.action_repeat):
            frac = (k + 1) / cfg.action_repeat
            self._command(start + frac * (target - start))
            ik_errors.append(self._last_ik_error)
            self._sim_step()
        self._check_world()
        p = self.positions()
        centroid = self.marker_centroid(p)
        distance = float(np.linalg.norm(centroid - self.goal))
        success = distance < self.task.success_tolerance
        reward = cfg.progress_scale * (self._prev_distance - distance) + (cfg.success_bonus if success else 0.0)
        self._prev_distance = distance
        self._episode_step += 1
        done = self._episode_step >= cfg.horizon
        tracking = float(np.linalg.norm(self.tcp_measured() - self._tcp_cmd))
        info = {
            "success": bool(success),
            "distance": distance,
            "tracking_error": tracking,
            "ik_error": float(max(ik_errors)),
            "episode_step": int(self._episode_step),
            "time_limit": bool(done),
        }
        return self.observation(p), float(reward), bool(done), info

    def observation(self, positions: np.ndarray | None = None) -> np.ndarray:
        p = self.positions() if positions is None else positions
        tcp = self.tcp_measured()
        rel = p[self._obs_subset] - tcp[None, :]
        return self.spec.pack(rel, self._obs_marker_mask, self.goal - tcp, tcp, attached=self._picker["active"])

    def state(self) -> dict:
        p = self.positions()
        return {
            "positions": p,
            "tcp": self.tcp_measured(),
            "tcp_cmd": self._tcp_cmd.copy(),
            "goal": self.goal.copy(),
            "marker_centroid": self.marker_centroid(p),
            "qpos": self.robot.get_qpos().detach().cpu().numpy().copy(),
            "frame": int(self._world.frame()),
        }

    def describe(self) -> dict:
        """Static description recorded in checkpoints and trajectory files."""
        return {
            "config": self.cfg.to_dict(),
            "task": self.task.name,
            "task_description": self.task.description,
            "deformable": self.deformable.kind,
            "vertex_count": self.deformable.vertex_count,
            "grasp_vertices": self.grasp_idx.tolist(),
            "marker_vertex_count": int(self.marker_idx.size),
            "obs_dim": self.spec.dim,
            "action_dim": self.action_dim,
            "faces": self.deformable.faces.tolist(),
            "edges": self.deformable.edges.tolist(),
            "radius": self.deformable.radius,
            "provenance": self.deformable.provenance,
            "build_seconds": float(self.build_seconds),
            "settle_displacement_m": float(self.settle_displacement),
            "snapshot_tracking_error_m": float(self._snapshot_tracking),
        }

    def close(self) -> None:
        self.scene = None

    # ------------------------------------------------------------------
    # Viewer drawing (only used with ``show_viewer``)
    # ------------------------------------------------------------------
    def _draw(self) -> None:
        for obj in self._debug_objects:
            self.scene.clear_debug_object(obj)
        self._debug_objects = []
        p = self.positions()
        if self.deformable.kind == "cloth":
            import trimesh

            visual = trimesh.Trimesh(p, self.deformable.faces, process=False)
            visual.visual.vertex_colors = np.tile([50, 115, 220, 255], (len(p), 1))
            drawing = self.scene.draw_debug_mesh(visual)
            for primitive in drawing.primitives:
                primitive.material.doubleSided = True
        else:
            drawing = self.scene.draw_debug_trajectory(p, radius=self.deformable.radius, color=(0.95, 0.38, 0.08, 1))
        self._debug_objects.append(drawing)
        if np.any(self.goal):
            self._debug_objects.append(self.scene.draw_debug_sphere(self.goal, radius=0.012, color=(0.2, 0.85, 0.3, 1)))
            self._debug_objects.append(
                self.scene.draw_debug_sphere(self.marker_centroid(p), radius=0.008, color=(0.9, 0.2, 0.2, 1))
            )
