"""Batched Genesis scene with Franka Pandas whose deformables are solved by libuipc.

One Genesis scene holds ``num_envs`` robots and ``num_envs`` copies of the
deformable, and one libuipc world solves all of them together. Each copy lives
in its own IPC subscene, so contact never crosses environments, while the
Newton solve and every kernel launch cover the whole batch at once. This is
the IPC counterpart of the batched simulation the Newton dressing teacher
trains against, and it is what makes the transition budget reachable.

Grasping uses the *picker* convention of SoftGym, Wang RSS 2023, and the
Newton teacher: a fixed set of vertices is attached to the tool centre point
with a soft position constraint and follows the commanded tool motion. The
finger pads stay open and still collide with the rest of the deformable.

Episodes are synchronised: every environment runs the same fixed horizon and
the whole world resets together from a dumped IPC snapshot, which is also how
the reference handles its fixed-horizon slots. The coupler is reached through
Genesis 1.1.2 private attributes, the same access pattern as the upstream IPC
examples.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

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
    progress_scale: float = 1.0
    """Reward per unit of progress, where one unit is ``max_translation`` of marker motion toward the goal.

    A unit action straight at the goal therefore earns about +1 per decision, which keeps per-step
    rewards on the reference's ``[-1, 1]`` scale that the SAC temperature and learning rates were
    calibrated for. Ten times smaller rewards left the entropy term dominating the policy objective."""
    success_bonus: float = 1.0
    """Added on every decision whose marker is within the task tolerance of the goal."""
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
    """Rotate ``v`` (``[..., 3]``) by unit quaternions ``q`` (``[..., 4]``, ``w, x, y, z``)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = q[..., :1]
    u = q[..., 1:]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class GenesisIPCManipEnv:
    """Batched goal-reaching task on a cloth sheet or cable; one libuipc world for all environments."""

    action_dim = 3

    def __init__(self, cfg: EnvConfig, num_envs: int = 1) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be at least 1")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.task: TaskSpec = get_task(cfg.task)
        self.spec = ObsSpec(cfg.point_budget)
        self.obs_dim = self.spec.dim
        self.rngs = [np.random.default_rng(cfg.seed * 1000 + i) for i in range(self.num_envs)]
        self._gs = _ensure_genesis(cfg.logging_level)
        self._torch = __import__("torch")
        self._uipc = __import__("uipc")
        self._device = self._gs.device
        self._debug_objects: list = []
        self._build_scene()
        self._prepare_grasp()
        self.goals = np.zeros((self.num_envs, 3))
        self._prev_distance = np.zeros(self.num_envs)
        self._episode_step = 0
        self._last_ik_error = np.zeros(self.num_envs)
        self.descriptions = [self.describe()] * self.num_envs

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _make_deformable(self) -> Deformable:
        if self.task.deformable == "cloth":
            return build_cloth()
        if self.task.deformable == "cable":
            return build_cable()
        raise ValueError(f"Unsupported deformable {self.task.deformable!r}")

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
        coupler._ipc_contact_tabular.default_model(cfg.friction, cfg.contact_resistance)

        self.deformable = self._make_deformable()
        rest = self.deformable.rest
        self.grasp_idx = np.asarray(self.task.grasp_vertices(rest), dtype=np.int64)
        self.marker_idx = np.asarray(self.task.marker_vertices(rest), dtype=np.int64)
        self.slots: list = []
        self._pickers: list[dict] = []
        uipc = self._uipc
        from uipc.constitution import SoftPositionConstraint

        # The per-environment subscenes only exist once the coupler has run
        # ``_init_ipc`` inside ``scene.build``. Wrapping ``_add_objects_to_ipc``
        # inserts the deformable copies at the one point where the subscenes
        # exist and the world has not been initialised yet.
        original_add_objects = coupler._add_objects_to_ipc

        def add_objects_with_deformables() -> None:
            original_add_objects()
            for env_idx in range(self.num_envs):
                deformable = self.deformable if env_idx == 0 else self._make_deformable()
                mesh = deformable.mesh
                SoftPositionConstraint().apply_to(mesh, cfg.constraint_strength)
                coupler._ipc_contact_tabular.default_element().apply_to(mesh)
                coupler._ipc_subscenes[env_idx].apply_to(mesh)
                ipc_object = coupler._ipc_objects.create(f"manip_{deformable.kind}_{env_idx}")
                self.slots.append(ipc_object.geometries().create(mesh)[0])
                picker = {"active": False, "targets": rest[self.grasp_idx].copy()}
                self._pickers.append(picker)
                grasp_idx = self.grasp_idx

                def animate(info, picker=picker, grasp_idx=grasp_idx):
                    geo = info.geo_slots()[0].geometry()
                    flags = uipc.view(geo.vertices().find(uipc.builtin.is_constrained)).reshape(-1)
                    flags[:] = 0
                    if picker["active"]:
                        flags[grasp_idx] = 1
                        aim = uipc.view(geo.vertices().find(uipc.builtin.aim_position)).reshape(-1, 3)
                        aim[grasp_idx] = picker["targets"]

                coupler._ipc_animator.insert(ipc_object, animate)

        coupler._add_objects_to_ipc = add_objects_with_deformables
        t0 = time.time()
        self.scene.build(n_envs=self.num_envs)
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
        tcps = np.tile(self._tcp_start, (self.num_envs, 1))
        q, _ = self._ik(tcps, seed=None, samples=50)
        q[:, 7:] = 0.04
        robot.set_dofs_position(q)
        robot.control_dofs_position(q)
        self._q_target = q.copy()
        self._tcp_cmd = tcps.copy()
        for picker in self._pickers:
            picker["active"] = True
        self._update_picker_targets()
        before = self.positions()
        for _ in range(cfg.settle_steps):
            self._sim_step()
        self._check_world()
        after = self.positions()
        self.settle_displacement = float(np.linalg.norm(after - before, axis=-1).max())
        self._snapshot_frame = int(self._world.frame())
        if not self._world.dump():
            raise RuntimeError("Failed to dump the settled IPC snapshot")
        self._snapshot_positions = after.copy()
        self._snapshot_q = robot.get_qpos().cpu().numpy().copy()
        self._snapshot_tracking = float(np.linalg.norm(self.tcp_measured() - self._tcp_cmd, axis=-1).max())

    # ------------------------------------------------------------------
    # Low level helpers
    # ------------------------------------------------------------------
    def _ik(self, tcps: np.ndarray, seed: np.ndarray | None, samples: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self._torch
        n = self.num_envs
        kwargs = dict(
            link=self.hand,
            pos=torch.as_tensor(np.asarray(tcps, dtype=np.float32).reshape(n, 3), device=self._device),
            quat=torch.as_tensor(np.tile(self.quat_down, (n, 1)).astype(np.float32), device=self._device),
            local_point=torch.as_tensor([0.0, 0.0, self.cfg.tcp_offset], dtype=torch.float32, device=self._device),
            max_samples=int(samples),
            return_error=True,
        )
        if seed is not None:
            kwargs["init_qpos"] = torch.as_tensor(np.asarray(seed, dtype=np.float32), device=self._device)
        q, err = self.robot.inverse_kinematics(**kwargs)
        err = np.asarray(err.detach().cpu().numpy(), dtype=np.float64).reshape(n, -1)
        return q.detach().cpu().numpy().astype(np.float64).reshape(n, -1), np.linalg.norm(err[:, :3], axis=-1)

    def _update_picker_targets(self) -> None:
        for env_idx, picker in enumerate(self._pickers):
            picker["targets"] = self._tcp_cmd[env_idx][None, :] + self._grasp_offsets

    def _command(self, tcps: np.ndarray) -> None:
        q, err = self._ik(tcps, seed=self._q_target, samples=1)
        self._last_ik_error = err
        lo = self._q_target[:, :7] - self.cfg.max_joint_delta
        hi = self._q_target[:, :7] + self.cfg.max_joint_delta
        self._q_target[:, :7] = np.clip(q[:, :7], lo, hi)
        self._tcp_cmd = np.asarray(tcps, dtype=np.float64).reshape(self.num_envs, 3).copy()
        self._update_picker_targets()

    def _sim_step(self) -> None:
        self.robot.control_dofs_position(self._q_target[:, :7], self._arm_dofs)
        self.robot.control_dofs_position(np.full((self.num_envs, 2), 0.04), self._finger_dofs)
        self.scene.step()
        if self.cfg.show_viewer:
            self._draw()

    def _check_world(self) -> None:
        if not self._world.is_valid():
            raise RuntimeError(f"Invalid IPC world at frame {self._world.frame()}")
        if not np.isfinite(self.positions()).all():
            raise RuntimeError("Non-finite deformable positions")

    def positions(self) -> np.ndarray:
        """Deformable vertex positions for every environment, ``[num_envs, V, 3]``."""
        view = self._uipc.view
        return np.stack([np.asarray(view(slot.geometry().positions())).reshape(-1, 3) for slot in self.slots])

    def tcp_measured(self) -> np.ndarray:
        pos = self.hand.get_pos().detach().cpu().numpy().reshape(self.num_envs, 3).astype(np.float64)
        quat = self.hand.get_quat().detach().cpu().numpy().reshape(self.num_envs, 4).astype(np.float64)
        return pos + quat_rotate(quat, np.tile([0.0, 0.0, self.cfg.tcp_offset], (self.num_envs, 1)))

    def marker_centroids(self, positions: np.ndarray | None = None) -> np.ndarray:
        p = self.positions() if positions is None else positions
        return p[:, self.marker_idx].mean(axis=1)

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------
    def reset(self, seeds: list[int | None] | None = None) -> np.ndarray:
        """Reset every environment together and return ``[num_envs, obs_dim]`` observations."""
        if seeds is not None:
            for i, seed in enumerate(seeds):
                if seed is not None:
                    self.rngs[i] = np.random.default_rng(int(seed))
        self.scene.reset()
        if not self._world.recover(self._snapshot_frame):
            raise RuntimeError(f"Failed to recover IPC frame {self._snapshot_frame}")
        self._world.retrieve()
        self.robot.set_dofs_position(self._snapshot_q)
        self.robot.control_dofs_position(self._snapshot_q)
        self._q_target = self._snapshot_q.copy()
        self._tcp_cmd = np.tile(self._tcp_start, (self.num_envs, 1))
        self._update_picker_targets()
        self.reset_restore_error = float(np.abs(self.positions() - self._snapshot_positions).max())
        for _ in range(self.cfg.hold_steps):
            self._sim_step()
        self._check_world()
        p = self.positions()
        centroids = self.marker_centroids(p)
        lo = np.asarray(self.cfg.workspace_min[:2])
        hi = np.asarray(self.cfg.workspace_max[:2])
        goals = []
        for env_idx in range(self.num_envs):
            goal = np.asarray(self.task.sample_goal(self.rngs[env_idx], self.deformable.rest, centroids[env_idx]))
            goal = goal.astype(np.float64)
            goal[:2] = np.clip(goal[:2], lo, hi)
            goals.append(goal)
        self.goals = np.stack(goals)
        self._prev_distance = np.linalg.norm(centroids - self.goals, axis=-1)
        self._episode_step = 0
        return self.observation(p)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Advance every environment by one decision.

        Returns ``(obs, rewards, dones, infos)``. All environments share the
        horizon, so ``dones`` is all-true on the final step; the world then
        resets and the returned observations start the next episode, with the
        final observations of the finished episode in ``info["terminal_obs"]``.
        """
        cfg = self.cfg
        n = self.num_envs
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(n, self.action_dim), -1.0, 1.0)
        start = self._tcp_cmd.copy()
        target = np.clip(start + a * cfg.max_translation, np.asarray(cfg.workspace_min), np.asarray(cfg.workspace_max))
        ik_errors = np.zeros(n)
        try:
            for k in range(cfg.action_repeat):
                frac = (k + 1) / cfg.action_repeat
                self._command(start + frac * (target - start))
                ik_errors = np.maximum(ik_errors, self._last_ik_error)
                self._sim_step()
            self._check_world()
        except RuntimeError as exc:
            # The whole world failed. Recover the snapshot and report every
            # environment as a simulation error so the trainer skips them.
            obs = self.reset()
            infos = [{"sim_error": True, "error": repr(exc), "success": False, "distance": float("nan")} for _ in range(n)]
            return obs, np.zeros(n, dtype=np.float32), np.ones(n, dtype=bool), infos
        p = self.positions()
        centroids = self.marker_centroids(p)
        distance = np.linalg.norm(centroids - self.goals, axis=-1)
        success = distance < self.task.success_tolerance
        progress = (self._prev_distance - distance) / cfg.max_translation
        rewards = cfg.progress_scale * progress + cfg.success_bonus * success
        self._prev_distance = distance
        self._episode_step += 1
        done = self._episode_step >= cfg.horizon
        tracking = np.linalg.norm(self.tcp_measured() - self._tcp_cmd, axis=-1)
        obs = self.observation(p)
        infos = [
            {
                "success": bool(success[i]),
                "distance": float(distance[i]),
                "tracking_error": float(tracking[i]),
                "ik_error": float(ik_errors[i]),
                "episode_step": int(self._episode_step),
                "time_limit": bool(done),
            }
            for i in range(n)
        ]
        dones = np.full(n, done, dtype=bool)
        if done:
            for i in range(n):
                infos[i]["terminal_obs"] = obs[i]
            obs = self.reset()
        return obs, rewards.astype(np.float32), dones, infos

    def observation(self, positions: np.ndarray | None = None) -> np.ndarray:
        p = self.positions() if positions is None else positions
        tcps = self.tcp_measured()
        out = np.empty((self.num_envs, self.spec.dim), dtype=np.float32)
        for i in range(self.num_envs):
            rel = p[i, self._obs_subset] - tcps[i][None, :]
            out[i] = self.spec.pack(rel, self._obs_marker_mask, self.goals[i] - tcps[i], tcps[i], self._pickers[i]["active"])
        return out

    def states(self) -> list[dict]:
        p = self.positions()
        tcps = self.tcp_measured()
        centroids = self.marker_centroids(p)
        qpos = self.robot.get_qpos().detach().cpu().numpy().reshape(self.num_envs, -1)
        return [
            {
                "positions": p[i],
                "tcp": tcps[i],
                "tcp_cmd": self._tcp_cmd[i].copy(),
                "goal": self.goals[i].copy(),
                "marker_centroid": centroids[i],
                "qpos": qpos[i].copy(),
                "frame": int(self._world.frame()),
            }
            for i in range(self.num_envs)
        ]

    def describe(self) -> dict:
        """Static description recorded in checkpoints and trajectory files."""
        return {
            "config": self.cfg.to_dict(),
            "num_envs": self.num_envs,
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
    # Viewer drawing (only used with ``show_viewer``; draws environment 0)
    # ------------------------------------------------------------------
    def _draw(self) -> None:
        for obj in self._debug_objects:
            self.scene.clear_debug_object(obj)
        self._debug_objects = []
        p = self.positions()[0]
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
        if np.any(self.goals[0]):
            self._debug_objects.append(self.scene.draw_debug_sphere(self.goals[0], radius=0.012, color=(0.2, 0.85, 0.3, 1)))
            self._debug_objects.append(
                self.scene.draw_debug_sphere(self.marker_centroids()[0], radius=0.008, color=(0.9, 0.2, 0.2, 1))
            )
