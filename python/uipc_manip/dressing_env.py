"""Batched Genesis + libuipc dressing environment: thread a garment sleeve over a human arm.

This is the Newton cloth-dressing teacher's environment rebuilt on the IPC
solver. Each of the ``num_envs`` slots holds one pre-worn (garment, human)
cell from the Newton bake cache; the human's right-arm mesh is a fixed rigid
collider in Genesis, the garment is a native libuipc shell in its own IPC
subscene, and a small patch of cuff vertices is held by a soft position
constraint that follows the 6-D gripper action, translation and rotation,
exactly as Newton's kinematic cuff grasp does. One libuipc world solves every
slot together.

The MDP follows the Wang RSS 2023 ``pointcloud_3`` preset that the Newton
``--fmvp-pretrain-defaults`` launcher reproduces: 900 decisions at 60 Hz, a
0.15 m/s end-effector speed cap split per axis, 5 degrees of rotation per
step with the x-rotation zeroed, a no-move collision shell of 12 mm around the
arm, the line-triangle progress reward, a visible dual-camera point cloud with
an explicit tool point, and no termination before the time limit.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .dressing_assets import DressingCache, DressingCacheConfig, DressingCell, erode_arm_mesh, write_obj
from .dressing_obs import DressingObsConfig, DressingObservationBuilder
from .dressing_reward import WangRewardConfig, opening_threaded, wang_progress
from .genesis_env import ViewerClosed, _ensure_genesis
from .obs import FEATURE_DIM, FLAG_DEFORMABLE, FLAG_MARKER, ObsSpec

DEFAULT_GARMENTS = ("tshirt_26", "tshirt_392", "tshirt_68")
"""Garments trained by default. The hospital gown has 10,437 vertices and the Newton notes record it
as unusable at the reference solver budget, so it is opt-in."""


@dataclass
class DressingConfig:
    human: int = 0
    garments: tuple[str, ...] = DEFAULT_GARMENTS
    horizon: int = 900
    dt: float = 1.0 / 60.0
    action_repeat: int = 1
    max_ee_speed_m_s: float = 0.15
    max_rotation: float = 0.08726646
    clip_rotation_to_yz: bool = True
    no_move_collision_threshold: float = 0.012
    point_budget: int = 768
    anchor_count: int = 12
    constraint_strength: float = 100.0
    """Soft position constraint strength of the anchored cuff vertices. Newton's FMVP preset pins
    its 12 picker-patch particles kinematically; a weak hold (3) let the 0.5 kg/m^2 garment fall."""
    arm_erosion_m: float = 0.006
    friction: float = 0.3
    contact_resistance: float = 1e7
    d_hat: float = 0.001
    newton_tolerance: float = 0.1
    newton_translation_tolerance: float = 1.0
    linesearch_iterations: int = 8
    linear_system_tolerance: float = 1e-3
    cloth_model: str = "slbw"
    """``slbw`` is the strain-limiting Baraff-Witkin shell the Genesis sim2sim path validated;
    ``neohookean`` scales membrane stiffness with the thin radius and stretched the garment by
    tens of centimetres under its own weight."""
    cloth_youngs: float = 6e4
    cloth_poisson: float = 0.49
    cloth_density: float = 3333.0
    """Volumetric density giving the Newton teacher's 0.5 kg/m^2 areal density at the thin radius.
    The heavier cloth is also far better conditioned for the linear solve than the 200 kg/m^3 of the
    Genesis sim2sim student runs."""
    fem_preconditioner: str = "mas"
    """libuipc FEM preconditioner. Multilevel additive Schwarz cut the garment step from 3.9 s to
    0.67 s against block-Jacobi in the single-cell probe; the PCG on this thin, soft cloth is the
    whole cost otherwise."""
    cloth_thickness: float = 1.5e-4
    """One-sided collision radius. The pre-worn garments hang with layers pressed together, and
    libuipc's build-time distance check rejects non-adjacent surfaces closer than the summed
    radii plus ``d_hat``; the thin radius the Genesis sim2sim path validated keeps them legal."""
    cloth_bending_stiffness: float = 10.0
    cloth_strain_rate: float = 100.0
    """Baraff-Witkin over-stretch amplification of the strain-limiting shell; lower lets the opening
    stretch further over the hand."""
    sanity_check: bool = True
    """Keep libuipc's build-time intersection and distance checks. Disabling them lets a cached
    state with residual self-contact build, at the cost of starting from an unverified state."""
    settle_steps: int = 30
    hold_steps: int = 2
    seed: int = 0
    augment_obs: bool = True
    show_viewer: bool = False
    logging_level: str = "warning"
    workspace: str = "output/uipc_manip/dressing_assets"
    obs: DressingObsConfig = field(default_factory=DressingObsConfig)
    reward: WangRewardConfig = field(default_factory=WangRewardConfig)
    cache: DressingCacheConfig = field(default_factory=DressingCacheConfig)

    @property
    def max_translation(self) -> float:
        """Per-axis metres per decision from the resultant speed cap (FMVP bijective scaling)."""
        return float(self.max_ee_speed_m_s) * float(self.dt) * int(self.action_repeat) / math.sqrt(3.0)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cache"] = {k: str(v) for k, v in data["cache"].items()}
        data["max_translation"] = self.max_translation
        return data


_PATCHED_BUILDER = None


def _install_preconditioner_patch(preconditioner: str) -> None:
    """Route ``linear_system.fem_preconditioner`` into the coupler's IPC scene config.

    Genesis 1.1.2 does not expose the preconditioner in ``IPCCouplerOptions``, and the
    coupler creates its libuipc scene from ``build_ipc_scene_config`` inside ``Scene``
    construction, so the value is injected by wrapping that builder in the coupler module.
    """
    global _PATCHED_BUILDER
    import genesis.engine.couplers.ipc_coupler.coupler as coupler_module

    if _PATCHED_BUILDER is None:
        _PATCHED_BUILDER = coupler_module.build_ipc_scene_config

    original = _PATCHED_BUILDER

    def build_with_preconditioner(options, sim_options):
        config = original(options, sim_options)
        config["linear_system"]["fem_preconditioner"] = str(preconditioner)
        return config

    coupler_module.build_ipc_scene_config = build_with_preconditioner


def _rodrigues(offsets: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Rotate cuff offsets by an axis-angle increment, preserving their lengths (Newton ``set_gripper_target``)."""
    angle = float(np.linalg.norm(rotation))
    if angle < 1e-12:
        return offsets
    axis = rotation / angle
    c, s = math.cos(angle), math.sin(angle)
    rotated = offsets * c + np.cross(axis[None, :], offsets) * s + axis[None, :] * (offsets @ axis)[:, None] * (1.0 - c)
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    rnorms = np.linalg.norm(rotated, axis=1, keepdims=True)
    return np.where(rnorms > 1e-12, rotated * norms / np.maximum(rnorms, 1e-12), rotated)


class GenesisIPCDressingEnv:
    """Batched sleeve-threading task; one libuipc world for every slot."""

    action_dim = 6
    metric_keys = ("upperarm_ratio", "forearm_ratio", "threaded", "task_reward")

    def __init__(self, cfg: DressingConfig, num_envs: int = 1) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be at least 1")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.spec = ObsSpec(cfg.point_budget)
        self.obs_dim = self.spec.dim
        self.rngs = [np.random.default_rng(cfg.seed * 1000 + i) for i in range(self.num_envs)]
        self._gs = _ensure_genesis(cfg.logging_level)
        self._torch = __import__("torch")
        self._uipc = __import__("uipc")
        self._device = self._gs.device
        self._debug_objects: list = []
        self.cache = DressingCache(cfg.cache)
        garments = [g for g in cfg.garments if (g, int(cfg.human)) in self.cache.cells]
        if not garments:
            raise ValueError(f"No cached cells for human {cfg.human} among {cfg.garments}; cache has {self.cache.cells}")
        self.cells: list[DressingCell] = [self.cache.load(garments[i % len(garments)], cfg.human) for i in range(self.num_envs)]
        self._obs_builder = DressingObservationBuilder(cfg.obs, self._device)
        self._episode_step = 0
        self._build_scene()
        self._prepare_start()
        self.descriptions = [self.describe(i) for i in range(self.num_envs)]

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        gs = self._gs
        cfg = self.cfg
        cell0 = self.cells[0]
        arm_eroded = erode_arm_mesh(cell0.arm_points, cell0.arm_faces, cfg.arm_erosion_m, cell0.finger, cell0.shoulder)
        arm_path = write_obj(Path(cfg.workspace) / f"arm_human_{cfg.human}_eroded_{int(round(cfg.arm_erosion_m * 1000))}mm.obj", arm_eroded, cell0.arm_faces)
        self.arm_collider_path = str(arm_path)
        _install_preconditioner_patch(cfg.fem_preconditioner)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.dt, substeps=1, gravity=(0.0, 0.0, -9.8)),
            coupler_options=gs.options.IPCCouplerOptions(
                contact_enable=True,
                contact_d_hat=cfg.d_hat,
                contact_resistance=cfg.contact_resistance,
                contact_friction_enable=True,
                enable_rigid_rigid_contact=False,
                n_linesearch_iterations=cfg.linesearch_iterations,
                linesearch_report_energy=False,
                newton_tolerance=cfg.newton_tolerance,
                newton_translation_tolerance=cfg.newton_translation_tolerance,
                newton_semi_implicit_enable=False,
                linear_system_tolerance=cfg.linear_system_tolerance,
                sanity_check_enable=bool(cfg.sanity_check),
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(960, 800),
                camera_pos=tuple(float(x) for x in (cell0.elbow + np.array([0.9, -1.1, 0.5]))),
                camera_lookat=tuple(float(x) for x in cell0.elbow),
                camera_fov=40,
                refresh_rate=60,
            ),
            show_viewer=cfg.show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only"))
        # The arm is a native libuipc fixed affine body rather than a Genesis mesh entity:
        # Genesis re-tessellates imported meshes (1307 vertices became 4885 with duplicates),
        # and the duplicated vertices produced NaN distances in the IPC trajectory filter.
        # The native body is the exact cached ``right_arm_faces`` collider, eroded.
        self.arm_vertices = arm_eroded
        self.arm_faces = cell0.arm_faces
        coupler = self.scene.sim.coupler
        self.coupler = coupler
        coupler._ipc_contact_tabular.default_model(cfg.friction, cfg.contact_resistance)
        self.slots: list = []
        self._pickers: list[dict] = []
        uipc = self._uipc
        from uipc.constitution import (
            AffineBodyConstitution,
            DiscreteShellBending,
            ElasticModuli2D,
            NeoHookeanShell,
            SoftPositionConstraint,
            StrainLimitingBaraffWitkinShell,
        )
        from uipc.geometry import label_surface
        from uipc.geometry import trimesh as ipc_trimesh

        original_add_objects = coupler._add_objects_to_ipc
        builtin = uipc.builtin

        def add_objects_with_garments() -> None:
            original_add_objects()
            for env_idx, cell in enumerate(self.cells):
                arm = ipc_trimesh(self.arm_vertices, self.arm_faces)
                label_surface(arm)
                # Open surface: use the explicit mass overload, the body is fixed anyway.
                AffineBodyConstitution().apply_to(arm, 1e8, np.eye(12), 1.0)
                uipc.view(arm.instances().find(builtin.is_fixed))[:] = 1
                coupler._ipc_contact_tabular.default_element().apply_to(arm)
                coupler._ipc_subscenes[env_idx].apply_to(arm)
                coupler._ipc_objects.create(f"arm_human_{cell.human}_{env_idx}").geometries().create(arm)
                mesh = ipc_trimesh(cell.cloth, cell.faces)
                label_surface(mesh)
                moduli = ElasticModuli2D.youngs_poisson(cfg.cloth_youngs, cfg.cloth_poisson)
                if cfg.cloth_model == "slbw":
                    StrainLimitingBaraffWitkinShell().apply_to(
                        mesh, moduli, cfg.cloth_density, cfg.cloth_thickness, cfg.cloth_strain_rate
                    )
                elif cfg.cloth_model == "neohookean":
                    NeoHookeanShell().apply_to(mesh, moduli, cfg.cloth_density, cfg.cloth_thickness)
                else:
                    raise ValueError(f"Unknown cloth_model {cfg.cloth_model!r}")
                DiscreteShellBending().apply_to(mesh, cfg.cloth_bending_stiffness)
                SoftPositionConstraint().apply_to(mesh, cfg.constraint_strength)
                coupler._ipc_contact_tabular.default_element().apply_to(mesh)
                coupler._ipc_subscenes[env_idx].apply_to(mesh)
                ipc_object = coupler._ipc_objects.create(f"garment_{cell.garment}_{env_idx}")
                self.slots.append(ipc_object.geometries().create(mesh)[0])
                anchor_idx = cell.anchor_indices(cfg.anchor_count)
                picker = {"active": False, "anchor_idx": anchor_idx, "targets": cell.cloth[anchor_idx].copy()}
                self._pickers.append(picker)

                def animate(info, picker=picker):
                    geo = info.geo_slots()[0].geometry()
                    flags = uipc.view(geo.vertices().find(uipc.builtin.is_constrained)).reshape(-1)
                    flags[:] = 0
                    if picker["active"]:
                        idx = picker["anchor_idx"]
                        flags[idx] = 1
                        aim = uipc.view(geo.vertices().find(uipc.builtin.aim_position)).reshape(-1, 3)
                        aim[idx] = picker["targets"]

                coupler._ipc_animator.insert(ipc_object, animate)

        coupler._add_objects_to_ipc = add_objects_with_garments
        t0 = time.time()
        self.scene.build(n_envs=self.num_envs)
        self.build_seconds = time.time() - t0
        if not coupler._ipc_world.is_valid():
            raise RuntimeError(
                "IPC world is invalid after build: a pre-worn garment intersects the arm collider or itself. "
                "Increase arm_erosion_m or drop the offending cell."
            )
        self._world = coupler._ipc_world

    def _prepare_start(self) -> None:
        cfg = self.cfg
        self._anchor = np.stack([cell.picker_pos.copy() for cell in self.cells])
        self._offsets = [cell.cloth[p["anchor_idx"]] - cell.picker_pos[None, :] for cell, p in zip(self.cells, self._pickers, strict=True)]
        self._initial_offsets = [o.copy() for o in self._offsets]
        for picker in self._pickers:
            picker["active"] = True
        self._update_targets()
        before = self.positions()
        for _ in range(cfg.settle_steps):
            self._sim_step()
        self._check_world()
        after = self.positions()
        self.settle_displacement = float(max(np.linalg.norm(a - b, axis=1).max() for a, b in zip(after, before, strict=True)))
        self._snapshot_frame = int(self._world.frame())
        if not self._world.dump():
            raise RuntimeError("Failed to dump the settled IPC snapshot")
        self._snapshot_positions = [p.copy() for p in after]

    # ------------------------------------------------------------------
    # Low level helpers
    # ------------------------------------------------------------------
    def _update_targets(self) -> None:
        for i, picker in enumerate(self._pickers):
            picker["targets"] = self._anchor[i][None, :] + self._offsets[i]

    def _sim_step(self) -> None:
        try:
            self.scene.step()
        except Exception as exc:
            if self.cfg.show_viewer and "Viewer closed" in str(exc):
                raise ViewerClosed("viewer window closed") from exc
            raise
        if self.cfg.show_viewer:
            self._draw()

    def _check_world(self) -> None:
        if not self._world.is_valid():
            raise RuntimeError(f"Invalid IPC world at frame {self._world.frame()}")
        for p in self.positions():
            if not np.isfinite(p).all() or np.abs(p).max() > 5.0:
                raise RuntimeError("Garment state diverged")

    def positions(self) -> list[np.ndarray]:
        view = self._uipc.view
        return [np.asarray(view(slot.geometry().positions())).reshape(-1, 3).copy() for slot in self.slots]

    def _progress(self, positions: list[np.ndarray]):
        out = []
        for cell, p in zip(self.cells, positions, strict=True):
            out.append(
                wang_progress(
                    p,
                    polygon_idx=cell.opening_idx,
                    triangle_idx=cell.polygon_triangles(),
                    cuff_idx=cell.cuff_idx,
                    finger=cell.finger,
                    elbow=cell.elbow,
                    shoulder=cell.shoulder,
                    human_points=cell.human_points,
                    cfg=self.cfg.reward,
                )
            )
        return out

    # ------------------------------------------------------------------
    # RL interface
    # ------------------------------------------------------------------
    def reset(self, seeds: list[int | None] | None = None) -> np.ndarray:
        if seeds is not None:
            for i, seed in enumerate(seeds):
                if seed is not None:
                    self.rngs[i] = np.random.default_rng(int(seed))
        self.scene.reset()
        if not self._world.recover(self._snapshot_frame):
            raise RuntimeError(f"Failed to recover IPC frame {self._snapshot_frame}")
        self._world.retrieve()
        self._anchor = np.stack([cell.picker_pos.copy() for cell in self.cells])
        self._offsets = [o.copy() for o in self._initial_offsets]
        self._update_targets()
        restored = self.positions()
        self.reset_restore_error = float(max(np.abs(a - b).max() for a, b in zip(restored, self._snapshot_positions, strict=True)))
        for _ in range(self.cfg.hold_steps):
            self._sim_step()
        self._check_world()
        self._episode_step = 0
        if getattr(self, "_heuristic", None) is not None:
            self._heuristic.reset()
        return self.observation(self.positions())

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        cfg = self.cfg
        n = self.num_envs
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(n, self.action_dim), -1.0, 1.0)
        translation = a[:, :3] * cfg.max_translation
        rotation = a[:, 3:] * cfg.max_rotation
        if cfg.clip_rotation_to_yz:
            rotation[:, 0] = 0.0
        try:
            for _ in range(cfg.action_repeat):
                for i, cell in enumerate(self.cells):
                    self._offsets[i] = _rodrigues(self._offsets[i], rotation[i] / cfg.action_repeat)
                    candidate = self._anchor[i] + translation[i] / cfg.action_repeat
                    # PyFlex no-move collision: a step that would put the anchor inside the shell is dropped.
                    if np.min(np.linalg.norm(cell.arm_points - candidate[None, :], axis=1)) >= cfg.no_move_collision_threshold:
                        self._anchor[i] = candidate
                self._update_targets()
                self._sim_step()
            self._check_world()
        except ViewerClosed:
            raise
        except RuntimeError as exc:
            obs = self.reset()
            infos = [{"sim_error": True, "error": repr(exc), "success": False, "distance": float("nan")} for _ in range(n)]
            return obs, np.zeros(n, dtype=np.float32), np.ones(n, dtype=bool), infos
        positions = self.positions()
        progress = self._progress(positions)
        self._episode_step += 1
        done = self._episode_step >= cfg.horizon
        rewards = np.array([pr.reward for pr in progress], dtype=np.float32)
        obs = self.observation(positions)
        infos = []
        for i, (cell, pr, p) in enumerate(zip(self.cells, progress, positions, strict=True)):
            threaded, _ = opening_threaded(p, cell.opening_idx, cell.finger, cell.shoulder)
            infos.append(
                {
                    "success": bool(pr.upperarm_ratio >= cfg.reward.success_upperarm_ratio),
                    "distance": float(1.0 - pr.upperarm_ratio),
                    "upperarm_ratio": float(pr.upperarm_ratio),
                    "forearm_ratio": float(pr.forearm_ratio),
                    "threaded": float(threaded),
                    "task_reward": float(pr.task_reward),
                    "on_forearm": bool(pr.on_forearm),
                    "on_upperarm": bool(pr.on_upperarm),
                    "collision": float(pr.collision),
                    "tracking_error": 0.0,
                    "garment": cell.garment,
                    "episode_step": int(self._episode_step),
                    "time_limit": bool(done),
                }
            )
        dones = np.full(n, done, dtype=bool)
        if done:
            for i in range(n):
                infos[i]["terminal_obs"] = obs[i]
            obs = self.reset()
        return obs, rewards, dones, infos

    def observation(self, positions: list[np.ndarray] | None = None) -> np.ndarray:
        positions = self.positions() if positions is None else positions
        out = np.empty((self.num_envs, self.spec.dim), dtype=np.float32)
        budget = self.spec.deformable_budget
        for i, (cell, p) in enumerate(zip(self.cells, positions, strict=True)):
            arm, cloth = self._obs_builder.visible_points(cell.arm_points, p, cell.finger, cell.shoulder, self.rngs[i], self.cfg.augment_obs)
            if arm.shape[0] + cloth.shape[0] > budget:
                cloth = cloth[: max(0, budget - arm.shape[0])]
                arm = arm[:budget]
            tool = self._anchor[i]
            pts = np.concatenate([arm, cloth], axis=0) - tool[None, :]
            flags = np.zeros((pts.shape[0], FEATURE_DIM), dtype=np.float32)
            flags[: arm.shape[0], FLAG_MARKER] = 1.0
            flags[arm.shape[0] :, FLAG_DEFORMABLE] = 1.0
            out[i] = self.spec.pack_labeled(pts, flags, cell.shoulder - tool, tool, attached=True)
        return out

    def scripted_actions(self) -> np.ndarray:
        """The Newton seven-stage dressing expert (the reachability baseline)."""
        if getattr(self, "_heuristic", None) is None:
            from .dressing_heuristic import HeuristicDressingPolicy

            self._heuristic = HeuristicDressingPolicy(self)
        return self._heuristic.actions()

    def scripted_stage_names(self) -> list[str]:
        heuristic = getattr(self, "_heuristic", None)
        return heuristic.stage_names() if heuristic is not None else ["none"] * self.num_envs

    def states(self) -> list[dict]:
        positions = self.positions()
        return [
            {
                "positions": p,
                "tcp": self._anchor[i].copy(),
                "tcp_cmd": self._anchor[i].copy(),
                "goal": cell.shoulder.copy(),
                "marker_centroid": p[cell.opening_idx].mean(axis=0),
                "qpos": np.zeros(0),
                "frame": int(self._world.frame()),
            }
            for i, (cell, p) in enumerate(zip(self.cells, positions, strict=True))
        ]

    def describe(self, slot: int = 0) -> dict:
        cell = self.cells[slot]
        return {
            "config": self.cfg.to_dict(),
            "num_envs": self.num_envs,
            "task": "dressing",
            "task_description": "Thread the sleeve opening of a pre-worn garment along the right arm to the shoulder.",
            "deformable": "cloth",
            "cell": cell.name,
            "garment": cell.garment,
            "human": cell.human,
            "vertex_count": int(cell.cloth.shape[0]),
            "anchor_vertices": self._pickers[slot]["anchor_idx"].tolist(),
            "obs_dim": self.spec.dim,
            "action_dim": self.action_dim,
            "faces": cell.faces.tolist(),
            "edges": [],
            "radius": float(self.cfg.cloth_thickness),
            "arm_collider": self.arm_collider_path,
            "build_seconds": float(self.build_seconds),
            "settle_displacement_m": float(self.settle_displacement),
            "snapshot_tracking_error_m": 0.0,
        }

    def close(self) -> None:
        self.scene = None

    def _draw(self) -> None:
        for obj in self._debug_objects:
            self.scene.clear_debug_object(obj)
        self._debug_objects = []
        import trimesh

        p = self.positions()[0]
        cell = self.cells[0]
        visual = trimesh.Trimesh(p, cell.faces, process=False)
        visual.visual.vertex_colors = np.tile([115, 158, 242, 255], (len(p), 1))
        drawing = self.scene.draw_debug_mesh(visual)
        for primitive in drawing.primitives:
            primitive.material.doubleSided = True
        self._debug_objects.append(drawing)
        arm_visual = trimesh.Trimesh(self.arm_vertices, self.arm_faces, process=False)
        arm_visual.visual.vertex_colors = np.tile([222, 184, 150, 255], (len(self.arm_vertices), 1))
        self._debug_objects.append(self.scene.draw_debug_mesh(arm_visual))
        self._debug_objects.append(self.scene.draw_debug_sphere(self._anchor[0], radius=0.012, color=(0.2, 0.2, 0.2, 1)))
        self._debug_objects.append(self.scene.draw_debug_sphere(cell.shoulder, radius=0.012, color=(0.2, 0.85, 0.3, 1)))
