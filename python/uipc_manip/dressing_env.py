"""Batched Genesis + libuipc dressing environment: thread a garment sleeve over a human arm.

This is the Newton cloth-dressing teacher's environment rebuilt on the IPC
solver. Each of the ``num_envs`` slots holds one pre-worn (garment, human)
cell from the Newton bake cache; the human's right-arm mesh is a fixed rigid
collider in Genesis, the garment is a native libuipc shell in its own IPC
subscene, and a small patch of cuff vertices is held by a soft position
constraint driven by the 6-D gripper action, translation and rotation.
Unlike Newton's kinematic cuff pin, this penalty participates in the IPC
contact solve and has measurable compliance. One libuipc world solves every
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
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .dressing_assets import DressingCache, DressingCacheConfig, DressingCell, erode_arm_mesh, write_obj
from .dressing_body import pose_region
from .dressing_obs import BatchedDressingObservationBuilder, DressingObsConfig, DressingObservationBuilder, sample_segmented_cloud
from .dressing_privileged import PRIVILEGED_DIM, privileged_state
from .dressing_reward import early_turn, WangRewardConfig, opening_threaded, wang_progress
from .genesis_env import ViewerClosed, _ensure_genesis
from .obs import FEATURE_DIM, FLAG_DEFORMABLE, FLAG_MARKER, ObsSpec

DEFAULT_GARMENTS = ("tshirt_26", "tshirt_392")
"""Cached garments that build for every cached body. Six of the twenty-three baked cells
carry residual self-intersections from the bake and libuipc refuses them; ``tshirt_68`` is
one such cell at bodies 0 and 2 although the cache advertises it, so a default that named
it crashed at construction. The live cell path has no such failure: it spawns the garment
outside the arm, so :class:`~uipc_manip.dressing_live.LiveCellFactory` offers all five."""
"""Garments trained by default. The hospital gown (10,437 vertices) stays opt-in: the Newton notes
record it as unusable at the reference solver budget, and on live cells it is dressable only
because it starts in its baked hang (``LiveCellConfig.hang_as_baked_garments``); see the
dressing-correctness record."""


@dataclass
class DressingConfig:
    human: int = 0
    garments: tuple[str, ...] = DEFAULT_GARMENTS
    cells: tuple[tuple[str, int], ...] = ()
    """The (garment, body) bound to each slot, in slot order. Empty keeps the legacy
    shorthand of cycling ``garments`` over the single ``human``. A slot's cell is fixed
    for the life of the world because ``reset`` restores one settled snapshot."""
    ground_plane: bool = False
    """Add a floor to the IPC scene. The arm is fixed in the air and the garment is
    carried by the tool, so the floor is never part of the task; a shirt hanging from a
    point near the hand reaches it, and libuipc then refuses the scene."""
    cell_source: str = "cache"
    """``cache`` reads the Newton bake's pre-worn states; ``live`` drapes each garment in
    libuipc and places it on a generated body, which admits any (garment, body) pair."""
    live: "LiveCellConfig" = field(default_factory=lambda: __import__("uipc_manip.dressing_live", fromlist=["LiveCellConfig"]).LiveCellConfig())
    """How live cells are baked and placed; ignored when ``cell_source`` is ``cache``."""
    horizon: int = 900
    dt: float = 1.0 / 60.0
    action_repeat: int = 1
    max_ee_speed_m_s: float = 0.15
    max_rotation: float = 0.08726646
    clip_rotation_to_yz: bool = True
    no_move_collision_threshold: float = 0.012
    point_budget: int = 768
    anchor_count: int = 48
    """Cuff vertices held by the picker: its own vertices plus the grasp-patch vertices
    nearest it. On live cells with 48 the scripted expert gets the sleeve over the hand on
    all eight tshirt_26 bodies, against one of eight with Newton's twelve, and a tenfold
    stiffer hold does not substitute for the count; see the dressing-correctness record."""
    constraint_strength: float = 1.0e4
    """Soft position constraint strength of the anchored cuff vertices. Newton's FMVP preset pins
    its 12 picker-patch particles kinematically; a weak hold (3) let the 0.5 kg/m^2 garment fall.
    The dimensionless strength multiplies vertex mass in the incremental energy;
    equivalent physical spring stiffness is strength * mass / dt**2."""
    arm_erosion_m: float = 0.006
    friction: float = 0.3
    contact_resistance: float = 1e7
    d_hat: float = 0.001
    newton_tolerance: float = 0.1
    newton_translation_tolerance: float = 1.0
    linesearch_iterations: int = 8
    linear_system_tolerance: float = 1e-2
    """Relative tolerance of the preconditioned conjugate-gradient solve. The library
    default is 1e-3; at 1e-2 a 100-decision expert run costs 330 ms per simulation step
    against 442 to 502, and reaches a forearm ratio of 0.859 against 0.866 to 0.873 with
    the same held-vertex error, which is inside the run-to-run spread."""
    linear_system_cuda_graph: int = 2
    """CUDA-graph mode of the fused PCG. 1, the library default, replays a block of
    iterations per launch; 2 runs the whole solve as one device-side conditional graph
    with no host round trip inside the loop, falling back to 1 where the driver lacks
    support. The kernels, arguments and order are identical either way, so this trades
    no accuracy: 368 ms per simulation step against 442 to 502, and 247 with the looser
    tolerance as well."""
    cloth_model: str = "slbw"
    """``slbw`` is the strain-limiting Baraff-Witkin shell the Genesis sim2sim path validated;
    ``neohookean`` scales membrane stiffness with the thin radius and stretched the garment by
    tens of centimetres under its own weight."""
    cloth_youngs: float = 6e3
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
    cloth_bending_stiffness: float = 0.1
    cloth_strain_rate: float = 100.0
    cloth_shear_ratio: float | None = 0.01
    """Shear Young's modulus as a fraction of the stretch modulus. libuipc measures stretch as
    ``E*2r/(1-nu^2)`` but shear as ``E/(2(1+nu))``, with no thickness factor, so sharing one
    modulus (``None``, the historical setting) makes shear about ``1/(2r)``, here 3,300, times
    stiffer than stretch and the fabric effectively unshearable. The drape bake measured 1/100
    as the ratio that reproduces the reference garment's hang.

    These three settings together are the default because on the 100-decision expert
    protocol they win on every axis at once against the historical 6e4 / shared / 10:
    116 to 118 ms per simulation step against 168 to 188, a forearm ratio of 0.955 to
    0.966 against 0.861 to 0.869, and a held-cuff error of 3.3 to 5.4 mm against 41 to 43."""
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
        # The live configuration nests Paths; its own to_dict renders them as strings so
        # the whole config survives json.dumps (run config, checkpoint metadata, rollouts).
        data["live"] = self.live.to_dict()
        data["cells"] = [[str(g), int(b)] for g, b in self.cells]
        data["max_translation"] = self.max_translation
        return data


_PATCHED_BUILDER = None


def _install_solver_patch(preconditioner: str, cuda_graph: int) -> None:
    """Route libuipc linear-system settings into the coupler's IPC scene config.

    Genesis 1.1.2 exposes neither the preconditioner nor the CUDA-graph mode in
    ``IPCCouplerOptions``, and the coupler creates its libuipc scene from
    ``build_ipc_scene_config`` inside ``Scene`` construction, so the values are injected
    by wrapping that builder in the coupler module.
    """
    global _PATCHED_BUILDER
    import genesis.engine.couplers.ipc_coupler.coupler as coupler_module

    if _PATCHED_BUILDER is None:
        _PATCHED_BUILDER = coupler_module.build_ipc_scene_config

    original = _PATCHED_BUILDER

    def build_with_solver_settings(options, sim_options):
        config = original(options, sim_options)
        config["linear_system"]["fem_preconditioner"] = str(preconditioner)
        config["linear_system"]["use_cuda_graph"] = int(cuda_graph)
        return config

    coupler_module.build_ipc_scene_config = build_with_solver_settings


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
    grasp_tracking_tolerance_m = 0.02
    privileged_dim = PRIVILEGED_DIM

    def __init__(self, cfg: DressingConfig, num_envs: int = 1, cell_factory=None) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be at least 1")
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.spec = ObsSpec(cfg.point_budget)
        self.obs_dim = self.spec.dim
        self.rngs = [np.random.default_rng(cfg.seed * 1000 + i) for i in range(self.num_envs)]
        # Genesis must come up before this process does any CUDA matrix work: once cuBLAS is
        # initialised, Quadrants cannot materialise its runtime and ``gs.init`` dies on a
        # device-side assert reading "Out of CUDA pre-allocated memory", which a larger
        # Quadrants pool does not cure. Generating a body runs SMPL-X on the GPU, so live cells
        # are built only after this line and pre-flight numbers come from :meth:`clearances`.
        torch_cuda_first = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
        try:
            self._gs = _ensure_genesis(cfg.logging_level)
        except RuntimeError as exc:
            if not torch_cuda_first:
                raise
            raise RuntimeError(
                "Genesis failed to initialise after torch had already used CUDA in this process. If that "
                "included matrix work, such as generating a body with LiveCellFactory or generate_body, "
                "cuBLAS came up before Quadrants and its runtime cannot start: construct "
                "GenesisIPCDressingEnv first and read spawn clearances from env.clearances()."
            ) from exc
        self._torch = __import__("torch")
        self._uipc = __import__("uipc")
        self._device = self._gs.device
        self._debug_objects: list = []
        self.cache = DressingCache(cfg.cache) if cfg.cell_source == "cache" else None
        plan = list(cfg.cells)
        if not plan:
            if cfg.cell_source == "cache":
                garments = [g for g in cfg.garments if (g, int(cfg.human)) in self.cache.cells]
                if not garments:
                    raise ValueError(f"No cached cells for human {cfg.human} among {cfg.garments}; cache has {self.cache.cells}")
            else:
                garments = list(cfg.garments)
            plan = [(garments[i % len(garments)], int(cfg.human)) for i in range(self.num_envs)]
        if len(plan) != self.num_envs:
            raise ValueError(f"{len(plan)} cells for {self.num_envs} slots; every slot needs its own cell")
        self.cell_factory = None
        if cfg.cell_source == "live":
            from .dressing_live import LiveCellFactory

            # A factory shared across worlds keeps its drapes, bodies and clearances, so a rebuilt
            # world bakes, generates and places only cells no earlier world held.
            if cell_factory is not None and cell_factory.cfg != cfg.live:
                raise ValueError("The shared cell factory was made for other live-cell settings than this world's")
            self.cell_factory = cell_factory or LiveCellFactory(cfg.live)
            self.cells: list[DressingCell] = [self.cell_factory.build(g, b) for g, b in plan]
        elif cfg.cell_source == "cache":
            self.cells = [self.cache.load(g, b) for g, b in plan]
        else:
            raise ValueError(f"Unknown cell_source {cfg.cell_source!r}; use 'cache' or 'live'")
        self.cell_labels = [cell.name for cell in self.cells]
        self._obs_builder = DressingObservationBuilder(cfg.obs, self._device)
        self._batched_obs = BatchedDressingObservationBuilder(cfg.obs, self._device)
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
        # One collider per slot: the slots may hold different bodies, and the IPC object is
        # named after the cell's own body, so a shared mesh would silently dress every body
        # with the first one's arm.
        # The bake's pre-worn states interpenetrate and need eroding; a live cell spawns
        # the garment a clearance step outside the fingertip and must not be eroded.
        erosion_m = cfg.arm_erosion_m if cfg.cell_source == "cache" else 0.0
        erosion = int(round(erosion_m * 1000))
        self.arm_meshes: list[tuple[np.ndarray, np.ndarray]] = []
        arm_paths: list[str] = []
        for cell in self.cells:
            eroded = erode_arm_mesh(cell.arm_points, cell.arm_faces, erosion_m, cell.finger, cell.shoulder)
            self.arm_meshes.append((eroded, cell.arm_faces))
            arm_paths.append(
                str(write_obj(Path(cfg.workspace) / f"arm_human_{cell.human}_eroded_{erosion}mm.obj", eroded, cell.arm_faces))
            )
        self.arm_collider_paths = arm_paths
        self.arm_collider_path = arm_paths[0]
        _install_solver_patch(cfg.fem_preconditioner, cfg.linear_system_cuda_graph)
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
        if cfg.ground_plane:
            self.scene.add_entity(gs.morphs.Plane(), material=gs.materials.Rigid(coup_type="ipc_only"))
        # The arm is a native libuipc fixed affine body rather than a Genesis mesh entity:
        # Genesis re-tessellates imported meshes (1307 vertices became 4885 with duplicates),
        # and the duplicated vertices produced NaN distances in the IPC trajectory filter.
        # The native body is the exact cached ``right_arm_faces`` collider, eroded.
        # These two are the viewer's and the trajectory export's slot-0 aliases.
        self.arm_vertices, self.arm_faces = self.arm_meshes[0]
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
                arm = ipc_trimesh(*self.arm_meshes[env_idx])
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
                if cfg.cloth_model == "slbw" and cfg.cloth_shear_ratio is not None:
                    shear = ElasticModuli2D.youngs_poisson(cfg.cloth_youngs * float(cfg.cloth_shear_ratio), cfg.cloth_poisson)
                    StrainLimitingBaraffWitkinShell().apply_to(
                        mesh, moduli, shear, cfg.cloth_density, cfg.cloth_thickness, cfg.cloth_strain_rate
                    )
                elif cfg.cloth_model == "slbw":
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
                f"Cells in this world: {[cell.name for cell in self.cells]}. Six of the baked cells carry "
                "residual self-intersections; raise arm_erosion_m, drop the offending cell, or build the "
                "cells live, which spawns each garment clear of the arm."
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
        self.snapshot_tracking_error = float(max(self._tracking_error(i, after) for i in range(self.num_envs)))
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

    def _tracking_error(self, i: int, positions: list[np.ndarray]) -> float:
        """Largest distance between a held cuff vertex and its commanded position [m]."""
        picker = self._pickers[i]
        return float(np.linalg.norm(positions[i][picker["anchor_idx"]] - picker["targets"], axis=1).max())

    def _actual_tcp(self, i: int, positions: list[np.ndarray]) -> np.ndarray:
        """Least-squares tool translation implied by the held patch at its commanded orientation."""
        return (positions[i][self._pickers[i]["anchor_idx"]] - self._offsets[i]).mean(axis=0)

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

    def privileged(self) -> np.ndarray:
        """The simulator state behind the current observation, one row per slot, for an asymmetric critic."""
        return self._privileged.copy()

    def _privileged_state(self, positions: list[np.ndarray], progress) -> np.ndarray:
        return np.stack(
            [
                privileged_state(
                    p,
                    opening_idx=cell.opening_idx,
                    finger=cell.finger,
                    elbow=cell.elbow,
                    shoulder=cell.shoulder,
                    arm_points=cell.arm_points,
                    tool=self._anchor[i],
                    offsets=self._offsets[i],
                    initial_offsets=self._initial_offsets[i],
                    progress=pr,
                    tracking_error=self._tracking_error(i, positions),
                    garment=cell.garment,
                )
                for i, (cell, p, pr) in enumerate(zip(self.cells, positions, progress, strict=True))
            ]
        )

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
        positions = self.positions()
        self._privileged = self._privileged_state(positions, self._progress(positions))
        return self.observation(positions)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        cfg = self.cfg
        n = self.num_envs
        a = np.clip(np.asarray(actions, dtype=np.float64).reshape(n, self.action_dim), -1.0, 1.0)
        translation = a[:, :3] * cfg.max_translation
        rotation = a[:, 3:] * cfg.max_rotation
        if cfg.clip_rotation_to_yz:
            rotation[:, 0] = 0.0
        tracking_max = np.zeros(n, dtype=np.float64)
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
                substep_positions = self.positions()
                tracking_max = np.maximum(tracking_max, [self._tracking_error(i, substep_positions) for i in range(n)])
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
        self._privileged = self._privileged_state(positions, progress)
        infos = []
        for i, (cell, pr, p) in enumerate(zip(self.cells, progress, positions, strict=True)):
            threaded, _ = opening_threaded(p, cell.opening_idx, cell.finger, cell.shoulder)
            grasp_valid = bool(tracking_max[i] <= self.grasp_tracking_tolerance_m)
            infos.append(
                {
                    "success": bool(pr.upperarm_ratio >= cfg.reward.success_upperarm_ratio),
                    "grasp_valid": grasp_valid,
                    "valid_grasp_success": bool(grasp_valid and pr.upperarm_ratio >= cfg.reward.success_upperarm_ratio),
                    "distance": float(1.0 - pr.upperarm_ratio),
                    "upperarm_ratio": float(pr.upperarm_ratio),
                    "forearm_ratio": float(pr.forearm_ratio),
                    "threaded": float(threaded),
                    "task_reward": float(pr.task_reward),
                    "on_forearm": bool(pr.on_forearm),
                    "on_upperarm": bool(pr.on_upperarm),
                    "collision": float(pr.collision),
                    "tracking_error": float(tracking_max[i]),
                    "final_tracking_error": self._tracking_error(i, positions),
                    "tool_translation_error": float(np.linalg.norm(self._actual_tcp(i, positions) - self._anchor[i])),
                    "early_turn": early_turn(self._anchor[i], cell.finger, cell.elbow, cell.shoulder),
                    "garment": cell.garment,
                    "human": int(cell.human),
                    "cell": cell.name,
                    "episode_step": int(self._episode_step),
                    "time_limit": bool(done),
                }
            )
        dones = np.full(n, done, dtype=bool)
        if done:
            for i in range(n):
                infos[i]["terminal_obs"] = obs[i]
                infos[i]["terminal_privileged"] = self._privileged[i]
            obs = self.reset()
        return obs, rewards, dones, infos

    def observation(self, positions: list[np.ndarray] | None = None) -> np.ndarray:
        positions = self.positions() if positions is None else positions
        out = np.empty((self.num_envs, self.spec.dim), dtype=np.float32)
        budget = self.spec.deformable_budget
        clouds = self._batched_obs.visible_points(
            [cell.arm_points for cell in self.cells], list(positions),
            [cell.finger for cell in self.cells], [cell.shoulder for cell in self.cells],
            self.rngs, self.cfg.augment_obs,
        )
        for i, (cell, (arm, cloth)) in enumerate(zip(self.cells, clouds, strict=True)):
            arm, cloth = sample_segmented_cloud(arm, cloth, budget, self.rngs[i])
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
                "tcp": self._actual_tcp(i, positions),
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
            "pose_region": pose_region(cell.human),
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
            "snapshot_tracking_error_m": float(self.snapshot_tracking_error),
            "grasp_tracking_tolerance_m": float(self.grasp_tracking_tolerance_m),
        }

    def clearances(self) -> list[dict[str, float]]:
        """Each slot's smallest spawn distances, garment to arm and opening to fingertip [m].

        These are the pre-flight's numbers, measured after Genesis is up. Asking a
        ``LiveCellFactory`` for them before the environment exists generates the bodies on
        the GPU first, and ``gs.init`` then fails (see ``__init__``).
        """
        if self.cell_factory is None:
            raise ValueError("Spawn clearances are measured for live cells; this world reads the Newton cache")
        return [self.cell_factory.clearances(cell.garment, cell.human) for cell in self.cells]

    def close(self) -> None:
        """Destroy the scene and release the libuipc world, so this process can build another.

        Dropping the scene alone leaves the IPC world referenced by the coupler, its objects
        and the geometry slots this environment holds. Each scene also leaves its libuipc
        workspace, holding the settled snapshot dump, under the temporary directory; only
        this scene's is removed.
        """
        scene, self.scene = self.scene, None
        if scene is None:
            return
        workspace = Path(tempfile.gettempdir()) / f"genesis_ipc_{scene.uid.full()}"
        scene.destroy()
        self.coupler = self._world = None
        self.slots, self._pickers = [], []
        shutil.rmtree(workspace, ignore_errors=True)

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
