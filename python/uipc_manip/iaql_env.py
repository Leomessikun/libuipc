"""Native direct-picker cloth_drag diagnostic, reusing task/assets and adjoint utilities.

This explicitly removes robot contacts/IK. Every decision has the same five
substeps as cloth_drag. Sensitivities use retained projected matrices and the
inertia chain; friction history derivatives are not included and must be gated
by measured finite differences. State is cloth x/v, commanded tool, and goal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import scipy.sparse.linalg

from .assets import build_cloth
from .tasks import get_task
from .physics_gradient_adjoint import system_to_matrix, tangent_pass


@dataclass
class IAQLEnvConfig:
    friction: float = 0.0
    action_repeat: int = 5
    dt: float = 0.01
    max_translation: float = 0.006
    constraint_strength: float = 1000.0
    settle_steps: int = 40
    horizon: int = 150
    distance_epsilon: float = 0.001
    action_cost: float = 0.01
    state_scale: float = 0.1
    velocity_tolerance: float = 0.001
    export_mode: str = "last_iterate"
    """Which system the backend leaves for the tangent: the last Newton iterate's projected
    matrix, or a re-assembly at the accepted state, projected (``converged``) or not
    (``converged_raw``); see LinearSystemAdjointFeature.set_export_mode."""


class IAQLClothEnv:
    action_dim = 3

    def __init__(self, workspace: Path, cfg: IAQLEnvConfig):
        import uipc
        from uipc.constitution import SoftPositionConstraint
        from uipc.geometry import ground
        from uipc.diff_sim import LinearSystemAdjointFeature
        from uipc.core import FiniteElementStateAccessorFeature

        self.uipc, self.cfg = uipc, cfg
        if cfg.action_repeat < 1 or min(cfg.dt, cfg.state_scale, cfg.max_translation) <= 0:
            raise ValueError("Positive integration/scaling parameters are required")
        workspace.mkdir(parents=True, exist_ok=True)
        uipc.Logger.set_level(uipc.Logger.Level.Warn)
        self.engine = uipc.Engine("cuda", str(workspace))
        self.world = uipc.World(self.engine)
        config = uipc.Scene.default_config()
        config["dt"] = cfg.dt
        config["gravity"] = [[0.0], [0.0], [-9.8]]
        config["newton"]["semi_implicit"]["enable"] = 0
        config["newton"]["velocity_tol"] = cfg.velocity_tolerance
        config["contact"]["d_hat"] = 0.001
        config["contact"]["friction"]["enable"] = int(cfg.friction > 0)
        self.scene = uipc.Scene(config)
        self.scene.contact_tabular().default_model(cfg.friction, 1e8)
        element = self.scene.contact_tabular().default_element()
        self.asset = build_cloth()
        self.task = get_task("cloth_drag")
        self.held = self.task.grasp_vertices(self.asset.rest)
        self.marker = self.task.marker_vertices(self.asset.rest)
        self.anchor = self.asset.rest[self.held].mean(0)
        self.offsets = self.asset.rest[self.held] - self.anchor
        self.origin = self.asset.rest.mean(0)
        self.goal = self.origin.copy()
        self.n = len(self.asset.rest)
        self.obs_dim = 6 * self.n + 6
        SoftPositionConstraint().apply_to(self.asset.mesh, cfg.constraint_strength)
        element.apply_to(self.asset.mesh)
        obj = self.scene.objects().create("iaql_cloth")
        self.slot = obj.geometries().create(self.asset.mesh)[0]
        floor = ground(0.0, np.array([0., 0., 1.]))
        element.apply_to(floor)
        self.scene.objects().create("table").geometries().create(floor)

        def animate(info):
            geo = info.geo_slots()[0].geometry()
            flags = uipc.view(geo.vertices().find(uipc.builtin.is_constrained)).reshape(-1)
            flags[:] = 0
            flags[self.held] = 1
            aim = uipc.view(geo.vertices().find(uipc.builtin.aim_position)).reshape(-1, 3)
            aim[self.held] = self.anchor + self.offsets

        self.scene.animator().insert(obj, animate)
        self.world.init(self.scene)
        self.state_accessor = self.world.features().find(FiniteElementStateAccessorFeature)
        self.state_geometry = self.state_accessor.create_geometry()
        self.state_geometry.vertices().create(uipc.builtin.velocity, np.zeros(3, dtype=np.float64))
        self.feature = self.world.features().find(LinearSystemAdjointFeature)
        if self.feature is None:
            raise RuntimeError("Use PYTHONPATH=build/python/src:python for the tree's adjoint feature")
        self.set_export_mode(cfg.export_mode)
        for _ in range(cfg.settle_steps):
            self._advance()
        geo = self.slot.geometry()
        read = lambda key: np.asarray(uipc.view(geo.meta().find(key))).reshape(-1)
        mass = np.asarray(uipc.view(geo.vertices().find(uipc.builtin.volume))).reshape(-1) * 200.0
        self.layout = dict(n=self.n, dof_offset=int(read(uipc.builtin.dof_offset)[0]),
                           dof_count=int(read(uipc.builtin.dof_count)[0]), mass=mass,
                           strength=cfg.constraint_strength, anchor_idx=self.held)
        self.steps = 0
        self.initial = self.snapshot()
        self.reset(0)

    def set_export_mode(self, mode: str):
        """Takes effect from the next advance; a build without the mode API accepts only the default."""
        if not hasattr(self.feature, "set_export_mode"):
            if mode != "last_iterate":
                raise RuntimeError("This build's adjoint feature has no export modes; rebuild the tree")
            return
        self.feature.set_export_mode(mode)
        self.cfg.export_mode = mode

    def _advance(self):
        self.world.advance()
        self.world.retrieve()
        if not self.world.is_valid():
            raise RuntimeError(f"Invalid world at frame {self.world.frame()}")

    def positions(self):
        return np.asarray(self.uipc.view(self.slot.geometry().positions())).reshape(-1, 3).copy()

    def velocities(self):
        # World.retrieve updates display positions, not the geometry's initial velocity attribute.
        self.state_accessor.copy_to(self.state_geometry)
        attr = self.state_geometry.vertices().find("velocity")
        if attr is None:
            raise RuntimeError("No retrieved velocity attribute; cannot define complete x/v state")
        return np.asarray(self.uipc.view(attr)).reshape(-1, 3).copy()

    def observation(self):
        scale = self.cfg.state_scale
        return np.concatenate(((self.positions()-self.asset.rest).ravel(), self.velocities().ravel(),
                               self.anchor-self.origin, self.goal-self.origin)) / scale

    def snapshot(self):
        if not self.world.dump():
            raise RuntimeError("World.dump failed")
        return dict(frame=int(self.world.frame()), anchor=self.anchor.copy(), goal=self.goal.copy(),
                    steps=self.steps, obs=self.observation().copy())

    def restore(self, snap):
        if not self.world.recover(snap["frame"]):
            raise RuntimeError("World.recover failed")
        self.world.retrieve()
        self.anchor, self.goal = snap["anchor"].copy(), snap["goal"].copy()
        self.steps = snap["steps"]
        error = np.max(np.abs(self.observation()-snap["obs"]))
        if error > 1e-8:
            raise RuntimeError(f"Snapshot x/v/bookkeeping mismatch: {error}")

    def reset(self, seed):
        self.restore(self.initial)
        self.goal = self.task.sample_goal(np.random.default_rng(seed), self.asset.rest, self.positions()[self.marker].mean(0))
        self.steps = 0
        return self.observation()

    def distance(self, positions=None):
        x = self.positions() if positions is None else positions
        delta = x[self.marker].mean(0)-self.goal
        return float(np.sqrt(delta@delta+self.cfg.distance_epsilon**2))

    def step(self, action, capture=False):
        cfg = self.cfg
        action = np.asarray(action, dtype=np.float64).reshape(3)
        if not np.isfinite(action).all() or (np.abs(action) > 1+1e-7).any():
            raise ValueError("Expected finite normalized actions in [-1,1]")
        start = self.anchor.copy()
        old_distance = self.distance()
        target = np.clip(start + cfg.max_translation*action, [0.25, -0.35, 0.004], [0.85, 0.35, 0.45])
        control_jac = np.diag(((start + cfg.max_translation*action > [0.25, -0.35, 0.004]) &
                               (start + cfg.max_translation*action < [0.85, 0.35, 0.45])).astype(float)) * cfg.max_translation
        lus, residuals = [], []
        t0 = time.monotonic()
        for k in range(cfg.action_repeat):
            self.anchor = start + (k+1)/cfg.action_repeat*(target-start)
            previous_positions = self.positions() if capture else None
            self._advance()
            if capture:
                velocity_error = np.max(np.abs(self.velocities()-(self.positions()-previous_positions)/cfg.dt))
                if velocity_error > 1e-7:
                    raise RuntimeError(f"BDF1 velocity/state accessor mismatch: {velocity_error}")
                rows, cols, values, gradient = self.feature.export_system()
                mat = system_to_matrix(rows, cols, values, int(self.feature.dof_count()))
                lus.append(scipy.sparse.linalg.splu(mat.tocsc()))
                residuals.append(float(np.linalg.norm(gradient)))
        forward_s = time.monotonic()-t0
        x = self.positions()
        distance = self.distance(x)
        reward = (old_distance-distance)/cfg.max_translation - cfg.action_cost*float(action@action)
        self.steps += 1
        out = dict(obs=self.observation(), reward=reward, done=self.steps >= cfg.horizon,
                   distance=distance, success=distance < self.task.success_tolerance,
                   capture_forward_s=forward_s, newton_gradient_norms=residuals)
        if capture:
            t1 = time.monotonic()
            frames = tangent_pass(lus, self.layout, return_frames=True) @ control_jac
            dx = frames[-1]
            previous = frames[-2] if len(frames) > 1 else np.zeros_like(dx)
            dv = (dx-previous)/cfg.dt
            tangent = np.concatenate((dx, dv, control_jac, np.zeros((3, 3)))) / cfg.state_scale
            delta = x[self.marker].mean(0)-self.goal
            dc = dx.reshape(self.n, 3, 3)[self.marker].mean(0)
            dr = -(delta/distance)@dc/cfg.max_translation - 2*cfg.action_cost*action
            if not np.isfinite(tangent).all() or not np.isfinite(dr).all():
                raise RuntimeError("Nonfinite decision tangent")
            out.update(tangent=tangent, reward_gradient=dr, tangent_s=time.monotonic()-t1)
        return out

    def describe(self):
        return dict(config=asdict(self.cfg), variant="cloth_drag_direct_picker", state_dim=self.obs_dim,
                    vertices=self.n, derivative=f"{self.cfg.export_mode} Hessian with inertia-only history")
