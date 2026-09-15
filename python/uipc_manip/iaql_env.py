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
from .physics_gradient_adjoint import slot_factorizations_from_triplets, system_to_matrix, tangent_pass
from .iaql_tangent import BatchedTangent


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
    reward_mode: str = "dense"
    num_slots: int = 1
    tangent_device: str = "cpu"
    """``cpu``: per-slot SuperLU factorisations on the host; ``cuda``: the whole decision tangent as
    batched dense block algebra on the GPU (``iaql_tangent.BatchedTangent``)."""
    """Identical cloths stepping in lockstep in one World, one metre apart; the observation, action,
    reward and tangent gain a leading slot axis when more than one."""
    """``dense``: progress toward the goal every decision; ``terminal``: the distance term only at the
    episode's last decision (the action cost stays), so the immediate reward gradient carries nothing
    about the task and only the continuation channel can."""
    friction_chain: bool = False
    """Add the friction gradient's lagged dependence on the previous substep (the backend's
    ``export_prev_coupling`` blocks) to the tangent chain; needs a converged export mode."""
    """Which system the backend leaves for the tangent: the last Newton iterate's projected
    matrix, or a re-assembly at the accepted state, projected (``converged``) or not
    (``converged_raw``); see LinearSystemAdjointFeature.set_export_mode."""


class IAQLClothEnv:
    """One or several identical cloths in one World. With ``cfg.num_slots > 1`` the slots sit one
    metre apart along y, never touch, and step in lockstep (one Newton solve, one exported system,
    one factorisation per substep serve every slot; each slot's tangent is its own block of the
    solution); the observation, action, reward and tangent then carry a leading slot axis, and a
    single slot keeps the scalar interface."""
    action_dim = 3
    SLOT_SPACING = 1.0
    BOX_LO = np.array([0.25, -0.35, 0.004])
    BOX_HI = np.array([0.85, 0.35, 0.45])

    def __init__(self, workspace: Path, cfg: IAQLEnvConfig):
        import uipc
        from uipc.constitution import SoftPositionConstraint
        from uipc.geometry import ground
        from uipc.diff_sim import LinearSystemAdjointFeature
        from uipc.core import FiniteElementStateAccessorFeature

        self.uipc, self.cfg = uipc, cfg
        if cfg.action_repeat < 1 or min(cfg.dt, cfg.state_scale, cfg.max_translation) <= 0:
            raise ValueError("Positive integration/scaling parameters are required")
        if cfg.num_slots < 1:
            raise ValueError("num_slots must be at least 1")
        self.N = int(cfg.num_slots)
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
        self.task = get_task("cloth_drag")
        self.shifts = np.array([[0.0, self.SLOT_SPACING * j, 0.0] for j in range(self.N)])
        self.assets = [build_cloth(center=(0.55, self.SLOT_SPACING * j, 0.01)) for j in range(self.N)]
        self.asset = self.assets[0]
        rest0 = self.assets[0].rest
        self.held = self.task.grasp_vertices(rest0)
        self.marker = self.task.marker_vertices(rest0)
        self.n = len(rest0)
        self.obs_dim = 6 * self.n + 6
        self.rests = np.stack([a.rest for a in self.assets])
        self.anchors = self.rests[:, self.held].mean(1)
        self.offsets = rest0[self.held] - self.anchors[0]
        self.origins = self.rests.mean(1)
        self.goals = self.origins.copy()
        obj = self.scene.objects().create("iaql_cloth")
        self.slots = []
        for asset in self.assets:
            SoftPositionConstraint().apply_to(asset.mesh, cfg.constraint_strength)
            element.apply_to(asset.mesh)
            self.slots.append(obj.geometries().create(asset.mesh)[0])
        self.slot = self.slots[0]
        floor = ground(0.0, np.array([0., 0., 1.]))
        element.apply_to(floor)
        self.scene.objects().create("table").geometries().create(floor)

        def animate(info):
            for j, geo_slot in enumerate(info.geo_slots()):
                geo = geo_slot.geometry()
                flags = uipc.view(geo.vertices().find(uipc.builtin.is_constrained)).reshape(-1)
                flags[:] = 0
                flags[self.held] = 1
                aim = uipc.view(geo.vertices().find(uipc.builtin.aim_position)).reshape(-1, 3)
                aim[self.held] = self.anchors[j] + self.offsets

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
        self.layouts = []
        for j, slot in enumerate(self.slots):
            geo = slot.geometry()
            read = lambda key: np.asarray(uipc.view(geo.meta().find(key))).reshape(-1)  # noqa: E731
            mass = np.asarray(uipc.view(geo.vertices().find(uipc.builtin.volume))).reshape(-1) * 200.0
            self.layouts.append(dict(n=self.n, dof_offset=int(read(uipc.builtin.dof_offset)[0]),
                                     dof_count=int(read(uipc.builtin.dof_count)[0]), mass=mass,
                                     strength=cfg.constraint_strength, anchor_idx=self.held,
                                     vertex_offset=int(read(uipc.builtin.global_vertex_offset)[0]),
                                     fem_offset=int(read(uipc.builtin.backend_fem_vertex_offset)[0])))
        self.layout = self.layouts[0]
        if cfg.friction_chain and not hasattr(self.feature, "export_prev_coupling"):
            raise RuntimeError("This build's adjoint feature exports no lagged coupling; rebuild the tree")
        self.batched = BatchedTangent(self.layouts, cfg.action_repeat, cfg.tangent_device) if cfg.tangent_device != "cpu" else None
        self.steps = 0
        self.initial = self.snapshot()
        self.reset(0)

    # ------------------------------------------------------------------ single-slot views
    @property
    def anchor(self):
        return self.anchors[0]

    @anchor.setter
    def anchor(self, value):
        self.anchors[0] = value

    @property
    def goal(self):
        return self.goals[0]

    @goal.setter
    def goal(self, value):
        self.goals[0] = value

    @property
    def origin(self):
        return self.origins[0]

    def _squeeze(self, value):
        return value[0] if self.N == 1 else value

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

    def all_positions(self):
        return np.stack([np.asarray(self.uipc.view(s.geometry().positions())).reshape(-1, 3) for s in self.slots])

    def positions(self):
        return self.all_positions()[0].copy()

    def all_velocities(self):
        # World.retrieve updates display positions, not the geometry's initial velocity attribute.
        self.state_accessor.copy_to(self.state_geometry)
        attr = self.state_geometry.vertices().find("velocity")
        if attr is None:
            raise RuntimeError("No retrieved velocity attribute; cannot define complete x/v state")
        v = np.asarray(self.uipc.view(attr)).reshape(-1, 3)
        return np.stack([v[lay["fem_offset"]:lay["fem_offset"] + self.n] for lay in self.layouts])

    def velocities(self):
        return self.all_velocities()[0].copy()

    def all_observations(self):
        scale = self.cfg.state_scale
        x, v = self.all_positions(), self.all_velocities()
        rows = [np.concatenate(((x[j] - self.rests[j]).ravel(), v[j].ravel(),
                                self.anchors[j] - self.origins[j], self.goals[j] - self.origins[j]))
                for j in range(self.N)]
        return np.stack(rows) / scale

    def observation(self):
        return self._squeeze(self.all_observations())

    def snapshot(self):
        if not self.world.dump():
            raise RuntimeError("World.dump failed")
        return dict(frame=int(self.world.frame()), anchors=self.anchors.copy(), goals=self.goals.copy(),
                    steps=self.steps, obs=self.all_observations().copy())

    def restore(self, snap):
        if not self.world.recover(snap["frame"]):
            raise RuntimeError("World.recover failed")
        self.world.retrieve()
        self.anchors, self.goals = snap["anchors"].copy(), snap["goals"].copy()
        self.steps = snap["steps"]
        error = np.max(np.abs(self.all_observations() - snap["obs"]))
        if error > 1e-8:
            raise RuntimeError(f"Snapshot x/v/bookkeeping mismatch: {error}")

    def reset(self, seed):
        """One seed draws every slot's goal in turn, so a single slot keeps the old seed semantics."""
        self.restore(self.initial)
        rng = np.random.default_rng(seed)
        x = self.all_positions()
        for j in range(self.N):
            local = self.task.sample_goal(rng, self.rests[j] - self.shifts[j], x[j][self.marker].mean(0) - self.shifts[j])
            self.goals[j] = local + self.shifts[j]
        self.steps = 0
        return self.observation()

    def all_distances(self, positions=None):
        x = self.all_positions() if positions is None else positions
        delta = x[:, self.marker].mean(1) - self.goals
        return np.sqrt((delta * delta).sum(-1) + self.cfg.distance_epsilon ** 2)

    def distance(self, positions=None):
        if positions is not None and positions.ndim == 2:
            positions = positions[None]
        return float(self.all_distances(positions)[0])

    def step(self, action, capture=False):
        cfg = self.cfg
        actions = np.asarray(action, dtype=np.float64).reshape(self.N, 3)
        if not np.isfinite(actions).all() or (np.abs(actions) > 1 + 1e-7).any():
            raise ValueError("Expected finite normalized actions in [-1,1]")
        starts = self.anchors.copy()
        old_distances = self.all_distances()
        lo, hi = self.BOX_LO[None] + self.shifts, self.BOX_HI[None] + self.shifts
        aims = starts + cfg.max_translation * actions
        targets = np.clip(aims, lo, hi)
        inside = ((aims > lo) & (aims < hi)).astype(float)
        control_jacs = np.stack([np.diag(inside[j]) * cfg.max_translation for j in range(self.N)])
        lus, residuals, couplings = [], [], []
        chain = cfg.friction_chain and self.cfg.export_mode != "last_iterate"
        batched = self.batched if capture else None
        if batched is not None:
            batched.begin()
        t0 = time.monotonic()
        tangent_gpu_s = 0.0
        for k in range(cfg.action_repeat):
            self.anchors = starts + (k + 1) / cfg.action_repeat * (targets - starts)
            previous_positions = self.all_positions() if capture else None
            self._advance()
            if capture:
                velocity_error = np.max(np.abs(self.all_velocities() - (self.all_positions() - previous_positions) / cfg.dt))
                if velocity_error > 1e-7:
                    raise RuntimeError(f"BDF1 velocity/state accessor mismatch: {velocity_error}")
                rows, cols, values, gradient = self.feature.export_system()
                residuals.append(float(np.linalg.norm(gradient)))
                coupling = None
                if chain:
                    b_rows, b_cols, blocks = self.feature.export_prev_coupling()
                    coupling = (np.asarray(b_rows), np.asarray(b_cols), np.asarray(blocks))
                    couplings.append(coupling)
                if batched is not None:
                    t2 = time.monotonic()
                    batched.substep(rows, cols, values, coupling)
                    tangent_gpu_s += time.monotonic() - t2
                elif self.N == 1:
                    mat = system_to_matrix(rows, cols, values, int(self.feature.dof_count()))
                    lus.append([(scipy.sparse.linalg.splu(mat.tocsc()), self.layouts[0])])
                else:
                    # Slots never touch: one factorisation per slot block straight from the triplets,
                    # on a thread pool, never the whole system.
                    lus.append(slot_factorizations_from_triplets(rows, cols, values, self.layouts))
        forward_s = time.monotonic() - t0
        x = self.all_positions()
        distances = self.all_distances(x)
        self.steps += 1
        done = self.steps >= cfg.horizon
        if cfg.reward_mode == "terminal":
            progress = -distances / cfg.max_translation if done else np.zeros(self.N)
        elif cfg.reward_mode == "dense":
            progress = (old_distances - distances) / cfg.max_translation
        else:
            raise ValueError(f"Unknown reward_mode {cfg.reward_mode!r}")
        rewards = progress - cfg.action_cost * (actions * actions).sum(-1)
        successes = distances < self.task.success_tolerance
        out = dict(obs=self.observation(), reward=self._squeeze(rewards), done=done,
                   distance=self._squeeze(distances), success=self._squeeze(successes),
                   capture_forward_s=forward_s, newton_gradient_norms=residuals)
        if capture:
            t1 = time.monotonic()
            if batched is not None:
                dx_all, _, tangent = batched.finish(control_jacs, cfg.dt, cfg.state_scale)
            else:
                dx_all, tangents = [], []
                for j, lay in enumerate(self.layouts):
                    slot_couplings = None
                    if chain:
                        off = lay["vertex_offset"]
                        slot_couplings = []
                        for b_rows, b_cols, blocks in couplings:
                            keep = (b_rows >= off) & (b_rows < off + self.n)
                            slot_couplings.append((b_rows[keep] - off, b_cols[keep] - off, blocks[keep]))
                    slot_lus = [frame[j][0] for frame in lus]
                    frames = tangent_pass(slot_lus, lus[0][j][1], return_frames=True, prev_coupling=slot_couplings) @ control_jacs[j]
                    dx = frames[-1]
                    previous = frames[-2] if len(frames) > 1 else np.zeros_like(dx)
                    dv = (dx - previous) / cfg.dt
                    dx_all.append(dx)
                    tangents.append(np.concatenate((dx, dv, control_jacs[j], np.zeros((3, 3)))) / cfg.state_scale)
                tangent = np.stack(tangents)
            reward_gradients = []
            for j in range(self.N):
                delta = x[j][self.marker].mean(0) - self.goals[j]
                dc = np.asarray(dx_all[j]).reshape(self.n, 3, 3)[self.marker].mean(0)
                task_term = -(delta / distances[j]) @ dc / cfg.max_translation
                if cfg.reward_mode == "terminal" and not done:
                    task_term = np.zeros(3)
                reward_gradients.append(task_term - 2 * cfg.action_cost * actions[j])
            dr = np.stack(reward_gradients)
            if not np.isfinite(tangent).all() or not np.isfinite(dr).all():
                raise RuntimeError("Nonfinite decision tangent")
            out.update(tangent=self._squeeze(tangent), reward_gradient=self._squeeze(dr), tangent_s=time.monotonic() - t1 + tangent_gpu_s,
                       friction_chain=chain, coupling_blocks=int(sum(len(c[0]) for c in couplings)))
        return out

    def describe(self):
        return dict(config=asdict(self.cfg), variant="cloth_drag_direct_picker", state_dim=self.obs_dim,
                    vertices=self.n, slots=self.N, derivative=f"{self.cfg.export_mode} Hessian with inertia-only history")
