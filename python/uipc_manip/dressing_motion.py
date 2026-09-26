"""Prescribed SMPL-X motion through IPC position constraints, with measured lag.

The human surface is driven inside the contact solve. No vertices are
teleported into a fixed collider after a step. This is a soft target drive,
not a guarantee of exact motion: excess tracking error invalidates a trial.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .dressing_env import GenesisIPCDressingEnv
from .grab_motion import BodyMotion


class MotionDressingEnv(GenesisIPCDressingEnv):
    def __init__(self, cfg, motion: BodyMotion, *, onset_s=1., speed=1.,
                 drive_strength=1e6, tracking_tolerance_m=.002, max_substep_m=.015,
                 cell_factory=None):
        values = [onset_s, speed, drive_strength, tracking_tolerance_m, max_substep_m]
        if not np.isfinite(values).all() or onset_s < 0 or min(values[1:]) <= 0:
            raise ValueError("Motion onset must be nonnegative; speed/strength/tolerances positive")
        if cfg.collision_geometry != "full_body":
            raise ValueError("The motion pilot requires full-body collision geometry")
        if cfg.obs.mode not in ("wang_live_arm", "wang_static_arm"):
            raise ValueError("Use wang_live_arm (occluded) or explicit wang_static_arm diagnostic")
        self.motion = motion
        self.motion_onset_s, self.motion_speed = float(onset_s), float(speed)
        self.drive_strength = float(drive_strength)
        self.body_tracking_tolerance_m = float(tracking_tolerance_m)
        self.max_body_substep_m = float(max_substep_m)
        self.motion_time_s = 0.
        self.motion_running = False
        self.body_tracking_max_m = 0.
        self._body_target = motion.vertices[0].copy()
        self._target_joints = motion.joints[0].copy()
        super().__init__(cfg, num_envs=1, cell_factory=cell_factory)
        self._original_motion_cells = deepcopy(self.cells)
        # Keep the camera at its initial pose; only the actual arm/body moves.
        initial_fingers = [c.finger.copy() for c in self.cells]
        initial_shoulders = [c.shoulder.copy() for c in self.cells]
        visible = self._batched_obs.visible_points

        def fixed_camera(arms, cloths, fingers, shoulders, rngs, augment, rig=None):
            return visible(arms, cloths, initial_fingers, initial_shoulders, rngs, augment, rig=rig)

        self._batched_obs.visible_points = fixed_camera

    def _make_human_collider(self, mesh, env_idx):
        from uipc.constitution import Empty, SoftPositionConstraint
        if int(self.cells[env_idx].human) != int(self.motion.metadata["body_id"]):
            raise ValueError("Motion recipient body ID differs from the scene")
        actual = self.collider_meshes[env_idx][0]
        error = np.max(np.linalg.norm(actual - self.motion.vertices[0], axis=1))
        if error > 1e-5:
            raise ValueError(f"Motion frame zero does not match scene body ({error:g} m)")
        Empty().apply_to(mesh, 1000., .00015)
        SoftPositionConstraint().apply_to(mesh, self.drive_strength)
        # SMPL-X has self-overlapping regions in seated poses. Body/body
        # self-contact was also absent for the original single affine collider.
        self._uipc.view(mesh.meta().find(self._uipc.builtin.self_collision))[:] = 0

    def _animate_human_collider(self, ipc_object, env_idx):
        def animate(info):
            mesh = info.geo_slots()[0].geometry()
            self._uipc.view(mesh.vertices().find(self._uipc.builtin.is_constrained))[:] = 1
            self._uipc.view(mesh.vertices().find(self._uipc.builtin.aim_position)).reshape(-1, 3)[:] = self._body_target
        self.coupler._ipc_animator.insert(ipc_object, animate)

    def sample_future(self, offset_s):
        """Privileged diagnostic only. Controllers with causal inputs must not call this."""
        return self.motion.sample(max(0., self.motion_time_s + offset_s - self.motion_onset_s)
                                  * self.motion_speed)

    def _sim_step(self):
        if self.motion_running:
            next_t = self.motion_time_s + self.cfg.dt
            target, joints, _ = self.motion.sample(max(0., next_t - self.motion_onset_s) * self.motion_speed)
            jump = float(np.max(np.linalg.norm(target - self._body_target, axis=1)))
            if jump > self.max_body_substep_m:
                raise RuntimeError(f"Body motion exceeds substep bound: {jump:g} m at {next_t:g} s")
            self._body_target, self._target_joints = target, joints
            self.motion_time_s = next_t
        super()._sim_step()
        actual = np.asarray(self._uipc.view(self.arm_slots[0].geometry().positions())).reshape(-1, 3).copy()
        error = float(np.max(np.linalg.norm(actual - self._body_target, axis=1)))
        if not np.isfinite(actual).all() or not np.isfinite(error) or error > self.body_tracking_tolerance_m:
            raise RuntimeError(f"Human target tracking invalid at {self.motion_time_s:g} s: {error:g} m")
        self.body_tracking_max_m = max(self.body_tracking_max_m, error)
        from .dressing_body import _reward_line_landmarks
        cell = self.cells[0]
        cell.human_points = actual
        cell.arm_points = actual[self.motion.arm_indices].copy()
        cell.landmarks.update(_reward_line_landmarks(actual, self._target_joints))
        self.arm_meshes[0] = (cell.arm_points, cell.arm_faces)
        self.collider_meshes[0] = (actual, self.motion.faces)
        self.arm_vertices, self.collider_vertices = cell.arm_points, actual
        # The static builder caches body occluders by object identity.
        self._batched_obs.__dict__.get("_occluders", {}).clear()

    def reset(self, seeds=None):
        self.motion_running = False
        self.motion_time_s = 0.
        self._body_target = self.motion.vertices[0].copy()
        self._target_joints = self.motion.joints[0].copy()
        self.cells = deepcopy(self._original_motion_cells)
        self.body_tracking_max_m = 0.
        obs = super().reset(seeds)
        self.motion_running = True
        return obs

    def motion_diagnostics(self):
        return dict(time_s=self.motion_time_s, body_tracking_max_m=self.body_tracking_max_m,
                    body_tracking_tolerance_m=self.body_tracking_tolerance_m,
                    drive="Empty FEM + per-vertex SoftPositionConstraint",
                    observation_mode=self.cfg.obs.mode)
