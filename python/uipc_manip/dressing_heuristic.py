"""Seven-stage scripted dressing expert, ported from the Newton teacher's ``mdp/heuristic.py``.

approach -> finger -> middle -> align-yaw -> align-pitch -> elbow-hook -> last.
The gripper is held ``z_offset`` metres above the arm landmarks while the sleeve
opening is driven finger -> past the elbow -> hooked over the elbow -> past the
shoulder; the cuff's alignment line is rotated toward the forearm and then the
upper-arm direction, at half rate during translation and at full rate in the
two dedicated alignment stages. Translation targets are requested at the
reference's 8 mm per decision and clipped by the environment's speed cap, so
the expert moves at the cap. This is the reachability baseline the Newton
pipeline runs before any policy training.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.asarray(fallback, dtype=np.float64)


class HeuristicDressingPolicy:
    STAGES = ("approach", "finger", "middle", "align_yaw", "align_pitch", "elbow_hook", "last", "done")

    def __init__(
        self,
        env,
        *,
        translation_step: float = 0.008,
        translation_tolerance: float = 0.012,
        translation_max_steps: int = 240,
        approach_max_steps: int = 120,
        z_offset: float = 0.12,
        z_offset_range: tuple[float, float] | None = None,
        elbow_overshoot: float = 0.08,
        elbow_hook_offset: float = 0.06,
        shoulder_overshoot: float = 0.10,
        rotation_step: float = 0.05,
        align_tolerance_deg: float = 12.0,
        align_max_steps: int = 30,
        proximity_push_z: float = 0.01,
        proximity_push_distance_m: float = 0.05,
    ) -> None:
        self.env = env
        self.n = int(env.num_envs)
        self.translation_step = float(translation_step)
        self.translation_tolerance = float(translation_tolerance)
        self.translation_max_steps = int(translation_max_steps)
        self.approach_max_steps = int(approach_max_steps)
        self.z_offset = float(z_offset)
        self.z_offset_range = z_offset_range
        self.elbow_overshoot = float(elbow_overshoot)
        self.elbow_hook_offset = float(elbow_hook_offset)
        self.shoulder_overshoot = float(shoulder_overshoot)
        self.rotation_step = float(rotation_step)
        self.align_tolerance = float(np.deg2rad(align_tolerance_deg))
        self.align_max_steps = int(align_max_steps)
        self.proximity_push_z = float(proximity_push_z)
        self.proximity_push_distance = float(proximity_push_distance_m)
        self.stage = np.zeros(self.n, dtype=np.int64)
        self._steps = np.zeros(self.n, dtype=np.int64)
        self._align_steps = np.zeros(self.n, dtype=np.int64)
        self._targets: list[dict] = [{} for _ in range(self.n)]
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        for i, cell in enumerate(self.env.cells):
            finger, elbow, shoulder = cell.finger, cell.elbow, cell.shoulder
            forearm_dir = _unit(elbow - finger)
            upperarm_dir = _unit(shoulder - elbow)
            z_off = self.z_offset
            if self.z_offset_range is not None and rng is not None:
                z_off = float(rng.uniform(*self.z_offset_range))
            z_up = np.array([0.0, 0.0, z_off])
            self._targets[i] = {
                "forearm_dir": forearm_dir,
                "upperarm_dir": upperarm_dir,
                "approach": finger + z_up,
                "finger": finger + z_up,
                "middle": elbow + forearm_dir * self.elbow_overshoot,
                "elbow_hook": elbow + upperarm_dir * self.elbow_hook_offset,
                "last": shoulder + upperarm_dir * self.shoulder_overshoot,
            }
            self.stage[i] = 0
            self._steps[i] = 0
            self._align_steps[i] = 0

    def stage_names(self) -> list[str]:
        return [self.STAGES[int(s)] for s in self.stage]

    # ------------------------------------------------------------------
    def actions(self, positions: list[np.ndarray] | None = None) -> np.ndarray:
        env = self.env
        positions = env.positions() if positions is None else positions
        out = np.zeros((self.n, env.action_dim), dtype=np.float32)
        max_t, max_r = float(env.cfg.max_translation), float(env.cfg.max_rotation)
        for i, (cell, p) in enumerate(zip(env.cells, positions, strict=True)):
            stage = int(self.stage[i])
            if stage >= 7:
                continue
            t = self._targets[i]
            gripper = env._anchor[i]
            opening = p[cell.opening_idx].mean(axis=0)
            line = p[cell.alignment_idx[0]] - p[cell.alignment_idx[1]]
            line_valid = float(np.linalg.norm(line)) > 1e-9
            line_dir = _unit(line)
            arm_min = float(np.min(np.linalg.norm(cell.arm_points - gripper[None, :], axis=1)))
            target_dir = t["forearm_dir"] if stage <= 2 else t["upperarm_dir"]
            a = np.zeros(6)
            if stage == 0:
                self._translate(a, i, gripper, t["approach"], self.approach_max_steps)
                self._continuous_align(a, line_dir, line_valid, target_dir, max_r)
            elif stage == 1:
                self._translate(a, i, gripper, t["finger"], self.translation_max_steps)
                self._continuous_align(a, line_dir, line_valid, target_dir, max_r)
            elif stage == 2:
                self._translate(a, i, opening, t["middle"], self.translation_max_steps)
                self._continuous_align(a, line_dir, line_valid, target_dir, max_r)
            elif stage in (3, 4):
                direction = t["elbow_hook"] - opening
                dist = float(np.linalg.norm(direction))
                if dist >= self.translation_tolerance:
                    a[:3] += direction / dist * min(self.translation_step, dist) / max_t
                self._dedicated_align(a, i, line_dir, line_valid, target_dir, max_r)
            elif stage == 5:
                self._translate(a, i, opening, t["elbow_hook"], self.translation_max_steps)
                self._continuous_align(a, line_dir, line_valid, target_dir, max_r)
            elif stage == 6:
                self._translate(a, i, opening, t["last"], self.translation_max_steps)
                self._continuous_align(a, line_dir, line_valid, target_dir, max_r)
                if arm_min < self.proximity_push_distance:
                    a[2] += self.proximity_push_z / max_t
            out[i] = np.clip(a, -1.0, 1.0)
        return out

    def _advance(self, i: int) -> None:
        self.stage[i] += 1
        self._steps[i] = 0
        self._align_steps[i] = 0

    def _translate(self, a: np.ndarray, i: int, point: np.ndarray, target: np.ndarray, max_steps: int) -> None:
        direction = target - point
        dist = float(np.linalg.norm(direction))
        if dist < self.translation_tolerance:
            self._advance(i)
            return
        self._steps[i] += 1
        if int(self._steps[i]) >= max_steps:
            self._advance(i)
            return
        a[:3] = direction / dist * self.translation_step / float(self.env.cfg.max_translation)

    def _continuous_align(self, a, line_dir, line_valid, target_dir, max_r) -> None:
        if not line_valid:
            return
        if float(line_dir @ target_dir) < 0.0:
            line_dir = -line_dir
        diff = float(np.arccos(np.clip(line_dir @ target_dir, -1.0, 1.0)))
        if diff <= self.align_tolerance:
            return
        axis = np.cross(line_dir, target_dir)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return
        a[3:6] += axis / n * (min(self.rotation_step * 0.5, diff) / max_r)

    def _dedicated_align(self, a, i, line_dir, line_valid, target_dir, max_r) -> None:
        if not line_valid:
            self._advance(i)
            return
        if float(line_dir @ target_dir) < 0.0:
            line_dir = -line_dir
        diff = float(np.arccos(np.clip(line_dir @ target_dir, -1.0, 1.0)))
        self._align_steps[i] += 1
        if diff < self.align_tolerance or int(self._align_steps[i]) >= self.align_max_steps:
            self._advance(i)
            return
        axis = np.cross(line_dir, target_dir)
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            self._advance(i)
            return
        a[3:6] = axis / n * (min(self.rotation_step, diff) / max_r)
