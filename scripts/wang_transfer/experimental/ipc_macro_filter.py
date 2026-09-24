"""Multi-decision IPC recovery on top of the one-decision candidate filter.

A one-decision lookahead never backs out of a snag: retreating loses progress now and pays off only
several decisions later. When the sleeve stalls, this simulates a set of short open-loop macros
(retreat, lift, lateral, yaw, hold, each followed by the policy's own action) from the current IPC
state, scores the state each one ends in, and returns the best macro to execute in full. It changes
executed actions, never network weights, and uses privileged simulation geometry.
"""
from __future__ import annotations

import numpy as np

from ipc_action_filter import IPCActionFilter
from physical_sleeve import measure


class IPCMacroFilter(IPCActionFilter):
    def __init__(self, env, sections, masses, *, strength_gain=1., horizon=8, margin=.003):
        super().__init__(env, sections, masses, strength_gain=strength_gain)
        self.horizon, self.margin = int(horizon), float(margin)
        finger, elbow = self.landmarks[0], self.landmarks[1]
        self.forward = (elbow - finger) / np.linalg.norm(elbow - finger)
        self.up = np.array([0., 0., 1.])
        lateral = np.cross(self.forward, self.up)
        self.lateral = lateral / np.linalg.norm(lateral)

    def progress(self, positions):
        geometry = measure(self.sections, positions, self.landmarks)
        return float(np.mean([r['s'] for r in geometry['rings']]) * self.arm_length), bool(geometry['sleeve_wrapped'])

    def macros(self, nominal):
        h, half = self.horizon, self.horizon // 2

        def move(direction, rotation=None):
            a = np.zeros(6)
            a[:3] = np.clip(direction / max(np.max(np.abs(direction)), 1e-9), -1, 1)
            if rotation is not None:
                a[5] = rotation
            return a

        nominal = np.clip(np.asarray(nominal, float), -1, 1)
        back_lift = move(-self.forward + self.up)
        yaw = [nominal.copy() for _ in range(h)]
        return {
            'nominal': [nominal] * h,
            'hold': [np.zeros(6)] * h,
            'back_then_policy': [move(-self.forward)] * half + [nominal] * (h - half),
            'lift_then_policy': [move(self.up)] * half + [nominal] * (h - half),
            'drop_then_policy': [move(-self.up)] * half + [nominal] * (h - half),
            'left_then_policy': [move(self.lateral)] * half + [nominal] * (h - half),
            'right_then_policy': [move(-self.lateral)] * half + [nominal] * (h - half),
            'back_lift_then_forward': [back_lift] * half + [move(self.forward)] * (h - half),
            'yaw_plus': [np.r_[a[:3], 0., 0., 1.] for a in yaw],
            'yaw_minus': [np.r_[a[:3], 0., 0., -1.] for a in yaw],
        }

    def rollout(self, actions):
        e = self.env
        tracking = 0.
        for action in actions:
            _, _, done, infos = e.step(np.asarray(action, float)[None])
            if done.any() or infos[0].get('sim_error'):
                return None
            tracking = max(tracking, float(infos[0]['tracking_error']))
        positions = e.positions()[0]
        progress, wrapped = self.progress(positions)
        target = e._anchor[0] + e._offsets[0]
        force = float(np.linalg.norm((self.stiffness[:, None] * (target - positions[self.indices])).sum(0)))
        stretch = np.linalg.norm(positions[self.edges[:, 0]] - positions[self.edges[:, 1]], axis=1) / self.rest_lengths
        feasible = tracking <= .019 and np.quantile(stretch, .99) <= 2.25 and stretch.max() <= 4.
        score = progress + .03 * wrapped - 2e-5 * max(force - 40., 0.) - (0. if feasible else 1.)
        return dict(score=float(score), progress_m=progress, wrapped=wrapped, gripper_N=force,
                    tracking_m=tracking, feasible=bool(feasible), edge_p99=float(np.quantile(stretch, .99)))

    def plan(self, nominal):
        macros = self.macros(nominal)
        snap = self.snapshot()
        rows = {}
        try:
            for name, actions in macros.items():
                self.restore(snap)
                rows[name] = self.rollout(actions) or dict(score=-1e9, feasible=False)
        finally:
            self.restore(snap)
        best = max(rows, key=lambda k: rows[k]['score'])
        if rows[best]['score'] < rows['nominal']['score'] + self.margin:
            best = 'nominal'
        return [np.asarray(a, np.float32) for a in macros[best]], dict(selected=best, candidates=rows)
