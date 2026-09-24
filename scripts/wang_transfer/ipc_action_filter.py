"""One-decision IPC candidate evaluation with explicit state restoration.

This changes executed actions, never checkpoint weights. Candidate scores use
privileged simulation geometry and are not a deployable real-world controller.
"""
from __future__ import annotations

import copy
import numpy as np

from physical_sleeve import measure


class IPCActionFilter:
    def __init__(self, env, sections, masses, *, strength_gain=1.):
        if env.num_envs != 1:
            raise ValueError('Use an isolated single-slot world for IPC candidate evaluation')
        self.env, self.sections = env, sections
        self.landmarks = np.stack([env.cells[0].finger, env.cells[0].elbow, env.cells[0].shoulder])
        self.arm_length = np.linalg.norm(np.diff(self.landmarks, axis=0), axis=1).sum()
        self.indices = env._pickers[0]['anchor_idx']
        self.stiffness = env.cfg.constraint_strength * strength_gain * masses[self.indices] / env.cfg.dt**2
        self.edges = sections.edges
        initial = env.positions()[0]
        self.rest_lengths = np.linalg.norm(initial[self.edges[:, 0]] - initial[self.edges[:, 1]], axis=1)
        self.replay_checked = False
        self.noise_margin = .0002

    def snapshot(self):
        e = self.env
        if not e._world.dump():
            raise RuntimeError('IPC candidate snapshot failed')
        names = ('_anchor', '_offsets', '_last_progress', '_privileged', '_episode_step', '_force_trackers')
        state = {name: copy.deepcopy(getattr(e, name)) for name in names}
        if hasattr(e, '_violated'):
            state['_violated'] = copy.deepcopy(e._violated)
        return dict(frame=int(e._world.frame()), state=state, positions=e.positions()[0].copy(),
                    sim_step=e.scene.sim._cur_substep_global,
                    rngs=[copy.deepcopy(r.bit_generator.state) for r in e.rngs])

    def restore(self, snap):
        e = self.env
        if not e._world.recover(snap['frame']):
            raise RuntimeError('IPC candidate restore failed')
        e._world.retrieve()
        for name, value in snap['state'].items():
            setattr(e, name, copy.deepcopy(value))
        e.scene.sim._cur_substep_global = snap['sim_step']
        for rng, state in zip(e.rngs, snap['rngs']):
            rng.bit_generator.state = copy.deepcopy(state)
        e._update_targets()
        e._decision_times.clear()
        error = float(np.max(abs(e.positions()[0] - snap['positions'])))
        if error > 1e-8 or int(e._world.frame()) != snap['frame']:
            raise RuntimeError(f'IPC snapshot did not restore exactly: {error} m')
        return error

    def evaluate(self, action, nominal):
        e = self.env
        _, _, done, infos = e.step(action[None])
        if done.any() or infos[0].get('sim_error'):
            raise RuntimeError('IPC candidate caused an environment reset; abort this diagnostic')
        positions = e.positions()[0]
        geometry = measure(self.sections, positions, self.landmarks)
        target = e._anchor[0] + e._offsets[0]
        force = float(np.linalg.norm((self.stiffness[:, None] * (target - positions[self.indices])).sum(0)))
        stretch = np.linalg.norm(positions[self.edges[:, 0]] - positions[self.edges[:, 1]], axis=1) / self.rest_lengths
        tracking = float(infos[0]['tracking_error'])
        progress = float(np.mean([r['s'] for r in geometry['rings']]) * self.arm_length)
        # Local progress, then reduced excessive load; constrain large tracking
        # errors and deformation instead of rewarding a torn-off held patch.
        score = (progress - 2e-5 * max(force - 40., 0.)
                 - .0002 * float(np.sum((action - nominal)**2)))
        feasible = tracking <= .019 and np.quantile(stretch, .99) <= 2.25 and stretch.max() <= 4.
        if not feasible:
            score -= 1. + 100. * max(tracking - .019, 0.)
        return dict(score=float(score), progress_m=progress, gripper_N=force, tracking_m=tracking,
                    feasible=bool(feasible), wrapped=geometry['sleeve_wrapped'],
                    edge_p99=float(np.quantile(stretch, .99)), edge_max=float(stretch.max())), positions.copy()

    def improve(self, nominal):
        nominal = np.clip(np.asarray(nominal, float), -1, 1)
        candidates = [nominal.copy(), nominal * .5, np.zeros(6)]
        no_rotation = nominal.copy()
        no_rotation[3:] = 0.
        candidates.append(no_rotation)
        for axis in range(3):
            for direction in (-1., 1.):
                candidate = nominal.copy()
                candidate[axis] += direction * .25
                candidates.append(np.clip(candidate, -1, 1))
        unique = {}
        for candidate in candidates:
            unique.setdefault(tuple(candidate), candidate)
        candidates = list(unique.values())
        snap = self.snapshot()
        rows = []
        try:
            for action in candidates:
                self.restore(snap)
                row, positions = self.evaluate(action, nominal)
                row['action'] = action.tolist()
                rows.append(row)
                if len(rows) == 1 and not self.replay_checked:
                    self.restore(snap)
                    replay, replay_positions = self.evaluate(action, nominal)
                    delta = abs(replay['score'] - row['score'])
                    self.noise_margin = max(.0002, 2 * delta)
                    row['replay_position_error_m'] = float(np.max(abs(replay_positions - positions)))
                    row['replay_score_error'] = float(delta)
                    self.replay_checked = True
            best = int(np.argmax([r['score'] for r in rows]))
            if rows[best]['score'] < rows[0]['score'] + self.noise_margin:
                best = 0
        finally:
            self.restore(snap)
        return candidates[best].astype(np.float32), dict(selected=best, candidates=rows, noise_margin=self.noise_margin)
