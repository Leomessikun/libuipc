"""One-decision IPC candidate evaluation with explicit state restoration.

This changes executed actions, never checkpoint weights. Candidate scores use
privileged simulation geometry and are not a deployable real-world controller.
"""
from __future__ import annotations

import copy
import numpy as np

from physical_sleeve import measure


# A ring counts toward progress only once its centre is near the arm axis. Far from the arm, ``ring_on_arm``
# clamps the ring to the nearer arm end, so an unthreaded sleeve hanging past the shoulder read as full
# progress (ClothesNet: 26-40 % of states; failed episodes ended higher than accepted ones).
RING_NEAR_M = .12


def arm_progress(rings, arm_length):
    return float(np.mean([r['s'] if r['center_distance_m'] <= RING_NEAR_M else 0. for r in rings]) * arm_length)


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
        progress = arm_progress(geometry['rings'], self.arm_length)
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


class BatchedIPCActionFilter:
    """``IPCActionFilter`` for a batched world: one snapshot, candidate k run for every active slot at once.

    Candidates, scores and the noise-margin rule are those of ``IPCActionFilter``; a slot's candidate list is
    padded with its nominal action so all slots advance through the same K world steps per evaluation.
    Slots that are not active keep their nominal action during the candidate steps and are restored after.
    """

    def __init__(self, env, slots, *, strength_gain=1., horizon=1, value_model=None, value_margin=.01):
        self.env = env
        # Optional learned continuation value (outcome_value.OutcomeValue): candidates are ranked by the
        # predicted final-success probability at the end of their hold instead of the local progress score.
        self.value_model, self.value_margin = value_model, float(value_margin)
        self.horizon = int(horizon)       # each candidate is held for this many decisions before scoring
        cfg = env.cfg
        initial = env.positions()
        self.per_slot = []
        for i, s in enumerate(slots):
            idx = env._pickers[i]['anchor_idx']
            edges = s['sleeve'].edges
            self.per_slot.append(dict(
                sections=s['sleeve'], landmarks=s['landmarks'],
                arm_length=float(np.linalg.norm(np.diff(s['landmarks'], axis=0), axis=1).sum()),
                indices=idx, stiffness=cfg.constraint_strength * strength_gain * s['masses'][idx] / cfg.dt**2,
                garment=s['cell'].garment,
                edges=edges, rest=np.linalg.norm(initial[i][edges[:, 0]] - initial[i][edges[:, 1]], axis=1)))
        self.noise_margin = .0002
        self.replay_checked = False

    snapshot = IPCActionFilter.snapshot

    def _positions(self):
        return np.stack(self.env.positions())

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
        error = float(np.max(abs(self._positions() - snap['all_positions'])))
        if error > 1e-8 or int(e._world.frame()) != snap['frame']:
            raise RuntimeError(f'IPC snapshot did not restore exactly: {error} m')

    def _grip(self, i, positions):
        p = self.per_slot[i]
        target = self.env._anchor[i] + self.env._offsets[i]
        return float(np.linalg.norm((p['stiffness'][:, None] * (target - positions[p['indices']])).sum(0)))

    def _value(self, i, positions, geometry, action, branch):
        from outcome_value import state_features
        p, e = self.per_slot[i], self.env
        progress = e._last_progress[i]
        features = state_features(
            step=e._episode_step, tcp=e._anchor[i], finger=p['landmarks'][0], shoulder=p['landmarks'][2],
            grip=branch['grip'][-1], grip_window=branch['grip'], track=branch['track'][-1],
            track_window=branch['track'], upper=float(progress.upperarm_ratio), fore=float(progress.forearm_ratio),
            cuff=float(geometry['cuff_s']), prox=float(geometry['proximal_upper_fraction']),
            prox_before=branch['prox0'], upper_before=branch['upper0'], action=action, garment=p['garment'])
        return float(self.value_model(features)[0])

    def _score(self, i, positions, info, action, nominal, branch=None):
        p = self.per_slot[i]
        geometry = measure(p['sections'], positions, p['landmarks'])
        target = self.env._anchor[i] + self.env._offsets[i]
        force = float(np.linalg.norm((p['stiffness'][:, None] * (target - positions[p['indices']])).sum(0)))
        stretch = np.linalg.norm(positions[p['edges'][:, 0]] - positions[p['edges'][:, 1]], axis=1) / p['rest']
        tracking = float(info['tracking_error'])
        progress = arm_progress(geometry['rings'], p['arm_length'])
        score = progress - 2e-5 * max(force - 40., 0.) - .0002 * float(np.sum((action - nominal) ** 2))
        feasible = tracking <= .019 and np.quantile(stretch, .99) <= 2.25 and stretch.max() <= 4.
        row = dict(progress_score=float(score), progress_m=progress, gripper_N=force, tracking_m=tracking,
                   feasible=bool(feasible), wrapped=geometry['sleeve_wrapped'])
        if self.value_model is not None and branch is not None:
            row['value'] = self._value(i, positions, geometry, action, branch)
            score = row['value']
        if not feasible:
            score -= 1. + 100. * max(tracking - .019, 0.)
        row['score'] = float(score)
        return row

    def candidates(self, nominal, recent=None):
        nominal = np.clip(np.asarray(nominal, float), -1, 1)
        out = [nominal.copy(), nominal * .5, np.zeros(6)]
        no_rotation = nominal.copy()
        no_rotation[3:] = 0.
        out.append(no_rotation)
        for axis in range(3):
            for direction in (-1., 1.):
                c = nominal.copy()
                c[axis] += direction * .25
                out.append(np.clip(c, -1, 1))
        if recent is not None and len(recent):
            # Retreat: undo the recent approach direction (successful episodes often step back before success).
            back = -np.asarray(recent, float)[:, :3].mean(0) / self.env.cfg.max_translation
            for gain in (1., 2.):
                c = np.zeros(6)
                c[:3] = back * gain
                out.append(np.clip(c, -1, 1))
        unique = {}
        for c in out:
            unique.setdefault(tuple(c), c)
        return list(unique.values())

    def improve(self, nominals, active, recent=None):
        """Best candidate per active slot; returns the new action array and one diagnostic per active slot."""
        e = self.env
        nominals = np.clip(np.asarray(nominals, float), -1, 1)
        cands = {i: self.candidates(nominals[i], None if recent is None else recent.get(i)) for i in active}
        K = max(len(c) for c in cands.values())
        for i in active:
            cands[i] += [cands[i][0]] * (K - len(cands[i]))
        snap = self.snapshot()
        snap['all_positions'] = self._positions()
        start = {}
        if self.value_model is not None:            # branch-start quantities for the value's window features
            for i in active:
                g0 = measure(self.per_slot[i]['sections'], snap['all_positions'][i], self.per_slot[i]['landmarks'])
                start[i] = dict(prox0=float(g0['proximal_upper_fraction']),
                                upper0=float(e._last_progress[i].upperarm_ratio),
                                grip0=self._grip(i, snap['all_positions'][i]))
        rows = {i: [] for i in active}
        try:
            for k in range(K + (0 if self.replay_checked else 1)):
                kk = min(k, K - 1) if k < K else 0          # the extra pass replays candidate 0
                self.restore(snap)
                acts = nominals.copy()
                for i in active:
                    acts[i] = cands[i][kk]
                failed, worst = False, [0.] * len(acts)
                branch = {i: dict(start[i], grip=[start[i]['grip0']], track=[0.]) for i in start}
                for _ in range(self.horizon):
                    _, _, done, infos = e.step(acts)
                    failed = bool(done.any() or any(inf.get('sim_error') for inf in infos))
                    if failed:
                        break
                    worst = [max(w, float(inf['tracking_error'])) for w, inf in zip(worst, infos)]
                    if branch:
                        step_positions = self._positions()
                        for i in branch:
                            branch[i]['grip'].append(self._grip(i, step_positions[i]))
                            branch[i]['track'].append(float(infos[i]['tracking_error']))
                if not failed:                              # score feasibility on the worst step of the hold
                    for inf, w in zip(infos, worst):
                        inf['tracking_error'] = w
                positions = None if failed else self._positions()
                for i in active:
                    row = (dict(score=-np.inf, feasible=False, sim_error=True) if failed else
                           self._score(i, positions[i], infos[i], cands[i][kk], nominals[i], branch.get(i)))
                    if k < K:
                        row['action'] = cands[i][kk].tolist()
                        rows[i].append(row)
                    elif np.isfinite(row['score']) and np.isfinite(rows[i][0]['score']):
                        self.noise_margin = max(self.noise_margin, 2 * abs(row['score'] - rows[i][0]['score']))
                if failed:                                  # a world error resets the env; recover it before going on
                    self.restore(snap)
            self.replay_checked = True
        finally:
            self.restore(snap)
        out, diags = nominals.astype(np.float32).copy(), {}
        for i in active:
            scores = [r['score'] for r in rows[i]]
            best = int(np.argmax(scores))
            margin = max(self.noise_margin, self.value_margin) if self.value_model is not None else self.noise_margin
            if scores[best] < scores[0] + margin:
                best = 0
            out[i] = np.asarray(cands[i][best], np.float32)
            diags[i] = dict(selected=best, candidates=rows[i], noise_margin=self.noise_margin)
        return out, diags
