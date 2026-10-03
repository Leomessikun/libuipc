"""One capped IPC repair-discovery pilot using transferred interaction goals.

Three 24-decision proposals (including no edit), one selected replay, and one
450-decision continuation per method/replay. All setup and replay costs count
toward the global 5,000-decision/eight-hour cap. This is a development probe,
not an independent-garment benchmark or an EXPO implementation.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from dynamic_student import sha256
from ipc_action_filter import IPCActionFilter, arm_progress
from outcome_goals import InteractionCoordinates, arm_frame, future_indices, transport
from outcome_policy import OutcomeClient
from physical_sleeve import SleeveSections, measure, read_obj
from train_dynamic_student import PACKAGE

TARGET = Path('output/uipc_manip/cn_train_r1_20261001/r1/cn_tcsc_top558/off0_1032x7/body_1032_seed_2026092700/baseline.npz')
METHODS = ('geometry', 'action_noise', 'tcp_transfer', 'generic')


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    tmp.replace(path)


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, out, maximum, hours):
        self.out, self.maximum, self.hours = out, maximum, hours
        self.started, self.substeps, self.restores = time.monotonic(), 0, 0
        self.phase = 'setup'

    def state(self):
        return dict(substeps=self.substeps, decision_equivalents=self.substeps / 6.,
                    charged_decisions=math.ceil(self.substeps / 6), snapshot_restores=self.restores,
                    wall_seconds=time.monotonic() - self.started, phase=self.phase,
                    maximum_decisions=self.maximum, maximum_hours=self.hours)

    def before_substep(self):
        if self.substeps >= self.maximum * 6 or time.monotonic() - self.started >= self.hours * 3600:
            raise BudgetExhausted('Global physical-query budget exhausted')
        # Charge attempted substeps too, including a solver exception.
        self.substeps += 1
        if self.substeps % 6 == 1:
            write_json(self.out / 'budget.json', self.state())


def config_from_record(default, values):
    result = {}
    available = {f.name for f in fields(default)}
    for key, value in values.items():
        if key not in available:
            if key != 'max_translation':
                raise ValueError(f'Unrecognized saved config field: {key}')
            continue
        original = getattr(default, key)
        result[key] = config_from_record(original, value) if is_dataclass(original) else value
        if isinstance(original, Path):
            result[key] = Path(value)
    return replace(default, **result)


def reconstruct_cell(source, cfg):
    """Recover the *initial rest hang*, never use a deformed rollout as rest.

    Archived initial anchor offsets identify the rigid transform of the scaled
    hanging mesh. The settled state is regenerated in a new single-slot world;
    it is explicitly not claimed to reproduce a historical batched trajectory.
    """
    from uipc_manip.dressing_assets import DressingCell
    from uipc_manip.dressing_body import smplx_faces
    record = json.loads((source.parent / 'config.json').read_text())
    run = json.loads((source.parent.parent / 'run.json').read_text())
    with np.load(source, allow_pickle=False) as d, np.load(source.parent.parent / 'hang.npz') as h:
        metadata = json.loads(str(d['metadata_json']))
        if metadata['body'] != cfg.cells[0][1]:
            raise ValueError('Target record and configured body differ')
        hang = h[run['hang_key']] * record['placement']['fit_scale']
        ids, offsets = d['grasp_indices'], d['initial_grasp_offsets']
        a, b = hang[ids], offsets
        u, singular, vt = np.linalg.svd((a - a.mean(0)).T @ (b - b.mean(0)))
        rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        error = float(np.max(np.abs(a @ rotation - b)))
        if error > 1e-7 or singular[-1] < 1e-8:
            raise ValueError(f'Cannot reconstruct original rest hang: {error}')
        if not np.array_equal(d['human_faces'], smplx_faces()):
            raise ValueError('Full-body collider topology changed')
        cloth = hang @ rotation + d['tcp'][0]
        landmarks = {f'right_{k}': d[k].copy() for k in ('finger', 'elbow', 'shoulder')}
        cell = DressingCell(garment=cfg.cells[0][0], human=metadata['body'], cloth=cloth,
                            faces=d['faces'].copy(), grasp_idx=ids.copy(), picker_idx=ids.copy(),
                            opening_idx=d['opening_idx'].copy(), alignment_idx=d['opening_idx'][:2].copy(),
                            picker_pos=d['tcp'][0].astype(float), arm_points=d['arm_vertices'].copy(),
                            arm_faces=d['arm_faces'].copy(), human_points=d['human_vertices'].copy(),
                            landmarks=landmarks, pull_waypoints=np.stack([d['tcp'][0], d['shoulder']]),
                            opening_radius_mean=float(np.linalg.norm(cloth[d['opening_idx']]
                                                      - cloth[d['opening_idx']].mean(0), axis=1).mean()))
    return cell, dict(anchor_reconstruction_max_error_m=error, source_sha256=sha256(source),
                      config_sha256=sha256(source.parent / 'config.json'),
                      hang_sha256=sha256(source.parent.parent / 'hang.npz'),
                      historical_world='batched; new single-slot settling and rollout required')


def goal_sections(sections):
    """Request exact material fractions independently of success-test fallbacks.

    SleeveSections' conservative vertex-count heuristic can use .4 in place
    of .5 (62 vs 41 points exceeds its 1.5 ratio). Goals require the actual .5
    plane. Keep only connected closed loops near the sleeve, and keep the
    original fallback sections unchanged for the historical success criterion.
    """
    result = copy.copy(sections)
    vertices, faces = sections.vertices, sections.faces
    center = vertices[sections.cuff].mean(0)
    length = np.linalg.norm(vertices[sections.armhole].mean(0) - center)
    cuff_radius = np.linalg.norm(vertices[sections.cuff] - center, axis=1).mean()
    result.sections = [sections.sections[0]]
    result.fractions = [0.]
    for fraction in (.25, .5):
        section = sections._section(vertices, faces, center, sections.axis, length, fraction)
        if section is None:
            raise ValueError('No closed canonical goal section')
        edge, weight = section
        points = vertices[edge[:, 0]] * (1 - weight[:, None]) + vertices[edge[:, 1]] * weight[:, None]
        if (np.linalg.norm(points - points.mean(0), axis=1).mean() > 2 * cuff_radius
                or np.linalg.norm(points.mean(0) - (center + fraction * length * sections.axis)) > .5 * length):
            raise ValueError('Canonical goal section may have left the sleeve')
        result.sections.append(section); result.fractions.append(fraction)
    return result


def retrieve(manifest, current, feature, target_landmarks, horizon):
    """Shared source windows for geometry, generic goals and TCP transfer."""
    candidates = []
    _, target_frame, target_length = arm_frame(target_landmarks)
    for episode, record in enumerate(manifest['episodes']):
        if record['garment'] == manifest['validation_garment']:
            continue
        with np.load(record['path'], allow_pickle=False) as data:
            d = {k: data[k] for k in data.files}
        for t in range(0, len(d['actions']) - horizon + 1, 8):
            path = d['descriptor'][t:t + horizon + 1]
            try:
                goals = transport(path, current)
            except ValueError:
                continue
            # Initial relation and size, without using target outcomes.
            distance = float(np.mean((path[0, :, :3] - current[:, :3])**2)
                             + .01 * np.mean((path[0, :, 3:6] - current[:, 3:6])**2)
                             + 4. * np.mean((path[0, :, 6:8] - current[:, 6:8])**2)
                             + .1 * np.mean((path[0, :, 8] - current[:, 8])**2))
            _, source_frame, source_length = arm_frame(d['landmarks'])
            tcp = ((d['tools'][t:t + horizon + 1] - d['tools'][t]) @ source_frame.T
                   * target_length / source_length) @ target_frame
            candidates.append(dict(distance=distance, episode=episode, start=t,
                                   source=record['source'], source_garment=record['garment'],
                                   goals=goals, generic=feature + d['features'][t:t + horizon + 1] - d['features'][t],
                                   tcp=tcp, rotations=d['actions'][t:t + horizon, 5]))
    candidates.sort(key=lambda x: x['distance'])
    selected = []
    for candidate in candidates:
        # A second source episode provides a distinct proposal, not an adjacent window.
        if any(c['episode'] == candidate['episode'] for c in selected):
            continue
        selected.append(candidate)
        if len(selected) == 2:
            return selected
    raise ValueError('Fewer than two supported training-only source paths')


def main(args):
    if args.max_decisions > 5000 or args.gpu_hours > 8 or args.horizon != 24:
        raise ValueError('This authorization is limited to the documented pilot budget')
    if len(set(args.replay_seeds)) != len(args.replay_seeds) or not set(args.replay_seeds) <= {20261003, 20261004}:
        raise ValueError('Use the two predeclared replay seeds, once each')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(PACKAGE))
    from uipc_manip.dressing_env import DressingConfig, GenesisIPCDressingEnv
    from uipc_manip.obs import ObsSpec
    budget = Budget(args.out, args.max_decisions, args.gpu_hours)
    env, client = None, None
    results = []
    status = 'initializing'

    def stop(signum, frame):
        raise BudgetExhausted(f'Stopped by signal {signum}; incomplete runs are censored')

    signal.signal(signal.SIGTERM, stop)
    timer = threading.Timer(args.gpu_hours * 3600, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True; timer.start()

    class ProbeEnv(GenesisIPCDressingEnv):
        branch = False

        def _sim_step(self):
            budget.before_substep()
            return super()._sim_step()

        def reset(self, seeds=None):
            if self.branch:
                # An invalid branch must not reset or overwrite its root world.
                return np.zeros((1, self.obs_dim), np.float32)
            return super().reset(seeds)

    try:
        saved = json.loads((args.target.parent / 'config.json').read_text())
        cfg = config_from_record(DressingConfig(), saved['config'])
        cfg = replace(cfg, cells=(tuple(cfg.cells[0]),), workspace=str((args.out / 'scene').resolve()),
                      horizon=2000, contact_force_readout=False, live=replace(cfg.live, body=replace(cfg.live.body, device='cpu')))
        cell, reconstruction = reconstruct_cell(args.target, cfg)
        template = Path('output/uipc_manip/clothesnet_assets/raw') / f'{cell.garment}.obj'
        vertices, faces = read_obj(template)
        if not np.array_equal(faces, cell.faces):
            raise ValueError('Target sleeve template topology differs')
        sections = SleeveSections(vertices, faces, cell.opening_idx)
        landmarks = np.stack([cell.finger, cell.elbow, cell.shoulder])
        canonical_sections = goal_sections(sections)
        coordinates = InteractionCoordinates(canonical_sections, landmarks)
        manifest = json.loads((args.features / 'manifest.json').read_text())
        if any(r['garment'] == cell.garment for r in manifest['episodes']):
            raise ValueError('Target garment contaminated the source dataset')
        run = dict(arguments=vars(args), target_garment=cell.garment, target_body=cell.human,
                   reconstruction=reconstruction, methods=METHODS, replay_seeds=args.replay_seeds,
                   feature_manifest_sha256=sha256(args.features / 'manifest.json'),
                   code_sha256=sha256(__file__), environment_sha256=sha256(PACKAGE / 'uipc_manip/dressing_env.py'),
                   models={k: sha256(args.models / f'{k}.pt') for k in ('geometry', 'generic')},
                   rest_of_config_matches_source=True, contact_force_readout=False,
                   goal_sections=canonical_sections.describe(), success_sections=sections.describe(),
                   success='armhole upper fraction >=0.7, all interior sections wrapped, 20 zero-action hold decisions, valid grasp',
                   notes='Development training cell; two RNG/numerical replays of one root are not independent task samples')
        write_json(args.out / 'run.json', run)
        client = OutcomeClient(args.models)
        factory = SimpleNamespace(cfg=cfg.live, build=lambda garment, human: copy.deepcopy(cell))
        env = ProbeEnv(cfg, num_envs=1, cell_factory=factory)
        obs = env.reset([20261003])
        tri = cell.cloth[cell.faces]
        area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        masses = np.zeros(len(cell.cloth)); np.add.at(masses, cell.faces.ravel(), np.repeat(area / 3, 3))
        masses *= 2 * cfg.cloth_thickness * cfg.cloth_density
        snapshotter = IPCActionFilter(env, sections, masses)
        rest_lengths = np.linalg.norm(cell.cloth[sections.edges[:, 0]] - cell.cloth[sections.edges[:, 1]], axis=1)
        spec = ObsSpec(cfg.point_budget)

        def observe(observation):
            pos, flags, valid, extra = spec.unpack_numpy(observation[0])
            actions, feature = client.request('observe', pos=pos[valid], flags=flags[valid], tool=extra[:3])
            return actions[0], feature

        def state(info=None):
            positions = env.positions()[0]
            geo = measure(sections, positions, landmarks)
            stretches = np.linalg.norm(positions[sections.edges[:, 0]] - positions[sections.edges[:, 1]], axis=1) / rest_lengths
            tracking = (info['tracking_error'] if info else env._tracking_error(0, [positions]))
            target = env._anchor[0] + env._offsets[0]
            force = float(np.linalg.norm((snapshotter.stiffness[:, None] * (target - positions[snapshotter.indices])).sum(0)))
            valid = bool(tracking <= .02 and np.quantile(stretches, .99) <= 2.25 and stretches.max() <= 4. and force <= 1000.)
            return dict(progress_m=arm_progress(geo['rings'], coordinates.length),
                        armhole_upper_fraction=geo['armhole_upper_fraction'], wrapped=geo['sections_wrapped'],
                        reached=bool(geo['sections_wrapped'] and geo['armhole_upper_fraction'] >= .7),
                        valid=valid, tracking_m=float(tracking), gripper_N=force,
                        edge_p99=float(np.quantile(stretches, .99)), edge_max=float(stretches.max()))

        def transition(action):
            next_obs, _, done, infos = env.step(np.asarray(action)[None])
            if infos[0].get('sim_error') or done.any():
                return next_obs, dict(valid=False, sim_error=infos[0].get('error', 'unexpected time limit'))
            row = state(infos[0])
            with (args.out / 'transitions.jsonl').open('a') as log:
                log.write(json.dumps(dict(phase=budget.phase, decision=budget.substeps / 6., **row)) + '\n')
            return next_obs, row

        base, feature = observe(obs)
        observation_history = [obs[0].copy()]
        if not state()['valid']:
            raise ValueError('Regenerated initial state is physically invalid')
        budget.phase = 'roll_in'
        for _ in range(args.root_step):
            obs, row = transition(base)
            if not row['valid']:
                raise ValueError('Nominal rollout invalid before predeclared repair root')
            base, feature = observe(obs)
            observation_history.append(obs[0].copy())
        root_state = state()
        root = snapshotter.snapshot(); client.request('save')
        root_anchor = env._anchor[0].copy()
        descriptor = coordinates.describe(env.positions()[0])
        sources = retrieve(manifest, descriptor, feature, landmarks, args.horizon)
        write_json(args.out / 'root.json', dict(geometry=root_state, frame=root['frame'],
                    state=args.root_step, target_sections=sections.describe(),
                    sources=[{k: v for k, v in c.items() if k not in ('goals', 'generic', 'tcp', 'rotations')} for c in sources]))
        np.savez_compressed(args.out / 'proposals.npz', descriptor=descriptor,
                            geometry=np.stack([s['goals'] for s in sources]),
                            generic=np.stack([s['generic'] for s in sources]), tcp=np.stack([s['tcp'] for s in sources]))
        env.branch = True

        def restore(seed):
            budget.restores += 1
            snapshotter.restore(root)
            env.rngs[0] = np.random.default_rng(seed)
            return client.request('restore')[0][0]

        def branch(method, candidate, seed, suffix=False):
            nominal = restore(seed)
            source = sources[candidate - 1] if candidate and method != 'exploration' else None
            repair_steps = args.probe_steps if method == 'exploration' else args.horizon
            rng = np.random.default_rng(seed + candidate * 101)
            knots = rng.normal(0., .3, (3, 4))
            observations, actions, geometries = [obs[0].copy()], [], []
            descriptors, tools = [descriptor.copy()], [root_anchor.copy()]
            row = root_state.copy()
            budget.phase = f'{method}/seed{seed}/candidate{candidate}/' + ('verification' if suffix else 'search')
            first_success, held = None, 0
            for t in range(repair_steps + (450 if suffix else 0)):
                if suffix and first_success is not None:
                    action = np.zeros(6, np.float32)
                elif t >= repair_steps or candidate == 0:
                    action = nominal
                elif method == 'exploration':
                    if candidate == 1:
                        correction = np.clip(-nominal[[0, 1, 2, 5]], -.5, .5)
                    elif candidate == 2:
                        correction = np.clip(-2 * nominal[[0, 1, 2, 5]], -.5, .5)
                    else:
                        phase = t * 2 / max(repair_steps - 1, 1)
                        lo = min(int(phase), 1); fraction = phase - lo
                        correction = np.clip((1 - fraction) * knots[lo] + fraction * knots[lo + 1], -.5, .5)
                    action = nominal.copy()
                    action[[0, 1, 2, 5]] = np.clip(action[[0, 1, 2, 5]] + correction, -1., 1.)
                elif method == 'geometry':
                    chunk = 1 if args.one_step_goals else 8
                    if t % chunk == 0:
                        remaining = 1 if args.one_step_goals else args.horizon - t
                        chunk_actions, _ = client.request('geometry', goals=source['goals'][future_indices(t, remaining)],
                                                         landmarks=landmarks, remaining=remaining)
                    action = chunk_actions[t % chunk]
                elif method == 'generic':
                    chunk = 1 if args.one_step_goals else 8
                    if t % chunk == 0:
                        remaining = 1 if args.one_step_goals else args.horizon - t
                        chunk_actions, _ = client.request('generic', goals=source['generic'][future_indices(t, remaining)],
                                                         remaining=remaining)
                    action = chunk_actions[t % chunk]
                elif method == 'tcp_transfer':
                    action = np.zeros(6, np.float32)
                    action[:3] = (root_anchor + source['tcp'][t + 1] - env._anchor[0]) / cfg.max_translation
                    action[5] = source['rotations'][t]
                    action = np.clip(action, -1., 1.)
                else:
                    action = nominal.copy()
                    phase = t * 2 / max(args.horizon - 1, 1)
                    lo = min(int(phase), 1); fraction = phase - lo
                    action[[0, 1, 2, 5]] += (1 - fraction) * knots[lo] + fraction * knots[lo + 1]
                    action = np.clip(action, -1., 1.)
                next_obs, row = transition(action)
                observations.append(next_obs[0].copy()); actions.append(action.copy()); geometries.append(row)
                actual = (coordinates.describe(env.positions()[0]) if not row.get('sim_error')
                          else np.full_like(descriptor, np.nan))
                descriptors.append(actual); tools.append(env._anchor[0].copy())
                if source is not None and t < args.horizon and not row.get('sim_error'):
                    desired = source['goals'][t + 1]
                    row['goal_center_error_m'] = float(np.linalg.norm(actual[:, :3] - desired[:, :3], axis=1).mean()
                                                      * coordinates.length)
                    row['goal_normal_error'] = float(np.linalg.norm(actual[:, 3:6] - desired[:, 3:6], axis=1).mean())
                if not row['valid']:
                    break
                if suffix:
                    if first_success is None and row['reached']:
                        first_success = t + 1
                    elif first_success is not None:
                        if not row['reached']:
                            row['hold_failed'] = True
                            break
                        held += 1
                        if held == 20:
                            break
                nominal, _ = observe(next_obs)
            score = float(row.get('progress_m', -1.)) if row['valid'] else -1e6
            result = dict(method=method, seed=seed, candidate=candidate, verification=suffix,
                          decisions=len(actions), score=score, final=row, first_success=first_success,
                          success=bool(suffix and held == 20 and row['valid']), budget=budget.state())
            name = f'{method}_{seed}_{candidate}_' + ('verification' if suffix else 'search')
            np.savez_compressed(args.out / (name + '.npz'), obs=np.asarray(observations), actions=np.asarray(actions),
                                history_obs=np.asarray(observation_history[-4:-1]),
                                descriptor=np.asarray(descriptors), tcp=np.asarray(tools), landmarks=landmarks,
                                grasp_valid=np.array([r['valid'] for r in geometries]),
                                geometry_json=json.dumps(geometries), metadata_json=json.dumps(result))
            with (args.out / 'branches.jsonl').open('a') as log:
                log.write(json.dumps(result) + '\n')
            print('[branch] ' + json.dumps(result), flush=True)
            return result

        status = 'running'
        write_json(args.out / 'status.json', dict(status=status, budget=budget.state(), policy_updated=False))
        for seed in args.replay_seeds:
            if args.collect_only:
                for candidate in range(args.probe_count):
                    results.append(branch('exploration', candidate, seed))
                    write_json(args.out / 'results.json', results)
                continue
            # Balanced order: reverse in the second replay to reduce time/load effects.
            for method in (METHODS if seed == 20261003 else METHODS[::-1]):
                trials = [branch(method, candidate, seed) for candidate in range(3)]
                best = max(range(3), key=lambda k: trials[k]['score'])
                if trials[best]['score'] <= -1e6:
                    result = dict(method=method, seed=seed, success=False, status='no_valid_repair',
                                  trials=trials, budget=budget.state())
                else:
                    result = branch(method, best, seed, suffix=True)
                    result['status'] = 'verified' if result['success'] else 'verification_failed'
                results.append(result)
                write_json(args.out / 'results.json', results)
        status = 'completed'
    except BudgetExhausted as exc:
        status = 'budget_censored'
        write_json(args.out / 'interruption.json', dict(reason=str(exc), phase=budget.phase))
    except Exception as exc:
        status = 'error'
        write_json(args.out / 'error.json', dict(error=repr(exc), phase=budget.phase))
        raise
    finally:
        timer.cancel()
        write_json(args.out / 'status.json', dict(status=status, results=len(results), budget=budget.state(),
                                                  policy_updated=False))
        if env is not None:
            env.close()
        if client is not None:
            client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, default=TARGET)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root-step', type=int, default=20)
    p.add_argument('--replay-seeds', type=int, nargs='+', default=[20261003, 20261004])
    p.add_argument('--horizon', type=int, default=24)
    p.add_argument('--max-decisions', type=int, default=5000)
    p.add_argument('--gpu-hours', type=float, default=8.)
    p.add_argument('--collect-only', action='store_true', help='Collect distinct short action consequences; no repair selection or success labels')
    p.add_argument('--probe-count', type=int, default=16)
    p.add_argument('--probe-steps', type=int, default=8)
    p.add_argument('--one-step-goals', action='store_true', help='Realize the next interaction goal and replan after every decision')
    main(p.parse_args())
