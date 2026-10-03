"""Finite collection -> matched learning -> physical effect-code study.

The prior pilot is separate. This study caps all new physical work, including
collection setup, at 8,000 charged decisions and four summed worker-hours.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from probe_outcome_repairs import write_json

REPO = Path(__file__).resolve().parents[2]
CPU = '/home/ge47gax/miniconda3/envs/curl/bin/python'
GPU = '/home/ge47gax/kun/genesis-world/.venv/bin/python'
PACKAGE = REPO / '.claude/worktrees/residual-rl/python'
OLD = REPO / 'output/uipc_manip/outcome_pilot_20261003'


def main(root):
    work = root / 'pipeline'
    work.mkdir(parents=True, exist_ok=True)
    old_status = work / 'status.json'
    previous = {}
    if old_status.exists():
        previous = json.loads(old_status.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Another study supervisor is still alive')
    state = dict(status='running', phase='starting', pid=os.getpid(), workers=[], actor_updated=False,
                 budget_limits=dict(decisions=8000, summed_worker_hours=4.), completed_stages=[])
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONPATH=f'/home/ge47gax/kun:{PACKAGE}')

    def costs():
        # Worker status is written at stage boundaries; budget.json is updated
        # during simulation. Use the larger counter, never count a worker twice.
        budgets = []
        for p in root.glob('*/status.json'):
            if p.parent == work:
                continue
            final = json.loads(p.read_text()).get('budget', {})
            live_path = p.with_name('budget.json')
            live = json.loads(live_path.read_text()) if live_path.exists() else {}
            budgets.append({k: max(final.get(k, 0), live.get(k, 0))
                            for k in ('charged_decisions', 'wall_seconds')})
        return dict(charged_decisions=sum(b.get('charged_decisions', 0) for b in budgets),
                    summed_worker_seconds=sum(b.get('wall_seconds', 0.) for b in budgets))

    def group(phase, jobs):
        state.update(phase=phase, workers=[], actual_budget=costs())
        processes = []
        for name, command, env in jobs:
            path = work / f'{name}.log'
            with path.open('w') as log:
                proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append(proc)
            state['workers'].append(dict(name=name, pid=proc.pid, command=command, log=str(path), returncode=None))
        write_json(work / 'status.json', state)
        while any(p.poll() is None for p in processes):
            for worker, proc in zip(state['workers'], processes):
                worker['returncode'] = proc.poll()
            state['actual_budget'] = costs()
            write_json(work / 'status.json', state)
            time.sleep(5)
        for worker, proc in zip(state['workers'], processes):
            worker['returncode'] = proc.returncode
        state['completed_stages'].append(dict(phase=phase, workers=state['workers']))
        write_json(work / 'status.json', state)
        if any(p.returncode for p in processes):
            raise RuntimeError(f'{phase} failed; no automatic retry')

    def probe(target, out, step, maximum=1000):
        return [GPU, '-u', 'scripts/wang_transfer/probe_outcome_repairs.py', '--target', str(target),
                '--features', str(OLD / 'features_canonical'), '--models', str(OLD / 'models_inverse_20261003'),
                '--out', str(out), '--root-step', str(step), '--max-decisions', str(maximum), '--gpu-hours', '.5']

    try:
        for body in (1032, 3041):
            status = root / f'collect_body{body}/status.json'
            while not status.exists() or json.loads(status.read_text())['status'] in ('initializing', 'running'):
                state.update(phase='waiting_for_initial_collectors', actual_budget=costs())
                write_json(work / 'status.json', state)
                time.sleep(5)
            if json.loads(status.read_text())['status'] != 'completed':
                raise RuntimeError('Initial branch collection failed; retain evidence and fix explicitly')
        garment_root = REPO / 'output/uipc_manip/cn_train_r1_20261001/r1'
        target = garment_root / 'cn_tcsc_083/off0_1032x7/body_2034_seed_2026092700/baseline.npz'
        command = probe(target, root / 'collect_body2034', 50)
        command += ['--collect-only', '--probe-count', '12', '--probe-steps', '24', '--replay-seeds', '20261003', '20261004']
        validation_status = root / 'collect_body2034/status.json'
        if validation_status.exists():
            state.update(phase='validation_body_collection', workers=previous.get('workers', []))
            while json.loads(validation_status.read_text())['status'] in ('initializing', 'running'):
                write_json(work / 'status.json', state)
                time.sleep(5)
            if json.loads(validation_status.read_text())['status'] != 'completed':
                raise RuntimeError('Validation collection failed')
        else:
            group('validation_body_collection', [('collect_body2034', command, dict(os.environ))])
        runs = [root / f'collect_body{body}' for body in (1032, 3041, 2034)]
        command = [CPU, '-u', 'scripts/wang_transfer/train_effect_codes.py', 'prepare',
                   '--features', str(OLD / 'features_canonical'), '--runs', *map(str, runs), '--out', str(root / 'data')]
        group('feature_preparation', [('prepare', command, cpu_env)])
        jobs = []
        for seed in (20261005, 20261006):
            command = [CPU, '-u', 'scripts/wang_transfer/train_effect_codes.py', 'train', '--data', str(root / 'data'),
                       '--out', str(root / f'models_{seed}'), '--seed', str(seed), '--updates', '1600']
            jobs.append((f'train_{seed}', command, cpu_env))
        group('matched_cpu_training', jobs)
        spent = costs()
        if spent['charged_decisions'] + 6000 > 8000 or spent['summed_worker_seconds'] + 3600 > 14400:
            raise RuntimeError('Physical evaluation reservation would exceed the new study budget')
        jobs = []
        for model_seed, replay, garment, body, step in [(20261005, 20261003, 'cn_tcsc_083', 2034, 50),
                                                       (20261006, 20261004, 'cn_tcsc_top558', 1032, 5)]:
            target = garment_root / f'{garment}/off0_1032x7/body_{body}_seed_2026092700/baseline.npz'
            command = probe(target, root / f'physical_{model_seed}', step, 3000)
            command += ['--effect-models', str(root / f'models_{model_seed}'), '--horizon', '24', '--replay-seeds', str(replay)]
            jobs.append((f'physical_{model_seed}', command, dict(os.environ)))
        group('physical_effect_realization', jobs)
        results = {str(seed): json.loads((root / f'physical_{seed}/results.json').read_text()) for seed in (20261005, 20261006)}
        state.update(status='completed', phase='finished', actual_budget=costs(), results=results)
        state['positive_repairs'] = sum(bool(row.get('improves_failed_baseline')) for rows in results.values() for row in rows)
        # A positive would require a separate verified absorption stage; never
        # silently train the base on short-horizon score or negative examples.
    except Exception as exc:
        state.update(status='error', error=repr(exc), actual_budget=costs())
        raise
    finally:
        write_json(work / 'status.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root.resolve())
