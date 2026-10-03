"""Finite consequence-learning continuation, with visible stage/PID records.

Requires completed branch extraction. Trains two CPU seeds, then runs two
bounded IPC mechanism probes. It never resumes M4, restarts failed workers,
changes the base r1 weights, or exceeds the remaining original pilot budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from probe_outcome_repairs import write_json

CPU = '/home/ge47gax/miniconda3/envs/curl/bin/python'
GPU = '/home/ge47gax/kun/genesis-world/.venv/bin/python'
REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / '.claude/worktrees/residual-rl/python'


def main(root):
    work = root / 'counterfactual_pipeline'
    work.mkdir(parents=True, exist_ok=False)
    state = dict(status='running', phase='starting', pid=os.getpid(), workers=[], r1_updated=False)
    cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONPATH=f'/home/ge47gax/kun:{PACKAGE}')

    def run_group(phase, jobs):
        state.update(phase=phase, workers=[])
        processes = []
        for name, command, env in jobs:
            log = work / f'{name}.log'
            with log.open('w') as stream:
                proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            processes.append(proc)
            state['workers'].append(dict(name=name, pid=proc.pid, command=command, log=str(log), returncode=None))
        write_json(work / 'status.json', state)
        print(f'[pipeline] {phase}: {[p.pid for p in processes]}', flush=True)
        while any(p.poll() is None for p in processes):
            for record, proc in zip(state['workers'], processes):
                record['returncode'] = proc.poll()
            write_json(work / 'status.json', state)
            time.sleep(5)
        for record, proc in zip(state['workers'], processes):
            record['returncode'] = proc.returncode
        write_json(work / 'status.json', state)
        if any(p.returncode for p in processes):
            raise RuntimeError(f'A {phase} worker failed; see its log')

    try:
        manifest = json.loads((root / 'features_counterfactual/manifest.json').read_text())
        if manifest.get('counterfactual_episodes') != 32 or manifest['horizon'] != 1:
            raise ValueError('Expected the declared 32-branch, one-step learning diagnostic')
        train = []
        for seed in (20261003, 20261004):
            train.append((f'train_{seed}', [CPU, '-u', 'scripts/wang_transfer/train_outcome_goals.py', 'train',
                          '--features', str(root / 'features_counterfactual'), '--out', str(root / f'models_inverse_{seed}'),
                          '--chunk', '1', '--seed', str(seed), '--updates', '1200'], cpu_env))
        run_group('cpu_training', train)
        spent = [json.loads(p.read_text()).get('budget', {}) for p in root.glob('*/status.json')]
        charged = sum(b.get('charged_decisions', 0) for b in spent)
        seconds = sum(b.get('wall_seconds', 0.) for b in spent)
        cap = min(1500, (5000 - charged) // 2)
        hours = min(.5, (8 * 3600 - seconds) / 7200)
        if cap < 1 or hours <= 0:
            raise RuntimeError('No original pilot budget remains')
        state['budget'] = dict(prior_charged_decisions=charged, prior_worker_seconds=seconds,
                               new_cap_per_worker=cap, new_hours_per_worker=hours,
                               maximum_total_decisions=charged + 2 * cap)
        jobs = []
        for seed in (20261003, 20261004):
            command = [GPU, '-u', 'scripts/wang_transfer/probe_outcome_repairs.py',
                       '--features', str(root / 'features_canonical'),  # original cross-garment goal bank
                       '--models', str(root / f'models_inverse_{seed}'), '--out', str(root / f'inverse_seed{seed}'),
                       '--root-step', '5', '--replay-seeds', str(seed), '--one-step-goals',
                       '--max-decisions', str(cap), '--gpu-hours', str(hours)]
            jobs.append((f'ipc_{seed}', command, dict(os.environ)))
        run_group('ipc_goal_realization', jobs)
        outcomes = {str(seed): json.loads((root / f'inverse_seed{seed}/status.json').read_text())
                    for seed in (20261003, 20261004)}
        state.update(status='completed', phase='finished', outcomes=outcomes)
    except Exception as exc:
        state.update(status='error', error=repr(exc))
        raise
    finally:
        write_json(work / 'status.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root.resolve())
