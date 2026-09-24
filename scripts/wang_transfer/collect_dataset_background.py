"""Standalone checkpoint collection. No LLM, API client, or agent polling.

Writes status.json and manifest.json after every batch. Creating STOP in the
output directory stops collection after the current batch. Existing samples
are re-audited; rejected attempts are preserved. Resume with the same output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from physical_sleeve import DEFAULT_OBJ, audit, read_obj


ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/ge47gax/kun/genesis-world/.venv/bin/python'


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--target', type=int, default=500)
    p.add_argument('--max-attempts', type=int, default=2000)
    p.add_argument('--wait-pid', type=int)
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--seed-runs', type=Path, nargs='*', default=[])
    p.add_argument('--bodies', type=int, nargs='+')
    p.add_argument('--variants', nargs='+', default=['baseline', 'half', 'quarter'])
    p.add_argument('--profiles-json', type=Path)
    p.add_argument('--endpoint', choices=('legacy_ratio', 'proximal_sleeve'), default='legacy_ratio')
    p.add_argument('--steps', type=int, default=650)
    a = p.parse_args()
    os.chdir(ROOT)
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.target < 1 or a.max_attempts < a.target:
        raise ValueError('Need a positive target and a sufficient attempt budget')
    config_path = a.out / 'config.json'
    config = dict(target=a.target, max_attempts=a.max_attempts, seed=a.seed,
                  bodies=a.bodies or list(range(14045, 14061)), variants=a.variants,
                  hold_decisions=20, max_edge_p99=2.25, max_single_edge_ratio=4.,
                  deformation_note='Empirical simulation guardrails, not calibrated textile limits.',
                  start_jitter_mm=[7., 5., 7.], force_cutoff_N=1000.,
                  seed_runs=[str(x.resolve()) for x in a.seed_runs],
                  policy='/home/ge47gax/Desktop/fmvp_sim.pt', simulator='Genesis+IPC',
                  agent_or_api_calls=False)
    if a.endpoint != 'legacy_ratio' or a.profiles_json is not None or a.steps != 650:
        config.update(endpoint=a.endpoint, endpoint_threshold=.9 if a.endpoint == 'proximal_sleeve' else .7,
                      steps=a.steps)
    profile_path = None
    if a.profiles_json is not None:
        from rollout_controls import load_profiles, profile_dict
        profiles = [profile_dict(x) for x in load_profiles(a.profiles_json, len(a.variants))]
        config['profiles'] = profiles
        profile_path = a.out / 'profiles.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Existing collection config differs; choose a new output directory')
    atomic_json(config_path, config)
    if profile_path is not None:
        atomic_json(profile_path, config['profiles'])
    rows = []
    checked = set()
    state_path = a.out / 'manifest.json'
    if state_path.exists():
        old = json.loads(state_path.read_text())
        rows = old['episodes']
        checked = set(old.get('checked_paths', []))
    ledger_path = a.out / 'attempts.jsonl'
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
    attempts = len(ledger)
    rest, faces = read_obj(DEFAULT_OBJ)
    rest *= 4.
    started = time.time()

    def status(phase, **extra):
        manifest = dict(target=a.target, accepted=len(rows), transitions=sum(r['transitions'] for r in rows),
                        body_ids=sorted({r['body'] for r in rows}), episodes=rows,
                        checked_paths=sorted(checked), quality_class='single-sleeve simulation; material uncalibrated')
        atomic_json(state_path, manifest)
        atomic_json(a.out / 'status.json', dict(pid=os.getpid(), phase=phase, target=a.target,
                    accepted=len(rows), attempts=attempts, updated_unix=time.time(),
                    transitions=manifest['transitions'], body_ids=manifest['body_ids'],
                    endpoint=a.endpoint,
                    elapsed_s=round(time.time() - started), **extra))

    def inspect(run):
        nonlocal attempts
        for path in sorted(run.glob('body_*/*.npz')):
            key = str(path.resolve())
            if key in checked:
                continue
            try:
                result = audit(path, rest, faces, 20, endpoint=a.endpoint)
                result['collection_accepted'] = bool(result['accepted']
                    and result['edge_p99_peak_vs_initial'] <= config['max_edge_p99']
                    and result['edge_max_peak_vs_initial'] <= config['max_single_edge_ratio'])
                if result['collection_accepted'] and len(rows) < a.target:
                    rows.append(result)
            except Exception as exc:
                result = dict(path=key, collection_accepted=False, audit_error=repr(exc))
            checked.add(key)
            attempts += 1
            with ledger_path.open('a') as f:
                f.write(json.dumps(result) + '\n')
            status('auditing')

    # Avoid overlapping the controlled pilot. A reused PID is not a reason
    # to wait for an unrelated process.
    status('starting')
    if a.wait_pid:
        while True:
            proc = Path(f'/proc/{a.wait_pid}/cmdline')
            if not proc.exists() or b'collect_better_rollouts.py' not in proc.read_bytes():
                break
            if (a.out / 'STOP').exists():
                status('stopped')
                return
            status('waiting_for_pilot', waiting_pid=a.wait_pid)
            time.sleep(15)
    for run in a.seed_runs:
        inspect(run.resolve())
    for run in sorted(a.out.glob('batch_*')):
        inspect(run)

    batch = len(list(a.out.glob('batch_*')))
    consecutive_errors = 0
    while len(rows) < a.target and attempts < a.max_attempts:
        if (a.out / 'STOP').exists():
            status('stopped')
            return
        if shutil.disk_usage(a.out).free < 20 * 1024**3:
            status('stopped_low_disk')
            return
        rng = np.random.default_rng(a.seed + batch)
        body = config['bodies'][batch % len(config['bodies'])]
        # Each batch has a distinct start. The collector recomputes a legal
        # full-body placement and rejects starts with no collision-free pose.
        offset = np.array([0., 5., 0.]) + rng.uniform(-1, 1, 3) * config['start_jitter_mm']
        destination = a.out / f'batch_{batch:05d}_body_{body}'
        cmd = [PYTHON, str(ROOT / 'scripts/wang_transfer/collect_better_rollouts.py'),
               '--hang', str(ROOT / 'output/uipc_manip/fmvp_better_rollouts_20260923/hang2.npz'),
               '--hang-key', 'k300', '--bodies', str(body), '--variants', *config['variants'],
               '--seed', str(a.seed + batch), '--steps', str(a.steps), '--hold', '20',
               '--success', '.7', '--success-geometry', 'physical_sleeve', '--slow-along', '.85',
               '--yaw', '267', '--rotation', 'fmvp', '--collision-geometry', 'full_body',
               '--placement-offset-mm', *[str(float(x)) for x in offset],
               '--abort-gripper-force', '1000', '--cloth-density', '750', '--cloth-strain-rate', '10',
               '--out', str(destination)]
        if a.endpoint == 'proximal_sleeve':
            cmd += ['--stop-proximal-upper', '.9']
        if profile_path is not None:
            cmd += ['--profiles-json', str(profile_path)]
        atomic_json(a.out / 'current_command.json', dict(command=cmd, body=body, offset_mm=offset.tolist()))
        status('collecting', batch=batch, body=body)
        before = attempts
        with (a.out / f'batch_{batch:05d}.log').open('w') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                code = -1
        inspect(destination)
        consecutive_errors = consecutive_errors + 1 if code != 0 and attempts == before else 0
        if attempts == before:
            attempts += len(config['variants'])
            with ledger_path.open('a') as f:
                for variant in config['variants']:
                    f.write(json.dumps(dict(batch=batch, body=body, variant=variant,
                                            collection_accepted=False, returncode=code,
                                            reason='No saved trajectory; see batch log')) + '\n')
        batch += 1
        status('between_batches', last_returncode=code)
        if consecutive_errors >= 3:
            status('stopped_repeated_process_errors', last_returncode=code)
            return
    status('complete' if len(rows) >= a.target else 'attempt_budget_exhausted')


if __name__ == '__main__':
    main()
