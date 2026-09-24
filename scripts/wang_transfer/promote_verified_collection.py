"""Wait for the controlled pilot, audit it, then hand collection to a verified recipe.

No model/API calls. Keeps the original data intact and changes the recipe only
after independent geometry, stationary-hold, grasp and deformation checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from collect_dataset_background import atomic_json
from physical_sleeve import DEFAULT_OBJ, audit, read_obj

ROOT = Path(__file__).resolve().parents[2]


def process_matches(pid, script):
    try:
        return script.encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--pilot-pid', type=int, required=True)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--target', type=int, default=500)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    status_path = a.out/'handoff_status.json'

    def status(phase, **extra):
        atomic_json(status_path, dict(pid=os.getpid(), phase=phase, updated_unix=time.time(), **extra))

    status('waiting_for_controlled_pilot')
    while process_matches(a.pilot_pid, 'collect_better_rollouts.py'):
        if (a.out/'STOP').exists():
            status('cancelled_before_handoff')
            return
        time.sleep(15)
    rest, faces = read_obj(DEFAULT_OBJ)
    rest *= 4.
    reports = []
    for path in sorted(a.pilot.glob('body_*/*.npz')):
        import numpy as np
        with np.load(path) as data:
            metadata = json.loads(str(data['metadata_json']))
        row = audit(path, rest, faces, 20, endpoint='proximal_sleeve')
        row['profile'] = metadata['profile']['name']
        row['passes_all'] = bool(row['accepted'] and row['edge_p99_peak_vs_initial'] <= 2.25
                                and row['edge_max_peak_vs_initial'] <= 4.)
        reports.append(row)
    atomic_json(a.out/'pilot_audit.json', reports)
    eligible = {}
    for name in ('legacy', 'single_voxel'):
        eligible[name] = {r['body'] for r in reports if r['profile'] == name and r['passes_all']}
    # Prefer removing the redundant mixed-segment voxel filter, provided its
    # validated body coverage includes the unchanged-input control's coverage.
    chosen = 'single_voxel' if len(eligible['single_voxel']) >= 2 and eligible['single_voxel'] >= eligible['legacy'] else 'legacy'
    if len(eligible[chosen]) < 2:
        status('pilot_not_sufficient', passed_bodies={k: sorted(v) for k, v in eligible.items()},
               note='Original collection remains running; no automatic recipe change.')
        return
    recipe = dict(name=chosen)
    if chosen == 'single_voxel':
        recipe['bridge_voxel'] = 0.
    recipe_path = a.out/'verified_profiles.json'
    atomic_json(recipe_path, [recipe, recipe])
    status('verified_waiting_for_old_batch', profile=chosen, passed_bodies=sorted(eligible[chosen]))
    # The user's request authorizes continuing collection with verified fixes.
    # STOP is consumed between batches, so no active trajectory is discarded.
    (a.old/'STOP').write_text('Verified anatomical-stop collection handoff; see new handoff_status.json.\n')
    old_status = json.loads((a.old/'status.json').read_text())
    old_pid = old_status['pid']
    while process_matches(old_pid, 'collect_dataset_background.py'):
        time.sleep(15)
    seed_runs = sorted(a.old.glob('batch_*_body_*'))
    seed_runs += [a.pilot, ROOT/'output/uipc_manip/fmvp_body14053_physical_stop090_20260924']
    command = [sys.executable, str(ROOT/'scripts/wang_transfer/collect_dataset_background.py'),
               '--out', str(a.out), '--target', str(a.target), '--max-attempts', '3000',
               '--bodies', *map(str, range(14045, 14085)), '--variants', 'half', 'quarter',
               '--profiles-json', str(recipe_path), '--endpoint', 'proximal_sleeve', '--steps', '750',
               '--seed-runs', *map(str, seed_runs)]
    atomic_json(a.out/'launch_command.json', command)
    with (a.out/'collector.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    status('new_collection_running', collector_pid=child.pid, profile=chosen,
           passed_bodies=sorted(eligible[chosen]), target=a.target,
           note='Existing episodes are re-audited against the new endpoint; old files remain intact.')


if __name__ == '__main__':
    main()
