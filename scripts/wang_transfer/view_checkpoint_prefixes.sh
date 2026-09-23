#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="${1:-$repo_dir/output/uipc_manip/fmvp_checkpoint_prefixes_20260923/manifest.json}"
python_bin="${GENESIS_PYTHON:-/home/ge47gax/kun/genesis-world/.venv/bin/python}"
mapfile -t episodes < <(python3 - "$manifest" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
selected = [r for r in manifest['episodes'] if r['selected']]
selected.sort(key=lambda r: (r['stage'] != 'upperarm', r['body'], r['seed'], r['variant']))
for record in selected:
    print(record['path'])
PY
)
if ((${#episodes[@]} == 0)); then
  echo "No selected episodes in $manifest" >&2
  exit 1
fi
exec env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$python_bin" -u \
  "$repo_dir/scripts/wang_transfer/view_rollouts_genesis.py" "${episodes[@]}" "${@:2}"
