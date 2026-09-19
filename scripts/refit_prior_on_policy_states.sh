#!/usr/bin/env bash
# Refit the behavior flow on policy-visited states labelled with the measured recovery,
# then reverse the same macros through the new prior and compare with the old one.
set -euo pipefail

DATASET=${DATASET:-output/uipc_manip/policy_labels_20260919/matrix25}
ROOT=${ROOT:-output/uipc_manip/policy_labels_20260919}
CHECKPOINT=${CHECKPOINT:-output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt}
OLD_PRIOR=${OLD_PRIOR:-output/uipc_manip/fql_pretrain_20260917/continue_a100_s17_30k/final.pt}
STEPS=${STEPS:-10000}
BRANCHES=${BRANCHES:-"output/uipc_manip/decision_branches_20260918/t26_14046 output/uipc_manip/decision_branches_20260918/t392_14046"}

export PYTHONPATH=build_raw/python/src:python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib
PY=${PY:-/home/ge47gax/kun/genesis-world/.venv/bin/python}

echo "[refit] training on $DATASET for $STEPS updates"
$PY -m uipc_manip.train_fql train --dataset "$DATASET" --reference "$CHECKPOINT" \
    --steps "$STEPS" --out "$ROOT/train" 2>&1 | tail -6

echo "[refit] reversing the recovery macros through the refitted prior"
$PY scripts/flow_reversal_macros.py --checkpoint "$ROOT/train/final.pt" \
    --branches $BRANCHES --out "$ROOT/macro_audit.json" 2>&1 | tail -10

echo "[refit] the same macros through the old prior, for comparison"
$PY scripts/flow_reversal_macros.py --checkpoint "$OLD_PRIOR" \
    --branches $BRANCHES --out "$ROOT/macro_audit_old.json" 2>&1 | tail -8
