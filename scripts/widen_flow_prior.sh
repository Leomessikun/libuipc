#!/usr/bin/env bash
# Widen the dressing behavior prior and re-audit whether it now contains the helpful action.
#
# The reversal audit found that the recovery directions which beat the policy need noise
# beyond every sample the prior draws. The answer to that is more and wider data, not a
# different learner, so this chains the four steps that produce it:
#
#   1. collect expert episodes over several arm-pose regions with the repaired teacher,
#   2. replay them to record observations and explicit successors,
#   3. refit the behavior flow on the result,
#   4. reverse the same recovery macros through the new prior and compare.
#
# Every stage writes under one dated root and refuses to overwrite an existing one.
set -euo pipefail

ROOT=${ROOT:-output/uipc_manip/wide_prior_$(date +%Y%m%d)}
CHECKPOINT=${CHECKPOINT:-output/uipc_manip/dressing_redesign_20260916/warm_sac/checkpoints/checkpoint_00127416.pt}
REGIONS=${REGIONS:-"13 4 22"}
GARMENTS=${GARMENTS:-"hospital_gown tshirt_26 tshirt_68 tshirt_4 tshirt_392"}
POSES=${POSES:-all}
STEPS=${STEPS:-30000}
SEED=${SEED:-0}
BRANCHES=${BRANCHES:-"output/uipc_manip/decision_branches_20260918/t26_14046 output/uipc_manip/decision_branches_20260918/t392_14046"}

export PYTHONPATH=build_raw/python/src:python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib
PY=${PY:-/home/ge47gax/kun/genesis-world/.venv/bin/python}

mkdir -p "$ROOT"
echo "[widen] root $ROOT, regions $REGIONS, poses $POSES"

for region in $REGIONS; do
  out="$ROOT/collect_r${region}"
  if [ -d "$out" ]; then echo "[widen] $out exists, skipping"; continue; fi
  echo "[widen] collecting region $region"
  $PY -m uipc_manip.expert_baseline --region "$region" --poses "$POSES" --garments $GARMENTS \
      --num-envs 25 --seed "$SEED" --supervised --out-dir "$out" 2>&1 | tail -3
done

for region in $REGIONS; do
  out="$ROOT/dataset_r${region}"
  if [ -d "$out" ]; then echo "[widen] $out exists, skipping"; continue; fi
  echo "[widen] replaying region $region for observations"
  $PY scripts/reconstruct_dressing_demonstrations.py --source "$ROOT/collect_r${region}" \
      --reference "$CHECKPOINT" --out "$out" --include-failures --upperarm-extension-m 0.05 \
      --repeats 1 2>&1 | tail -3
done

echo "[widen] refitting the behavior flow"
$PY -m uipc_manip.train_fql train --dataset "$ROOT/dataset_r$(echo $REGIONS | cut -d' ' -f1)" \
    --reference "$CHECKPOINT" --steps "$STEPS" --out "$ROOT/train" 2>&1 | tail -5

echo "[widen] auditing the new prior"
$PY scripts/flow_prior_support_audit.py --checkpoint "$ROOT/train/final.pt" \
    --dataset "$ROOT/dataset_r$(echo $REGIONS | cut -d' ' -f1)" --out "$ROOT/prior_audit.json" 2>&1 | tail -12
$PY scripts/flow_reversal_macros.py --checkpoint "$ROOT/train/final.pt" \
    --branches $BRANCHES --out "$ROOT/macro_audit.json" 2>&1 | tail -10
echo "[widen] done; compare $ROOT/macro_audit.json with output/uipc_manip/flow_reversal_20260919/macro_audit.json"
