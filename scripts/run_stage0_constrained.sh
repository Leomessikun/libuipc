#!/bin/bash
# Stage 0: does putting the success criterion into the training signal change anything?
# Two arms differing only in the objective. Both carry the observation's constraint slot,
# both start from scratch, same seed, same cells, same budget.
set -eu
ARM=$1  # control | treatment
case "$ARM" in
  control)   LAMBDA_LR=0.0 ;;
  treatment) LAMBDA_LR=1.0 ;;
  *) echo "arm must be control or treatment" >&2; exit 2 ;;
esac
shift
cd "$(dirname "$0")/.."
exec env PYTHONPATH=build_raw/python/src:python OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  LD_LIBRARY_PATH=build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib \
  /home/ge47gax/kun/genesis-world/.venv/bin/python -m uipc_manip.train_sac \
  --task dressing --cell-source live \
  --garments hospital_gown tshirt_26 tshirt_392 tshirt_4 tshirt_68 \
  --body-seeds 14045,14046,14047,14048,14049 \
  --num-envs 25 --horizon 300 --action-repeat 6 --total-transitions 270000 \
  --num-eval-episodes 25 --eval-freq 600 --checkpoint-interval 1800 \
  --obs-mode visible_dual --no-obs-augment --seed 1 \
  --constraint-objective --constraint-episode --constraint-lambda-lr "$LAMBDA_LR" \
  --constraint-budget 0.0 --constraint-lambda-max 50.0 \
  --work-dir output/uipc_manip/stage0_20260921_episode --run-name "$ARM" "$@"
