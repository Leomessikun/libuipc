#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/ge47gax/kun/genesis-world/.venv/bin/python"
export PYTHONPATH="$ROOT/build_raw/python/src:$ROOT/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib:${LD_LIBRARY_PATH:-}"

common=(
  --phase online --online-steps 2048 --eval-every 512 --eval-episodes 16
  --eval-steps 50 --horizon 50 --num-slots 64 --updates-per-step 2
  --batch-size 32 --warmup-transitions 1024 --friction 0
  --device cuda --actor-mode mix --actor-step-radius 0.03
  --actor-step-retries 8 --continuation-trust 2 --online-weights 0
)

for seed in 0 1 2; do
  checkpoint="$ROOT/output/iaql/vec20k_s${seed}_sac64/online_beta_0.0.pt"
  for arm in replay_sac fresh_sac fresh_ipc fresh_random; do
    out="$ROOT/output/iaql/matched_s${seed}_${arm}"
    extra=()
    case "$arm" in
      replay_sac)
        extra+=(--actor-batch replay --actor-weight 0)
        ;;
      fresh_sac)
        extra+=(--actor-batch fresh --actor-weight 0 --fresh-replay-actor-updates 31)
        ;;
      fresh_ipc)
        extra+=(--actor-batch fresh --actor-weight 1 --actor-rho 1
                --physics-control paired --physics-signal bellman
                --fresh-replay-actor-updates 31)
        ;;
      fresh_random)
        extra+=(--actor-batch fresh --actor-weight 1 --actor-rho 1
                --physics-control random --physics-signal bellman
                --fresh-replay-actor-updates 31)
        ;;
    esac
    echo "launch seed=$seed arm=$arm checkpoint=$checkpoint out=$out" | tee -a "$ROOT/output/iaql/matched_launch.log"
    "$PY" -m uipc_manip.iaql_benchmark --out "$out" \
      --resume-checkpoint "$checkpoint" "${common[@]}" "${extra[@]}" \
      > "$ROOT/output/iaql/matched_s${seed}_${arm}.log" 2>&1
  done
done
