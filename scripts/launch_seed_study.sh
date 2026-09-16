#!/usr/bin/env bash
# Five-seed, 40k-transition study of the IPC actor label on the 64-slot direct-picker benchmark.
# Group A replicates the configuration of the 2026-09-15 20k round (replay-batch actor term,
# mix rho .5, continuation trust 2, Gaussian locality .5) against SAC with the same settings.
# Group B is the fresh-batch protocol (label at the executed action, 31 replay SAC actor steps +
# 1 bounded fresh step per lockstep step) with its matched SAC and norm-matched random controls.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/ge47gax/kun/genesis-world/.venv/bin/python"
OUT="$ROOT/output/iaql/study40k"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/build_raw/python/src:$ROOT/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/build_raw/Release/bin:/home/ge47gax/Toolchain/uipc_cuda128/lib:${LD_LIBRARY_PATH:-}"
common=(--phase online --num-slots 64 --device cuda --tangent-device cuda --online-steps 40000
        --eval-every 5000 --eval-episodes 64 --eval-steps 50 --updates-per-step 2 --friction 0
        --online-weights 0 --actor-mode mix --continuation-trust 2)
run() {
  local name=$1; shift
  echo "$(date +%H:%M) launch $name" >> "$OUT/launch.log"
  "$PY" -m uipc_manip.iaql_benchmark --out "$OUT/$name" "${common[@]}" "$@" > "$OUT/$name.log" 2>&1 &
}
for seed in 0 1 2 3 4; do
  A=(--seed "$seed" --horizon 150 --actor-batch replay --warmup-transitions 64)
  run "A_sac_s$seed"   "${A[@]}" --actor-weight 0
  run "A_actor_s$seed" "${A[@]}" --actor-weight 1 --actor-rho 0.5 --actor-sigma 0.5
  B=(--seed "$seed" --horizon 50 --actor-batch fresh --warmup-transitions 1024 --fresh-replay-actor-updates 31
     --actor-step-radius 0.03 --actor-step-retries 8)
  run "B_fresh_sac_s$seed"    "${B[@]}" --actor-weight 0
  run "B_fresh_ipc_s$seed"    "${B[@]}" --actor-weight 1 --actor-rho 1 --physics-control paired --physics-signal bellman
  run "B_fresh_random_s$seed" "${B[@]}" --actor-weight 1 --actor-rho 1 --physics-control random --physics-signal bellman
done
wait
echo "$(date +%H:%M) all done" >> "$OUT/launch.log"
