#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/flux_native_energy_launcher_lib.sh"
configure_native_energy "$@"
if [[ "${FLUX_NATIVE_DRY_RUN:-0}" == 1 ]]; then
  printf 'layout=%sx%s world=%s node=%s visible=%s\n' "$NNODES" "$LOCAL_GPUS" "$GPUS" "$NODE_RANK" "$VISIBLE"
  printf '%q ' "${TORCHRUN[@]}"; printf '\n'; exit 0
fi
cd "$ROOT"
[[ -x "$PYTHON" ]] || { echo "ERROR: qualified Python missing: $PYTHON"; exit 2; }
export CUDA_VISIBLE_DEVICES="$VISIBLE" NNODES NODE_RANK MASTER_ADDR MASTER_PORT
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export "${TRANSPORT[@]}"
mkdir -p logs
# Per-node locks: a single shared run lock would incorrectly reject nodes 1/2.
exec 8>/tmp/zhenhaowang_flux_native_energy.lock
flock -n 8 || { echo 'ERROR: another native-energy launcher owns this node'; exit 3; }
exec 9>"logs/flux_native_energy_${RUN_ID}_node${NODE_RANK}.lock"
flock -n 9 || { echo 'ERROR: this node rank already has a launcher'; exit 3; }
LOG="logs/flux_native_energy_${RUN_ID}_node${NODE_RANK}_$(date -u +%Y%m%dT%H%M%SZ).log"
ARGS=(--run-id "$RUN_ID" --world-size "$GPUS" --node-rank "$NODE_RANK")
[[ "$RESUME" == resume ]] && ARGS+=(--resume)
START_EPOCH=$(date -u +%s)
(
  trap 'rc=$?; elapsed=$(( $(date -u +%s) - START_EPOCH )); printf "NATIVE_ENERGY_END status=%s node_rank=%s wall_seconds=%s local_gpu_seconds=%s\n" "$rc" "$NODE_RANK" "$elapsed" "$((elapsed*LOCAL_GPUS))"' EXIT
  echo "NATIVE_ENERGY_START run=$RUN_ID layout=${NNODES}x${LOCAL_GPUS} node_rank=$NODE_RANK mode=$RESUME utc=$(date -u +%FT%TZ)"
  echo "UNMEASURED_GPU_PHASE_CAP=7200_seconds_per_node including_rendezvous_cache_train_probe; total_gpu_seconds_cap=$((7200*GPUS)); excludes_CPU_preflight_and_cleanup; no_auto_trainfit"
  "$PYTHON" -u smoke/train_foundir_flux_native_energy_v1.py --hardware --world-size "$GPUS"
  "$PYTHON" -u smoke/train_foundir_flux_native_energy_v1.py --prepare "${ARGS[@]}"
  timeout --signal=TERM --kill-after=60s 7200s "${TORCHRUN[@]}"
) 2>&1 | tee "$LOG"
