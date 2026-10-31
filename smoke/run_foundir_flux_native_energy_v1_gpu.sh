#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
RUN_ID=${1:-qualification_v1}
GPUS=${2:-8}
RESUME=${3:-fresh}
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; exit 2; }
[[ "$GPUS" == 4 || "$GPUS" == 8 ]] || { echo 'ERROR: use 4 or 8 GPUs'; exit 2; }
[[ "$RESUME" == fresh || "$RESUME" == resume ]] || { echo 'ERROR: fresh or resume required'; exit 2; }
PYTHON=${FLUX_NATIVE_PYTHON:-"$ROOT/.conda/gpu-train-stable/bin/python"}
[[ -x "$PYTHON" ]] || { echo "ERROR: qualified Python missing: $PYTHON"; exit 2; }
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$GPUS" == 8 ]]; then export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  else export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
fi
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"
unset MASTER_ADDR MASTER_PORT NNODES NODE_RANK NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME
mkdir -p logs
exec 9>"logs/flux_native_energy_${RUN_ID}.lock"
flock -n 9 || { echo 'ERROR: this run already has a launcher'; exit 3; }
LOG="logs/flux_native_energy_${RUN_ID}_$(date -u +%Y%m%dT%H%M%SZ).log"
ARGS=(--run-id "$RUN_ID" --world-size "$GPUS")
[[ "$RESUME" == resume ]] && ARGS+=(--resume)
START_EPOCH=$(date -u +%s)
(
  trap 'rc=$?; elapsed=$(( $(date -u +%s) - START_EPOCH )); printf "NATIVE_ENERGY_END status=%s wall_seconds=%s allocated_gpu_seconds=%s\n" "$rc" "$elapsed" "$((elapsed*GPUS))"' EXIT
  echo "NATIVE_ENERGY_START run=$RUN_ID gpus=$GPUS mode=$RESUME utc=$(date -u +%FT%TZ)"
  echo 'UNMEASURED_GPU_PHASE_CAP=2_wall_hours_including_cache_excluding_CPU_preflight; not an expected runtime; no automatic trainfit'
  "$PYTHON" -u smoke/train_foundir_flux_native_energy_v1.py --hardware --world-size "$GPUS"
  "$PYTHON" -u smoke/train_foundir_flux_native_energy_v1.py --prepare "${ARGS[@]}"
  timeout --signal=TERM --kill-after=60s 7200s "$PYTHON" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$GPUS" --max_restarts=0 \
    smoke/train_foundir_flux_native_energy_v1.py "${ARGS[@]}"
) 2>&1 | tee "$LOG"
