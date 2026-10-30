#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/flux_tail_v3_launcher_lib.sh"
configure_tail_v3 "$@"
tail_v3_torchrun_command
if [[ "${FLUX_TAIL_V3_DRY_RUN:-0}" == 1 ]]; then
  printf 'ROOT=%s\nLOG=%s\nWALL_CAP=%s LOCAL_GPUS=%s WORLD=%s\n' "$ROOT" "$LOG" "$WALL_CAP" "$LOCAL_GPUS" "$GPU_COUNT"
  printf '%q ' "${TORCHRUN[@]}"; printf '\n'; exit 0
fi
cd "$ROOT"
unset PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# These locks are local to each node. Only global rank zero acquires the
# shared cache lock inside the single distributed pipeline.
exec 7>/tmp/zhenhaowang_flux_tail_switch_v2.lock
flock -n 7 || { echo 'ERROR: another tail-energy job owns this node'; exit 3; }
exec 9>/tmp/zhenhaowang_dual_retry_v1.lock
flock -n 9 || { echo 'ERROR: another project job owns this node'; exit 3; }
exec 8>/tmp/zhenhaowang_rain_blur_image_moe_v1.lock
flock -n 8 || { echo 'ERROR: another project job owns this node'; exit 3; }
export CUDA_VISIBLE_DEVICES="$VISIBLE" OMP_NUM_THREADS=2 NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT/third_party/flux2/src:$ROOT" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export "${TRANSPORT[@]}"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
STARTED=$SECONDS
trap 'status=$?; elapsed=$((SECONDS-STARTED)); echo "TAIL_RELATIVE_V3_END status=$status node_rank=$NODE_RANK elapsed_seconds=$elapsed local_gpu_seconds=$((LOCAL_GPUS*elapsed)) utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"' EXIT
echo "TAIL_RELATIVE_V3_START mode=$MODE run_id=$RUN_ID node_rank=$NODE_RANK nodes=$NNODES local_gpus=$LOCAL_GPUS world=$GPU_COUNT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "TAIL_RELATIVE_V3_RUNTIME root=$ROOT python=$PYTHON master=$MASTER_ADDR:$MASTER_PORT visible=$VISIBLE"
echo "TAIL_RELATIVE_V3_BUDGET wall_cap_seconds=$WALL_CAP total_used_gpu_seconds_cap=$((WALL_CAP*GPU_COUNT)) no_auto_trainfit no_auto_formal"
timeout --signal=TERM --kill-after=30s "$WALL_CAP" "${TORCHRUN[@]}"
