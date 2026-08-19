#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/qwen_lowlight_supervised_overfit40_ddp8.json"
TRAINER="$ROOT/smoke/train_qwen_lowlight_supervised_ddp.py"
PREFLIGHT="$ROOT/smoke/qwen_lowlight_supervised_ddp_preflight.py"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
LOG="$ROOT/logs/qwen_lowlight_supervised_overfit40_ddp8.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_supervised_overfit40_ddp8_gpu.csv"
LOCK="$ROOT/logs/qwen_lowlight_supervised_overfit40_ddp8.lock"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_supervised_overfit40_ddp8"

mkdir -p "$ROOT/logs"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: the distributed supervised Qwen gate is already active" >&2
  exit 1
fi

monitor_pid=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT

monitor_gpus() {
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
      --format=csv,noheader,nounits
    sleep 2
  done
}

run_gate() {
  set -e
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
    exit 1
  fi
  if [[ ! -f "$CONFIG" || ! -f "$TRAINER" || ! -f "$PREFLIGHT" ]]; then
    echo "ERROR: checked-in DDP gate files are missing" >&2
    exit 1
  fi
  if [[ "$DEVICE_SET" != "0,1,2,3,4,5,6,7" ]]; then
    echo "ERROR: this gate requires CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
    exit 1
  fi
  cd "$ROOT"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -c \
    'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())'
  "$PYTHON" "$PREFLIGHT" --config "$CONFIG" --skip-large-model-hashes
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node=8 --max-restarts=0 \
    "$TRAINER" --config "$CONFIG" --resume auto --stop-after-step 2
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node=8 --max-restarts=0 \
    "$TRAINER" --config "$CONFIG" --resume auto
}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

launch_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_revision="$(git -C "$ROOT" rev-parse HEAD)"
config_sha256="$(sha256sum "$CONFIG" | awk '{print $1}')"
{
  echo "QWEN_SUPERVISED_DDP8_LAUNCH_BEGIN $launch_utc"
  echo "workspace=$ROOT"
  echo "config=$CONFIG"
  echo "config_sha256=$config_sha256"
  echo "git_revision=$git_revision"
  echo "CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "output_dir=$OUTPUT"
  echo "persistent_log=$LOG"
  echo "gpu_monitor_log=$GPU_LOG"
} | tee -a "$LOG"

{
  echo "QWEN_SUPERVISED_DDP8_GPU_MONITOR_BEGIN $launch_utc"
  echo "timestamp,index,name,utilization_gpu_percent,memory_used_MiB,memory_total_MiB,power_draw_W,sm_clock_MHz"
} >>"$GPU_LOG"
monitor_gpus >>"$GPU_LOG" 2>&1 &
monitor_pid="$!"

set +e
run_gate 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"
set -e
stop_monitor
monitor_pid=""

completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "QWEN_SUPERVISED_DDP8_TERMINAL_EXIT_STATUS $status"
  echo "QWEN_SUPERVISED_DDP8_LAUNCH_END $completion_utc status=$status"
} | tee -a "$LOG"
echo "QWEN_SUPERVISED_DDP8_GPU_MONITOR_END $completion_utc status=$status" >>"$GPU_LOG"
exit "$status"
