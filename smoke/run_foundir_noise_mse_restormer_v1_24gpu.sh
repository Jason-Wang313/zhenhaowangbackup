#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
CONFIG="$ROOT/configs/foundir_noise_mse_restormer_v1_24gpu.json"
OUTPUT="$ROOT/checkpoints/foundir_noise_mse_restormer_v1_24gpu"
RESULT="$OUTPUT/development_selection.json"
NODE_RANK="${NODE_RANK:?set NODE_RANK to 0, 1, or 2}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to node 0 private IP}"
MASTER_PORT="${MASTER_PORT:-29622}"
LOG="$ROOT/logs/foundir_noise_mse_restormer_v1_24gpu_node${NODE_RANK}.log"
GPU_LOG="$ROOT/logs/foundir_noise_mse_restormer_v1_24gpu_node${NODE_RANK}_gpu.csv"
LOCK="$ROOT/logs/foundir_noise_mse_restormer_v1_24gpu_node${NODE_RANK}.lock"

[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" || "$NODE_RANK" == "2" ]] || { echo "ERROR: NODE_RANK must be 0, 1, or 2" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: Noise MSE node $NODE_RANK already running" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal selection already exists" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_NOISE_MSE_24GPU_END node_rank=$NODE_RANK status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
echo "FOUNDIR_NOISE_MSE_24GPU_START node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE training_only Noise 3x8_H100 global_batch24 1334_steps RGB_MSE total_examples32016"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run \
  --nnodes=3 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  "$TRAINER" --config "$CONFIG"
[[ -f "$RESULT" ]] || { echo "ERROR: terminal selection missing" >&2; exit 1; }
