#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QWEN_PY="$ROOT/.venv/flux2-smoke/bin/python"
METRIC_PY="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_lowlight_gate3_causal_v1.json"
CONFIG_SHA256="d70ef493834c0da6e4c2f519d96e5cfd300f23c0eaed20994979f29aa1b7a111"
LOG="$ROOT/logs/foundir_lowlight_gate3_causal_v1.log"
EVENTS="$ROOT/logs/foundir_lowlight_gate3_causal_v1_events.log"
GPU_LOG="$ROOT/logs/foundir_lowlight_gate3_causal_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_lowlight_gate3_causal_v1.lock"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3" >&2
  exit 1
fi
if [[ ! -x "$QWEN_PY" || ! -x "$METRIC_PY" ]]; then
  echo "ERROR: frozen workspace Python environments are unavailable" >&2
  exit 1
fi
if [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA256" ]]; then
  echo "ERROR: frozen Gate 3 causal config changed" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: Gate 3 causal bundle is already active" >&2
  exit 1
fi

mapfile -t gpu_inventory < <(nvidia-smi --id=0,1,2,3 --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
if [[ "${#gpu_inventory[@]}" -ne 4 ]]; then
  echo "ERROR: expected four selected GPUs" >&2
  exit 1
fi
for position in 0 1 2 3; do
  row="${gpu_inventory[$position]}"
  if [[ ! "$row" =~ ^[[:space:]]*$position,[[:space:]]*NVIDIA[[:space:]]+H(100|200)[^,]*,[[:space:]]*([0-9]+) ]]; then
    echo "ERROR: GPU $position must be an H100/H200 with parseable memory: $row" >&2
    exit 1
  fi
  if [[ "${BASH_REMATCH[2]}" -lt 76800 ]]; then
    echo "ERROR: GPU $position has insufficient memory: $row" >&2
    exit 1
  fi
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=12
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="$ROOT/models/torch"
export XDG_CACHE_HOME="$ROOT/.cache/foundir-metrics"

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
    utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while IFS= read -r gpu_row; do printf '%s,%s\n' "$utc" "$gpu_row"; done < <(
      nvidia-smi --id=0,1,2,3 --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm --format=csv,noheader,nounits
    )
    sleep 2
  done
}
monitor_gpus >>"$GPU_LOG" 2>&1 &
monitor_pid="$!"

begin="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_GATE3_CAUSAL_BEGIN utc=$begin git_revision=$(git -C "$ROOT" rev-parse HEAD)" | tee -a "$EVENTS"

"$QWEN_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$ROOT/smoke/export_foundir_lowlight_gate3_causal_qwen.py" --config "$CONFIG"
echo "FOUNDIR_GATE3_CAUSAL_QWEN_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EVENTS"

"$METRIC_PY" "$ROOT/smoke/run_foundir_lowlight_gate3_causal_adapter.py" --config "$CONFIG" --prepare-features
"$METRIC_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$ROOT/smoke/run_foundir_lowlight_gate3_causal_adapter.py" --config "$CONFIG"
echo "FOUNDIR_GATE3_CAUSAL_ADAPTER_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EVENTS"

"$METRIC_PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$ROOT/smoke/evaluate_foundir_lowlight_gate3_causal.py" --config "$CONFIG"
echo "FOUNDIR_GATE3_CAUSAL_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=0" | tee -a "$EVENTS"
