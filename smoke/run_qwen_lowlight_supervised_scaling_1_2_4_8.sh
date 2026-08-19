#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/qwen_lowlight_supervised_scaling_1_2_4_8.json"
BENCHMARK="$ROOT/smoke/benchmark_qwen_lowlight_supervised_scaling.py"
VALIDATOR="$ROOT/smoke/validate_qwen_lowlight_supervised_scaling.py"
ALL_DEVICES="0,1,2,3,4,5,6,7"
LOG="$ROOT/logs/qwen_lowlight_supervised_scaling_1_2_4_8.log"
EVENTS="$ROOT/logs/qwen_lowlight_supervised_scaling_1_2_4_8_events.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_supervised_scaling_1_2_4_8_gpu.csv"
LOCK="$ROOT/logs/qwen_lowlight_supervised_scaling_1_2_4_8.lock"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_supervised_scaling_1_2_4_8"

REQUIRED_TRACKED_FILES=(
  configs/qwen_lowlight_smoke25.json
  configs/qwen_lowlight_supervised_overfit40_ddp8.json
  configs/qwen_lowlight_supervised_scaling_1_2_4_8.json
  lowlight_restoration/qwen_distributed.py
  lowlight_restoration/qwen_scaling.py
  lowlight_restoration/qwen_supervised.py
  smoke/benchmark_qwen_lowlight_supervised_scaling.py
  smoke/launch_qwen_lowlight_supervised_scaling_tmux.sh
  smoke/qwen_lowlight_supervised_ddp_preflight.py
  smoke/qwen_lowlight_supervised_preflight.py
  smoke/train_qwen_lowlight_supervised.py
  smoke/train_qwen_lowlight_supervised_ddp.py
  smoke/validate_qwen_lowlight_supervised_scaling.py
  smoke/run_qwen_lowlight_supervised_scaling_1_2_4_8.sh
)

validate_static_contract() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
    return 1
  fi
  if [[ ! -f "$CONFIG" || ! -f "$BENCHMARK" || ! -f "$VALIDATOR" ]]; then
    echo "ERROR: checked-in scaling gate files are missing" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_qwen_lowlight_supervised_scaling_1_2_4_8.sh"
  "$PYTHON" -m py_compile "$BENCHMARK" "$VALIDATOR"
  "$PYTHON" "$VALIDATOR" --config "$CONFIG" --contract-only
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  validate_static_contract
  echo "QWEN_SCALING_RUNNER_PREFLIGHT_PASS"
  echo "Persistent log: $LOG"
  echo "GPU telemetry: $GPU_LOG"
  echo "Output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected runner argument: $1" >&2
  exit 1
fi

validate_static_contract
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: the Qwen scaling gate is already active" >&2
  exit 1
fi

attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
export QWEN_SCALING_ATTEMPT_ID="$attempt_id"
monitor_pid=""

stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}

monitor_gpus() {
  local stage="$1"
  while true; do
    local sample_utc
    sample_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while IFS= read -r gpu_row; do
      printf '%s,%s,%s,%s\n' \
        "$attempt_id" "$stage" "$sample_utc" "$gpu_row"
    done < <(
      nvidia-smi \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
        --format=csv,noheader,nounits
    )
    sleep 2
  done
}

start_monitor() {
  local stage="$1"
  if [[ -n "$monitor_pid" ]]; then
    echo "ERROR: GPU monitor is already active" >&2
    return 1
  fi
  monitor_gpus "$stage" >>"$GPU_LOG" 2>&1 &
  monitor_pid="$!"
}

emit_event() {
  local marker="$1"
  printf '%s attempt_id=%s utc=%s\n' \
    "$marker" "$attempt_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EVENTS"
}

run_world() {
  local world_size="$1"
  local devices="$2"
  emit_event "QWEN_SCALING_WORLD_${world_size}_BEGIN"
  start_monitor "world_${world_size}"
  export QWEN_SCALING_SHELL_START_NS
  QWEN_SCALING_SHELL_START_NS="$(date +%s%N)"
  set +e
  CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node="$world_size" --max-restarts=0 \
    "$BENCHMARK" --config "$CONFIG" --world-size "$world_size"
  local status="$?"
  set -e
  stop_monitor
  printf 'QWEN_SCALING_WORLD_%s_TERMINAL_EXIT_STATUS %s\n' \
    "$world_size" "$status"
  emit_event "QWEN_SCALING_WORLD_${world_size}_END status=$status"
  return "$status"
}

run_gate() {
  set -e
  trap stop_monitor EXIT
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$ALL_DEVICES" ]]; then
    echo "ERROR: launch this gate with CUDA_VISIBLE_DEVICES=$ALL_DEVICES" >&2
    exit 1
  fi
  cd "$ROOT"
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    if ! git ls-files --error-unmatch "$relative_path" >/dev/null 2>&1; then
      echo "ERROR: scaling gate file is not tracked: $relative_path" >&2
      exit 1
    fi
  done
  if ! git diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: scaling gate files differ from the committed Git revision" >&2
    exit 1
  fi

  mapfile -t gpu_inventory < <(
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
  )
  if [[ "${#gpu_inventory[@]}" -ne 8 ]]; then
    echo "ERROR: scaling gate requires exactly eight visible physical GPUs" >&2
    exit 1
  fi
  for index in "${!gpu_inventory[@]}"; do
    if [[ ! "${gpu_inventory[$index]}" =~ ^${index},.*H200 ]]; then
      echo "ERROR: expected physical GPU $index to be an H200: ${gpu_inventory[$index]}" >&2
      exit 1
    fi
  done
  printf '%s\n' "${gpu_inventory[@]}"
  CUDA_VISIBLE_DEVICES="$ALL_DEVICES" "$PYTHON" -c \
    'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())'

  emit_event "QWEN_SCALING_PREFLIGHT_BEGIN"
  start_monitor "preflight"
  set +e
  CUDA_VISIBLE_DEVICES="$ALL_DEVICES" "$PYTHON" "$VALIDATOR" \
    --config "$CONFIG" --preflight-only
  local preflight_status="$?"
  set -e
  stop_monitor
  printf 'QWEN_SCALING_PREFLIGHT_TERMINAL_EXIT_STATUS %s\n' "$preflight_status"
  emit_event "QWEN_SCALING_PREFLIGHT_END status=$preflight_status"
  if [[ "$preflight_status" -ne 0 ]]; then
    return "$preflight_status"
  fi

  run_world 1 "0"
  run_world 2 "0,1"
  run_world 4 "0,1,2,3"
  run_world 8 "$ALL_DEVICES"

  emit_event "QWEN_SCALING_SELECTOR_BEGIN"
  set +e
  "$PYTHON" "$VALIDATOR" --config "$CONFIG"
  local selector_status="$?"
  set -e
  printf 'QWEN_SCALING_SELECTOR_TERMINAL_EXIT_STATUS %s\n' "$selector_status"
  emit_event "QWEN_SCALING_SELECTOR_END status=$selector_status"
  return "$selector_status"
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
  echo "# QWEN_SCALING_GPU_MONITOR_BEGIN attempt_id=$attempt_id utc=$launch_utc"
  echo "attempt_id,stage,sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz"
} >>"$GPU_LOG"
{
  echo "QWEN_SCALING_LAUNCH_BEGIN attempt_id=$attempt_id utc=$launch_utc"
  echo "workspace=$ROOT"
  echo "config=$CONFIG"
  echo "config_sha256=$config_sha256"
  echo "git_revision=$git_revision"
  echo "CUDA_VISIBLE_DEVICES=$ALL_DEVICES"
  echo "world_sizes=1,2,4,8"
  echo "output_dir=$OUTPUT"
  echo "persistent_log=$LOG"
  echo "events_log=$EVENTS"
  echo "gpu_monitor_log=$GPU_LOG"
} | tee -a "$LOG" "$EVENTS"

set +e
(run_gate) 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"
set -e
stop_monitor

completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "QWEN_SCALING_TERMINAL_EXIT_STATUS $status"
  echo "QWEN_SCALING_LAUNCH_END attempt_id=$attempt_id utc=$completion_utc status=$status"
} | tee -a "$LOG" "$EVENTS"
echo "# QWEN_SCALING_GPU_MONITOR_END attempt_id=$attempt_id utc=$completion_utc status=$status" >>"$GPU_LOG"
exit "$status"
