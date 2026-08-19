#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/qwen_lowlight_real_validation_step600.json"
PREFLIGHT="$ROOT/smoke/qwen_lowlight_real_validation_preflight.py"
EVALUATOR="$ROOT/smoke/evaluate_qwen_lowlight_real_validation.py"
SELECTED_DEVICE="0"
LOG="$ROOT/logs/qwen_lowlight_real_validation_step600.log"
EVENTS="$ROOT/logs/qwen_lowlight_real_validation_step600_events.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_real_validation_step600_gpu.csv"
LOCK="$ROOT/logs/qwen_lowlight_real_validation_step600.lock"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_real_validation_step600"

REQUIRED_TRACKED_FILES=(
  configs/qwen_lowlight_smoke25.json
  configs/qwen_lowlight_supervised_pilot_8k_ddp8.json
  configs/qwen_lowlight_real_validation_step600.json
  manifests/qwen_lowlight_real_validation.csv
  manifests/qwen_lowlight_real_validation.json
  lowlight_restoration/data.py
  lowlight_restoration/qwen_pilot.py
  lowlight_restoration/qwen_real_validation.py
  lowlight_restoration/qwen_supervised.py
  smoke/qwen_lowlight_supervised_preflight.py
  smoke/train_qwen_lowlight_supervised.py
  smoke/qwen_lowlight_real_validation_preflight.py
  smoke/evaluate_qwen_lowlight_real_validation.py
  smoke/run_qwen_lowlight_real_validation.sh
  smoke/launch_qwen_lowlight_real_validation_tmux.sh
)

validate_static_contract() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
    return 1
  fi
  if [[ ! -f "$CONFIG" || ! -f "$PREFLIGHT" || ! -f "$EVALUATOR" ]]; then
    echo "ERROR: checked-in real-validation files are missing" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_qwen_lowlight_real_validation.sh"
  bash -n "$ROOT/smoke/launch_qwen_lowlight_real_validation_tmux.sh"
  "$PYTHON" -m py_compile "$PREFLIGHT" "$EVALUATOR"
  "$PYTHON" "$PREFLIGHT" \
    --config "$CONFIG" \
    --contract-only \
    --skip-large-model-hashes \
    --no-write-artifact
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  validate_static_contract
  echo "QWEN_REAL_VALIDATION_RUNNER_PREFLIGHT_PASS"
  echo "Selected device: CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE"
  echo "Persistent log: $LOG"
  echo "GPU telemetry: $GPU_LOG"
  echo "Output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected real-validation runner argument: $1" >&2
  exit 1
fi

validate_static_contract
mkdir -p "$ROOT/logs" "$ROOT/logs/qwen_lowlight_real_validation_step600_attempts"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: frozen-Qwen real validation is already active" >&2
  exit 1
fi

attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ATTEMPT_DIR="$ROOT/logs/qwen_lowlight_real_validation_step600_attempts"
ATTEMPT_LOG="$ATTEMPT_DIR/${attempt_id}.log"
ATTEMPT_EVENTS="$ATTEMPT_DIR/${attempt_id}_events.log"
ATTEMPT_GPU_LOG="$ATTEMPT_DIR/${attempt_id}_gpu.csv"
monitor_pid=""

stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}

monitor_gpu() {
  while true; do
    sample_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    {
      while IFS= read -r gpu_row; do
        printf '%s,%s,%s\n' "$attempt_id" "$sample_utc" "$gpu_row"
      done < <(
        nvidia-smi \
          --id="$SELECTED_DEVICE" \
          --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
          --format=csv,noheader,nounits
      )
    } | tee -a "$GPU_LOG" >>"$ATTEMPT_GPU_LOG"
    sleep 2
  done
}

emit_event() {
  marker="$1"
  printf '%s attempt_id=%s utc=%s\n' \
    "$marker" "$attempt_id" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "$EVENTS" "$ATTEMPT_EVENTS"
}

run_validation() {
  set -e
  trap stop_monitor EXIT
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$SELECTED_DEVICE" ]]; then
    echo "ERROR: launch with CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE" >&2
    return 1
  fi
  cd "$ROOT"
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    if ! git ls-files --error-unmatch "$relative_path" >/dev/null 2>&1; then
      echo "ERROR: required real-validation file is not tracked: $relative_path" >&2
      return 1
    fi
  done
  if ! git diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: required real-validation files differ from the committed Git revision" >&2
    return 1
  fi

  gpu_inventory="$(
    nvidia-smi \
      --id="$SELECTED_DEVICE" \
      --query-gpu=index,name,memory.total \
      --format=csv,noheader,nounits
  )"
  if [[ ! "$gpu_inventory" =~ ^[[:space:]]*${SELECTED_DEVICE},[^,]*H(100|200) ]]; then
    echo "ERROR: selected GPU must be an H100 or H200: $gpu_inventory" >&2
    return 1
  fi
  echo "$gpu_inventory"
  CUDA_VISIBLE_DEVICES="$SELECTED_DEVICE" "$PYTHON" -c \
    'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count(), "gpu", torch.cuda.get_device_name(0))'

  emit_event "QWEN_REAL_VALIDATION_FULL_PREFLIGHT_BEGIN"
  set +e
  CUDA_VISIBLE_DEVICES="$SELECTED_DEVICE" "$PYTHON" "$PREFLIGHT" \
    --config "$CONFIG" \
    --require-gpu
  preflight_status="$?"
  set -e
  emit_event "QWEN_REAL_VALIDATION_FULL_PREFLIGHT_END status=$preflight_status"
  if [[ "$preflight_status" -ne 0 ]]; then
    return "$preflight_status"
  fi

  emit_event "QWEN_REAL_VALIDATION_INFERENCE_BEGIN"
  set +e
  CUDA_VISIBLE_DEVICES="$SELECTED_DEVICE" "$PYTHON" "$EVALUATOR" --config "$CONFIG"
  inference_status="$?"
  set -e
  emit_event "QWEN_REAL_VALIDATION_INFERENCE_END status=$inference_status"
  return "$inference_status"
}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

launch_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_revision="$(git -C "$ROOT" rev-parse HEAD)"
config_sha256="$(sha256sum "$CONFIG" | awk '{print $1}')"
{
  echo "attempt_id,sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz"
} >>"$ATTEMPT_GPU_LOG"
{
  echo "# QWEN_REAL_VALIDATION_GPU_MONITOR_BEGIN attempt_id=$attempt_id utc=$launch_utc"
  echo "attempt_id,sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz"
} >>"$GPU_LOG"
{
  echo "QWEN_REAL_VALIDATION_LAUNCH_BEGIN attempt_id=$attempt_id utc=$launch_utc"
  echo "workspace=$ROOT"
  echo "config=$CONFIG"
  echo "config_sha256=$config_sha256"
  echo "git_revision=$git_revision"
  echo "CUDA_VISIBLE_DEVICES=$SELECTED_DEVICE"
  echo "gpu_count=1"
  echo "job_kind=frozen_real_domain_evaluation"
  echo "output_dir=$OUTPUT"
  echo "persistent_log=$LOG"
  echo "attempt_log=$ATTEMPT_LOG"
  echo "events_log=$EVENTS"
  echo "gpu_monitor_log=$GPU_LOG"
} | tee -a "$LOG" "$EVENTS" "$ATTEMPT_LOG" "$ATTEMPT_EVENTS"

monitor_gpu &
monitor_pid="$!"
set +e
(run_validation) 2>&1 | tee -a "$LOG" "$ATTEMPT_LOG"
status="${PIPESTATUS[0]}"
set -e
stop_monitor

completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "QWEN_REAL_VALIDATION_TERMINAL_EXIT_STATUS $status"
  echo "QWEN_REAL_VALIDATION_LAUNCH_END attempt_id=$attempt_id utc=$completion_utc status=$status"
  if [[ "$status" -eq 2 ]]; then
    echo "QWEN_REAL_VALIDATION_LEGITIMATE_AUTOMATIC_GATE_FAILURE status=2"
  fi
} | tee -a "$LOG" "$EVENTS" "$ATTEMPT_LOG" "$ATTEMPT_EVENTS"
echo "# QWEN_REAL_VALIDATION_GPU_MONITOR_END attempt_id=$attempt_id utc=$completion_utc status=$status" >>"$GPU_LOG"
echo "# QWEN_REAL_VALIDATION_GPU_MONITOR_END attempt_id=$attempt_id utc=$completion_utc status=$status" >>"$ATTEMPT_GPU_LOG"
exit "$status"
