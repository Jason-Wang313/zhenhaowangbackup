#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/qwen_lowlight_real_adaptation_step600.json"
PREFLIGHT="$ROOT/smoke/qwen_lowlight_real_adaptation_preflight.py"
TRAINER="$ROOT/smoke/train_qwen_lowlight_real_adaptation_ddp.py"
RESULT_STATUS="$ROOT/smoke/qwen_real_adaptation_result_status.py"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"
LOG="$ROOT/logs/qwen_lowlight_real_adaptation_step600.log"
EVENTS="$ROOT/logs/qwen_lowlight_real_adaptation_step600_events.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_real_adaptation_step600_gpu.csv"
LOCK="$ROOT/logs/qwen_lowlight_real_adaptation_step600.lock"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_real_adaptation_step600"

case "$DEVICE_SET" in
  0,1)
    WORLD_SIZE=2
    OMP_THREADS=16
    ;;
  0,1,2,3,4,5,6,7)
    WORLD_SIZE=8
    OMP_THREADS=4
    ;;
  *)
    echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1 or 0,1,2,3,4,5,6,7" >&2
    exit 1
    ;;
esac

REQUIRED_TRACKED_FILES=(
  configs/qwen_lowlight_smoke25.json
  configs/qwen_lowlight_supervised_pilot_8k_ddp8.json
  configs/qwen_lowlight_real_validation_step600.json
  configs/qwen_lowlight_real_validation_constrained_step600.json
  configs/qwen_lowlight_real_adaptation_step600.json
  manifests/low_light_pilot_10000.csv
  manifests/qwen_lowlight_real_validation.csv
  manifests/qwen_lowlight_real_validation.json
  manifests/qwen_lowlight_real_adaptation.csv
  manifests/qwen_lowlight_real_adaptation.json
  lowlight_restoration/data.py
  lowlight_restoration/qwen_distributed.py
  lowlight_restoration/qwen_pilot.py
  lowlight_restoration/qwen_supervised.py
  lowlight_restoration/qwen_real_validation.py
  lowlight_restoration/qwen_real_adaptation.py
  smoke/build_qwen_lowlight_real_adaptation_manifest.py
  smoke/qwen_lowlight_supervised_preflight.py
  smoke/train_qwen_lowlight_supervised.py
  smoke/train_qwen_lowlight_supervised_ddp.py
  smoke/qwen_lowlight_real_adaptation_preflight.py
  smoke/train_qwen_lowlight_real_adaptation_ddp.py
  smoke/qwen_real_adaptation_result_status.py
  smoke/run_qwen_lowlight_real_adaptation.sh
  smoke/launch_qwen_lowlight_real_adaptation_tmux.sh
  smoke/run_real_adapt.sh
)

validate_static_contract() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
    return 1
  fi
  if [[ ! -f "$CONFIG" || ! -f "$PREFLIGHT" || ! -f "$TRAINER" || ! -f "$RESULT_STATUS" ]]; then
    echo "ERROR: checked real-adaptation files are missing" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_qwen_lowlight_real_adaptation.sh"
  bash -n "$ROOT/smoke/launch_qwen_lowlight_real_adaptation_tmux.sh"
  bash -n "$ROOT/smoke/run_real_adapt.sh"
  "$PYTHON" -m py_compile "$PREFLIGHT" "$TRAINER" "$RESULT_STATUS"
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
  echo "QWEN_REAL_ADAPTATION_RUNNER_PREFLIGHT_PASS"
  echo "Selected devices: CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "World size: $WORLD_SIZE"
  echo "Persistent log: $LOG"
  echo "GPU telemetry: $GPU_LOG"
  echo "Output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected real-adaptation runner argument: $1" >&2
  exit 1
fi

validate_static_contract
mkdir -p "$ROOT/logs" "$ROOT/logs/qwen_lowlight_real_adaptation_step600_attempts"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: the Qwen real-adaptation gate is already active" >&2
  exit 1
fi

attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
ATTEMPT_DIR="$ROOT/logs/qwen_lowlight_real_adaptation_step600_attempts"
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

query_selected_gpus() {
  nvidia-smi \
    --id="$DEVICE_SET" \
    --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
    --format=csv,noheader,nounits
}

monitor_gpus() {
  while true; do
    sample_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    {
      while IFS= read -r gpu_row; do
        printf '%s,%s,%s\n' "$attempt_id" "$sample_utc" "$gpu_row"
      done < <(query_selected_gpus)
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

run_adaptation() {
  set -e
  trap stop_monitor EXIT
  cd "$ROOT"
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    if ! git ls-files --error-unmatch "$relative_path" >/dev/null 2>&1; then
      echo "ERROR: required real-adaptation file is not tracked: $relative_path" >&2
      return 1
    fi
  done
  if ! git diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: required real-adaptation files differ from the committed revision" >&2
    return 1
  fi

  mapfile -t gpu_inventory < <(
    nvidia-smi \
      --id="$DEVICE_SET" \
      --query-gpu=index,name,memory.total \
      --format=csv,noheader,nounits
  )
  if [[ "${#gpu_inventory[@]}" -ne "$WORLD_SIZE" ]]; then
    echo "ERROR: expected $WORLD_SIZE selected GPUs, found ${#gpu_inventory[@]}" >&2
    return 1
  fi
  IFS=',' read -r -a expected_indices <<<"$DEVICE_SET"
  for position in "${!gpu_inventory[@]}"; do
    row="${gpu_inventory[$position]}"
    if [[ "$row" =~ ^[[:space:]]*([0-9]+),[[:space:]]*NVIDIA[[:space:]]+H(100|200)[^,]*,[[:space:]]*([0-9]+) ]]; then
      gpu_index="${BASH_REMATCH[1]}"
      gpu_memory_mib="${BASH_REMATCH[3]}"
    else
      echo "ERROR: selected GPU must be an H100/H200 with parseable memory: $row" >&2
      return 1
    fi
    if [[ "$gpu_index" != "${expected_indices[$position]}" ]]; then
      echo "ERROR: selected GPU index order changed: $row" >&2
      return 1
    fi
    if (( gpu_memory_mib < 76800 )); then
      echo "ERROR: selected GPU has less than 75 GiB memory: $row" >&2
      return 1
    fi
  done
  printf '%s\n' "${gpu_inventory[@]}"
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -c \
    "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == $WORLD_SIZE; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'count', torch.cuda.device_count(), 'gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"

  emit_event "QWEN_REAL_ADAPTATION_FULL_PREFLIGHT_BEGIN"
  set +e
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" "$PREFLIGHT" \
    --config "$CONFIG" \
    --require-gpu
  preflight_status="$?"
  set -e
  emit_event "QWEN_REAL_ADAPTATION_FULL_PREFLIGHT_END status=$preflight_status"
  if [[ "$preflight_status" -ne 0 ]]; then
    return "$preflight_status"
  fi

  if [[ -f "$OUTPUT/training_summary.json" ]]; then
    set +e
    "$PYTHON" "$RESULT_STATUS" \
      --output-dir "$OUTPUT" \
      --config-sha256 "$config_sha256"
    existing_status="$?"
    set -e
    if [[ "$existing_status" -eq 0 || "$existing_status" -eq 2 ]]; then
      emit_event "QWEN_REAL_ADAPTATION_TRAINING_SKIPPED_EXISTING_VERIFIED_RESULT status=$existing_status"
      return "$existing_status"
    fi
    echo "ERROR: existing adaptation summary is inconsistent" >&2
    return 1
  fi

  emit_event "QWEN_REAL_ADAPTATION_TRAINING_BEGIN"
  set +e
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node="$WORLD_SIZE" --max-restarts=0 \
    "$TRAINER" --config "$CONFIG" --resume auto
  training_status="$?"
  set -e
  emit_event "QWEN_REAL_ADAPTATION_TRAINING_END status=$training_status"
  set +e
  "$PYTHON" "$RESULT_STATUS" \
    --output-dir "$OUTPUT" \
    --config-sha256 "$config_sha256"
  result_status="$?"
  set -e
  case "$result_status" in
    0)
      if [[ "$training_status" -eq 0 ]]; then
        return 0
      fi
      echo "ERROR: verified pass conflicts with launcher status $training_status" >&2
      return 1
      ;;
    2)
      return 2
      ;;
    *)
      echo "ERROR: adaptation summary missing or inconsistent after launcher status $training_status" >&2
      return 1
      ;;
  esac
}

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$OMP_THREADS"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

launch_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_revision="$(git -C "$ROOT" rev-parse HEAD)"
config_sha256="$(sha256sum "$CONFIG" | awk '{print $1}')"
{
  echo "attempt_id,sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz"
} >>"$ATTEMPT_GPU_LOG"
{
  echo "# QWEN_REAL_ADAPTATION_GPU_MONITOR_BEGIN attempt_id=$attempt_id utc=$launch_utc devices=$DEVICE_SET"
  echo "attempt_id,sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz"
} >>"$GPU_LOG"
{
  echo "QWEN_REAL_ADAPTATION_LAUNCH_BEGIN attempt_id=$attempt_id utc=$launch_utc"
  echo "workspace=$ROOT"
  echo "config=$CONFIG"
  echo "config_sha256=$config_sha256"
  echo "git_revision=$git_revision"
  echo "CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "world_size=$WORLD_SIZE"
  echo "job_kind=source_disjoint_real_lowlight_adaptation_gate"
  echo "global_batch_size=40"
  echo "optimizer_steps=200"
  echo "real_samples_per_step=20"
  echo "synthetic_replay_samples_per_step=20"
  echo "gradient_accumulation_steps=$((40 / (5 * WORLD_SIZE)))"
  echo "output_dir=$OUTPUT"
  echo "persistent_log=$LOG"
  echo "attempt_log=$ATTEMPT_LOG"
  echo "events_log=$EVENTS"
  echo "gpu_monitor_log=$GPU_LOG"
} | tee -a "$LOG" "$EVENTS" "$ATTEMPT_LOG" "$ATTEMPT_EVENTS"

monitor_gpus &
monitor_pid="$!"
set +e
(run_adaptation) 2>&1 | tee -a "$LOG" "$ATTEMPT_LOG"
status="${PIPESTATUS[0]}"
set -e
stop_monitor

completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "QWEN_REAL_ADAPTATION_TERMINAL_EXIT_STATUS $status"
  echo "QWEN_REAL_ADAPTATION_LAUNCH_END attempt_id=$attempt_id utc=$completion_utc status=$status"
  if [[ "$status" -eq 2 ]]; then
    echo "QWEN_REAL_ADAPTATION_LEGITIMATE_AUTOMATIC_GATE_FAILURE status=2"
  fi
} | tee -a "$LOG" "$EVENTS" "$ATTEMPT_LOG" "$ATTEMPT_EVENTS"
echo "# QWEN_REAL_ADAPTATION_GPU_MONITOR_END attempt_id=$attempt_id utc=$completion_utc status=$status" >>"$GPU_LOG"
echo "# QWEN_REAL_ADAPTATION_GPU_MONITOR_END attempt_id=$attempt_id utc=$completion_utc status=$status" >>"$ATTEMPT_GPU_LOG"
exit "$status"
