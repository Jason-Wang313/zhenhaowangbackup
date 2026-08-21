#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/qwen_lowlight_real_adaptation_step600.json"
DRIVER="$ROOT/smoke/diagnose_qwen_real_adaptation_checkpoints.py"
PREFLIGHT="$ROOT/smoke/qwen_lowlight_real_adaptation_preflight.py"
OUTPUT_BASE="$ROOT/checkpoints/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821"
LOG="$ROOT/logs/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821.log"
EVENTS="$ROOT/logs/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821_events.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821_gpu.csv"
LOCK="$ROOT/logs/qwen_lowlight_real_adaptation_checkpoint_diagnostics_20260821.lock"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

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
  configs/qwen_lowlight_real_adaptation_step600.json
  manifests/low_light_pilot_10000.csv
  manifests/qwen_lowlight_real_adaptation.csv
  lowlight_restoration/data.py
  lowlight_restoration/qwen_pilot.py
  lowlight_restoration/qwen_real_adaptation.py
  lowlight_restoration/qwen_real_validation.py
  lowlight_restoration/qwen_supervised.py
  smoke/diagnose_qwen_real_adaptation_checkpoints.py
  smoke/qwen_lowlight_real_adaptation_preflight.py
  smoke/qwen_lowlight_supervised_preflight.py
  smoke/train_qwen_lowlight_real_adaptation_ddp.py
  smoke/train_qwen_lowlight_supervised.py
  smoke/train_qwen_lowlight_supervised_ddp.py
  smoke/run_qwen_real_adaptation_checkpoint_diagnostics.sh
  smoke/launch_qwen_real_adaptation_checkpoint_diagnostics_tmux.sh
)

static_check() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
    return 1
  fi
  if [[ ! -f "$CONFIG" || ! -f "$DRIVER" || ! -f "$PREFLIGHT" ]]; then
    echo "ERROR: diagnostic config, driver, or preflight is missing" >&2
    return 1
  fi
  if [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "0a817d70d88f172d6d16d37853ce9c3cd77308446b3eed3d5e5cdab0c3b3b541" ]]; then
    echo "ERROR: frozen real-adaptation config hash changed" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_qwen_real_adaptation_checkpoint_diagnostics.sh"
  bash -n "$ROOT/smoke/launch_qwen_real_adaptation_checkpoint_diagnostics_tmux.sh"
  "$PYTHON" -m py_compile "$DRIVER"
  "$PYTHON" "$PREFLIGHT" --config "$CONFIG" --contract-only \
    --skip-large-model-hashes --no-write-artifact
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    if ! git -C "$ROOT" ls-files --error-unmatch "$relative_path" >/dev/null 2>&1; then
      echo "ERROR: required diagnostic file is not tracked: $relative_path" >&2
      return 1
    fi
  done
  if ! git -C "$ROOT" diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: required diagnostic files differ from the committed revision" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  static_check
  echo "QWEN_REAL_ADAPTATION_DIAGNOSTIC_RUNNER_CHECK_PASS"
  echo "Selected devices: CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "World size: $WORLD_SIZE"
  echo "Immutable output base: $OUTPUT_BASE"
  echo "Persistent log: $LOG"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected diagnostic runner argument: $1" >&2
  exit 1
fi

static_check
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: checkpoint diagnostics are already active" >&2
  exit 1
fi

mapfile -t gpu_inventory < <(
  nvidia-smi --id="$DEVICE_SET" --query-gpu=index,name,memory.total \
    --format=csv,noheader,nounits
)
if [[ "${#gpu_inventory[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "ERROR: expected $WORLD_SIZE selected GPUs, found ${#gpu_inventory[@]}" >&2
  exit 1
fi
IFS=',' read -r -a expected_indices <<<"$DEVICE_SET"
for position in "${!gpu_inventory[@]}"; do
  row="${gpu_inventory[$position]}"
  if [[ "$row" =~ ^[[:space:]]*([0-9]+),[[:space:]]*NVIDIA[[:space:]]+H(100|200)[^,]*,[[:space:]]*([0-9]+) ]]; then
    gpu_index="${BASH_REMATCH[1]}"
    gpu_memory_mib="${BASH_REMATCH[3]}"
  else
    echo "ERROR: selected GPU must be an H100/H200 with parseable memory: $row" >&2
    exit 1
  fi
  if [[ "$gpu_index" != "${expected_indices[$position]}" || "$gpu_memory_mib" -lt 76800 ]]; then
    echo "ERROR: selected GPU identity or memory differs: $row" >&2
    exit 1
  fi
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$OMP_THREADS"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

attempt_id="$(date -u +%Y%m%dT%H%M%SZ)-$$"
OUTPUT="$OUTPUT_BASE/attempt_$attempt_id"
monitor_pid=""
stop_monitor() {
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    monitor_pid=""
  fi
}
trap stop_monitor EXIT

monitor_gpus() {
  while true; do
    sample_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while IFS= read -r gpu_row; do
      printf '%s,%s,%s\n' "$attempt_id" "$sample_utc" "$gpu_row"
    done < <(
      nvidia-smi --id="$DEVICE_SET" \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
        --format=csv,noheader,nounits
    )
    sleep 2
  done
}

launch_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git_revision="$(git -C "$ROOT" rev-parse HEAD)"
{
  echo "QWEN_REAL_ADAPTATION_DIAGNOSTICS_BEGIN attempt_id=$attempt_id utc=$launch_utc"
  echo "workspace=$ROOT"
  echo "git_revision=$git_revision"
  echo "config=$CONFIG"
  echo "CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "world_size=$WORLD_SIZE"
  echo "steps=25,50,100,150,200"
  echo "real_test_inference=false"
  echo "official_foundir_used=false"
  echo "output_dir=$OUTPUT"
} | tee -a "$LOG" "$EVENTS"
echo "attempt_id,sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz" >>"$GPU_LOG"
monitor_gpus >>"$GPU_LOG" &
monitor_pid="$!"

set +e
(
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node="$WORLD_SIZE" --max-restarts=0 \
    "$DRIVER" --config "$CONFIG" --output-dir "$OUTPUT" \
    --steps 25 50 100 150 200
) 2>&1 | tee -a "$LOG"
status="${PIPESTATUS[0]}"
set -e
stop_monitor
completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "QWEN_REAL_ADAPTATION_DIAGNOSTICS_TERMINAL_EXIT_STATUS $status"
  echo "QWEN_REAL_ADAPTATION_DIAGNOSTICS_END attempt_id=$attempt_id utc=$completion_utc status=$status"
} | tee -a "$LOG" "$EVENTS"
exit "$status"
