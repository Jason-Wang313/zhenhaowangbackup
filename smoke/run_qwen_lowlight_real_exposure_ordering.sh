#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/qwen_lowlight_real_exposure_ordering_step600.json"
CONFIG_SHA256="817436c9011834852e9482c14fe9c008f1998a87e6f4811fd3b3b871929f1eba"
DRIVER="$ROOT/smoke/evaluate_qwen_lowlight_real_exposure_ordering.py"
STATUS_HELPER="$ROOT/smoke/qwen_lowlight_real_exposure_ordering_status.py"
BUILDER="$ROOT/smoke/build_qwen_lowlight_real_exposure_ordering_manifest.py"
OUTPUT="$ROOT/checkpoints/qwen_lowlight_real_exposure_ordering_step600"
SUMMARY="$OUTPUT/result_summary.json"
LOG="$ROOT/logs/qwen_lowlight_real_exposure_ordering_step600.log"
EVENTS="$ROOT/logs/qwen_lowlight_real_exposure_ordering_step600_events.log"
GPU_LOG="$ROOT/logs/qwen_lowlight_real_exposure_ordering_step600_gpu.csv"
LOCK="$ROOT/logs/qwen_lowlight_real_exposure_ordering_step600.lock"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

if [[ "$DEVICE_SET" != "0,1" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1" >&2
  exit 1
fi
WORLD_SIZE=2

REQUIRED_TRACKED_FILES=(
  agents.md
  SESSION_HANDOFF_2026-08-21.md
  configs/qwen_lowlight_real_exposure_ordering_step600.json
  configs/qwen_lowlight_real_validation_step600.json
  lowlight_restoration/exposure_ordering.py
  lowlight_restoration/qwen_constrained.py
  lowlight_restoration/qwen_supervised.py
  manifests/qwen_lowlight_corrective_holdout_candidates.csv
  manifests/qwen_lowlight_corrective_holdout_candidates.json
  manifests/qwen_lowlight_real_exposure_ordering.csv
  manifests/qwen_lowlight_real_exposure_ordering.json
  reports/foundir_lowlight_sota_strategy_2026-08-21.md
  smoke/build_qwen_lowlight_real_exposure_ordering_manifest.py
  smoke/evaluate_qwen_lowlight_real_exposure_ordering.py
  smoke/qwen_lowlight_real_exposure_ordering_status.py
  smoke/run_qwen_lowlight_real_exposure_ordering.sh
  smoke/launch_qwen_lowlight_real_exposure_ordering_tmux.sh
  tests/test_real_exposure_ordering.py
)

static_check() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: validated workspace Python is unavailable: $PYTHON" >&2
    return 1
  fi
  if [[ ! -f "$CONFIG" || ! -f "$DRIVER" || ! -f "$STATUS_HELPER" ]]; then
    echo "ERROR: exposure-ordering config or implementation is missing" >&2
    return 1
  fi
  if [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA256" ]]; then
    echo "ERROR: frozen exposure-ordering config hash changed" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_qwen_lowlight_real_exposure_ordering.sh"
  bash -n "$ROOT/smoke/launch_qwen_lowlight_real_exposure_ordering_tmux.sh"
  "$PYTHON" -m py_compile "$DRIVER" "$STATUS_HELPER" "$BUILDER" \
    "$ROOT/lowlight_restoration/exposure_ordering.py"
  "$PYTHON" -m unittest tests.test_real_exposure_ordering
  "$PYTHON" "$BUILDER"
  "$PYTHON" "$DRIVER" --config "$CONFIG" --contract-only --skip-visual-hash
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    if ! git -C "$ROOT" ls-files --error-unmatch "$relative_path" >/dev/null 2>&1; then
      echo "ERROR: required exposure-ordering file is not tracked: $relative_path" >&2
      return 1
    fi
  done
  if ! git -C "$ROOT" diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: required exposure-ordering files differ from the committed revision" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  static_check
  echo "QWEN_REAL_EXPOSURE_ORDERING_RUNNER_CHECK_PASS"
  echo "Selected devices: CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "Persistent log: $LOG"
  echo "Immutable output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected exposure-ordering runner argument: $1" >&2
  exit 1
fi

static_check
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: exposure-ordering diagnostic is already active" >&2
  exit 1
fi

mapfile -t gpu_inventory < <(
  nvidia-smi --id="$DEVICE_SET" --query-gpu=index,name,memory.total \
    --format=csv,noheader,nounits
)
if [[ "${#gpu_inventory[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "ERROR: expected two selected GPUs, found ${#gpu_inventory[@]}" >&2
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
export OMP_NUM_THREADS=16
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
      printf '%s,%s\n' "$sample_utc" "$gpu_row"
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
  echo "QWEN_REAL_EXPOSURE_ORDERING_BEGIN utc=$launch_utc"
  echo "workspace=$ROOT"
  echo "git_revision=$git_revision"
  echo "config=$CONFIG"
  echo "config_sha256=$CONFIG_SHA256"
  echo "CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "world_size=$WORLD_SIZE"
  echo "absolute_real_l1_l5_labels=false"
  echo "official_foundir_used=false"
  echo "output_dir=$OUTPUT"
} | tee -a "$LOG" "$EVENTS"
echo "sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz" >>"$GPU_LOG"
monitor_gpus >>"$GPU_LOG" &
monitor_pid="$!"

set +e
(
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node="$WORLD_SIZE" --max-restarts=0 \
    "$DRIVER" --config "$CONFIG"
) 2>&1 | tee -a "$LOG"
torchrun_status="${PIPESTATUS[0]}"
set -e
stop_monitor

terminal_status=1
if [[ "$torchrun_status" -eq 0 && -f "$SUMMARY" ]]; then
  set +e
  "$PYTHON" "$STATUS_HELPER" --config "$CONFIG" --summary "$SUMMARY" \
    2>&1 | tee -a "$LOG"
  terminal_status="${PIPESTATUS[0]}"
  set -e
  if [[ "$terminal_status" -ne 0 && "$terminal_status" -ne 2 ]]; then
    terminal_status=1
  fi
fi

completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "QWEN_REAL_EXPOSURE_ORDERING_TORCHRUN_STATUS $torchrun_status"
  echo "QWEN_REAL_EXPOSURE_ORDERING_TERMINAL_EXIT_STATUS $terminal_status"
  echo "QWEN_REAL_EXPOSURE_ORDERING_END utc=$completion_utc status=$terminal_status"
} | tee -a "$LOG" "$EVENTS"
exit "$terminal_status"
