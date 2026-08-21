#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/lowlight_vae_ceiling_v1.json"
CONFIG_SHA256="04d8c261633635dc57a7c79d0b109035e03801059fb2a18cf5899088d8db96a3"
DRIVER="$ROOT/smoke/evaluate_lowlight_vae_ceiling_ddp.py"
STATUS_HELPER="$ROOT/smoke/lowlight_vae_ceiling_status.py"
OUTPUT="$ROOT/checkpoints/lowlight_vae_ceiling_v1"
SUMMARY="$OUTPUT/result_summary.json"
LOG="$ROOT/logs/lowlight_vae_ceiling_v1.log"
EVENTS="$ROOT/logs/lowlight_vae_ceiling_v1_events.log"
GPU_LOG="$ROOT/logs/lowlight_vae_ceiling_v1_gpu.csv"
LOCK="$ROOT/logs/lowlight_vae_ceiling_v1.lock"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"
WORLD_SIZE=2

REQUIRED_TRACKED_FILES=(
  agents.md
  SESSION_HANDOFF_2026-08-21.md
  configs/lowlight_vae_ceiling_v1.json
  lowlight_restoration/paired_development.py
  lowlight_restoration/representation_ceiling.py
  manifests/lowlight_real_paired_development_v1.csv
  manifests/lowlight_real_paired_development_v1.json
  reports/foundir_lowlight_sota_strategy_2026-08-21.md
  reports/lowlight_real_photometric_ceiling_result_2026-08-21.md
  smoke/evaluate_lowlight_vae_ceiling_ddp.py
  smoke/lowlight_vae_ceiling_status.py
  smoke/run_lowlight_vae_ceiling.sh
  smoke/launch_lowlight_vae_ceiling_tmux.sh
  tests/test_paired_development.py
  tests/test_representation_ceiling.py
  tests/test_vae_ceiling_status.py
)

if [[ "$DEVICE_SET" != "0,1" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1" >&2
  exit 1
fi

static_check() {
  if [[ ! -x "$PYTHON" || ! -f "$CONFIG" || ! -f "$DRIVER" || ! -f "$STATUS_HELPER" ]]; then
    echo "ERROR: VAE-ceiling environment or implementation is missing" >&2
    return 1
  fi
  if [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA256" ]]; then
    echo "ERROR: frozen VAE-ceiling config hash changed" >&2
    return 1
  fi
  if [[ "$(sha256sum "$ROOT/models/torch/hub/checkpoints/alexnet-owt-7be5be79.pth" | awk '{print $1}')" != "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02" ]]; then
    echo "ERROR: offline AlexNet LPIPS backbone is missing or changed" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_lowlight_vae_ceiling.sh"
  bash -n "$ROOT/smoke/launch_lowlight_vae_ceiling_tmux.sh"
  "$PYTHON" -m py_compile "$DRIVER" "$STATUS_HELPER" \
    "$ROOT/lowlight_restoration/representation_ceiling.py"
  "$PYTHON" -m unittest tests.test_representation_ceiling tests.test_vae_ceiling_status
  "$PYTHON" "$DRIVER" --config "$CONFIG" --contract-only
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    if ! git -C "$ROOT" ls-files --error-unmatch "$relative_path" >/dev/null 2>&1; then
      echo "ERROR: required VAE-ceiling file is not tracked: $relative_path" >&2
      return 1
    fi
  done
  if ! git -C "$ROOT" diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: required VAE-ceiling files differ from the committed revision" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  static_check
  echo "LOWLIGHT_VAE_CEILING_RUNNER_CHECK_PASS"
  echo "Selected devices: CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "Persistent log: $LOG"
  echo "Immutable output: $OUTPUT"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected VAE-ceiling runner argument: $1" >&2
  exit 1
fi

static_check
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: VAE-ceiling diagnostic is already active" >&2
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

export AE_MODEL_PATH="$ROOT/models/flux2-klein-base-4b/ae.safetensors"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
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
  echo "LOWLIGHT_VAE_CEILING_BEGIN utc=$launch_utc"
  echo "workspace=$ROOT"
  echo "git_revision=$git_revision"
  echo "config=$CONFIG"
  echo "config_sha256=$CONFIG_SHA256"
  echo "CUDA_VISIBLE_DEVICES=$DEVICE_SET"
  echo "world_size=$WORLD_SIZE"
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
  if [[ "$terminal_status" -ne 0 ]]; then
    terminal_status=1
  fi
fi

completion_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
  echo "LOWLIGHT_VAE_CEILING_TORCHRUN_STATUS $torchrun_status"
  echo "LOWLIGHT_VAE_CEILING_TERMINAL_EXIT_STATUS $terminal_status"
  echo "LOWLIGHT_VAE_CEILING_END utc=$completion_utc status=$terminal_status"
} | tee -a "$LOG" "$EVENTS"
exit "$terminal_status"
