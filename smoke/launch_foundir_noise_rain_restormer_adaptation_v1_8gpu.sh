#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
RUNNER="$ROOT/smoke/run_foundir_noise_rain_restormer_adaptation_v1_8gpu.sh"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
MODEL_CODE="$ROOT/lowlight_restoration/task_native_restormer.py"
NOISE_CONFIG="$ROOT/configs/foundir_noise_restormer_adaptation_v1.json"
RAIN_CONFIG="$ROOT/configs/foundir_rain_restormer_adaptation_v1.json"
CAPACITY="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/noise_rain_task_native_capacity_v1/result.json"
LOG="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1_gpu.csv"
PID_FILE="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1.pid"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
(( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )) || {
  echo "ERROR: Bash 5.1 or newer is required" >&2
  exit 1
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
for path in "$PYTHON" "$RUNNER" "$TRAINER" "$MODEL_CODE" "$NOISE_CONFIG" "$RAIN_CONFIG" \
  "$CAPACITY" "$ROOT/manifests/foundir_five_type_all_qwen_conditions_v3.csv" \
  "$ROOT/models/restormer-task-native/real_denoising.pth" \
  "$ROOT/models/restormer-task-native/deraining.pth"; do
  [[ -f "$path" ]] || { echo "ERROR: missing Noise/Rain adaptation dependency $path" >&2; exit 1; }
done

"$PYTHON" -m py_compile "$TRAINER" "$MODEL_CODE"
"$PYTHON" -m json.tool "$NOISE_CONFIG" >/dev/null
"$PYTHON" -m json.tool "$RAIN_CONFIG" >/dev/null
bash -n "$RUNNER" "$0"
"$PYTHON" -u "$TRAINER" --config "$NOISE_CONFIG" --preflight
"$PYTHON" -u "$TRAINER" --config "$RAIN_CONFIG" --preflight
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_LAUNCHER_CHECK_PASS"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise RuntimeError("Noise/Rain adaptation requires exactly eight visible CUDA GPUs")
devices = []
for index in range(8):
    properties = torch.cuda.get_device_properties(index)
    if not any(name in properties.name for name in ("H100", "H200")):
        raise RuntimeError(f"GPU {index} is not H100/H200: {properties.name}")
    if properties.total_memory < 75 * 1024**3:
        raise RuntimeError(f"GPU {index} has insufficient memory: {properties.total_memory}")
    devices.append({"index": index, "name": properties.name, "bytes": properties.total_memory})
print(f"FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_GPU_CHECK_PASS {devices}", flush=True)
PY

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  previous_pid=$(<"$PID_FILE")
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "ERROR: existing adaptation coordinator is live at PID $previous_pid" >&2
    exit 1
  fi
fi
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 5
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: Noise/Rain adaptation exited during launch; inspect $LOG" >&2
  tail -120 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_DETACHED process_group=$pid"
echo "Persistent coordinator log: $LOG"
echo "Noise log: $ROOT/logs/foundir_noise_restormer_adaptation_v1.log"
echo "Rain log: $ROOT/logs/foundir_rain_restormer_adaptation_v1.log"
echo "GPU telemetry: $GPU_LOG"
echo "Stop safely: kill -- -$pid"
