#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
RUNNER="$ROOT/smoke/run_foundir_rain_restormer_adaptation_v2_8gpu.sh"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
CONFIG="$ROOT/configs/foundir_rain_restormer_adaptation_v2_8gpu.json"
LOG="$ROOT/logs/foundir_rain_restormer_adaptation_v2_8gpu.log"
GPU_LOG="$ROOT/logs/foundir_rain_restormer_adaptation_v2_8gpu_gpu.csv"
PID_FILE="$ROOT/logs/foundir_rain_restormer_adaptation_v2_8gpu.pid"
OLD_PID_FILE="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1.pid"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
for path in "$PYTHON" "$RUNNER" "$TRAINER" "$CONFIG" \
  "$ROOT/configs/foundir_rain_restormer_adaptation_v1.json" \
  "$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/rain_restormer_adaptation_v1/step_001500.pt" \
  "$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/rain_restormer_adaptation_v1/development_step_001500.json"; do
  [[ -f "$path" ]] || { echo "ERROR: missing Rain scale-up dependency $path" >&2; exit 1; }
done
"$PYTHON" -m py_compile "$TRAINER"
"$PYTHON" -m json.tool "$CONFIG" >/dev/null
bash -n "$RUNNER" "$0"
"$PYTHON" -u "$TRAINER" --config "$CONFIG" --preflight
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_LAUNCHER_CHECK_PASS"
  exit 0
fi

for command in nvidia-smi awk seq; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done

"$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise RuntimeError("Rain v2 scale-up requires exactly eight visible GPUs")
for index in range(8):
    p = torch.cuda.get_device_properties(index)
    if "H200" not in p.name or p.total_memory < 130 * 1024**3:
        raise RuntimeError(f"Rain v2 scale-up requires eight H200 GPUs; GPU {index}={p.name}")
print("FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_GPU_CHECK_PASS", flush=True)
PY

if [[ -f "$OLD_PID_FILE" ]]; then
  old_pid=$(<"$OLD_PID_FILE")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_STOPPING_V1 process_group=$old_pid"
    kill -- -"$old_pid"
    for _ in $(seq 1 60); do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$old_pid" 2>/dev/null && {
      echo "ERROR: old v1 coordinator did not stop" >&2
      exit 1
    }
  fi
fi
for _ in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 2000 {n++} END {print n+0}')
  [[ "$busy" == 0 ]] && break
  sleep 1
done
busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 2000 {n++} END {print n+0}')
[[ "$busy" == 0 ]] || { echo "ERROR: H200 workers did not release all GPUs" >&2; exit 1; }

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  previous_pid=$(<"$PID_FILE")
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "ERROR: existing Rain v2 coordinator is live at PID $previous_pid" >&2
    exit 1
  fi
fi
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 5
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: Rain v2 scale-up exited during launch; inspect $LOG" >&2
  tail -120 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V2_8GPU_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "GPU telemetry: $GPU_LOG"
echo "Stop safely: kill -- -$pid"
