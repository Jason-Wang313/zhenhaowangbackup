#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_noise_rain_task_native_capacity_v1_8gpu.sh"
EVALUATOR="$ROOT/smoke/evaluate_foundir_noise_rain_task_native_capacity_ddp.py"
MODEL_CODE="$ROOT/lowlight_restoration/task_native_restormer.py"
CONFIG="$ROOT/configs/foundir_noise_rain_task_native_capacity_v1.json"
CHECKPOINT_DIR="$ROOT/models/restormer-task-native"
LOG="$ROOT/logs/foundir_noise_rain_task_native_capacity_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_rain_task_native_capacity_v1_gpu.csv"
PID_FILE="$ROOT/logs/foundir_noise_rain_task_native_capacity_v1.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"

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
for path in "$PYTHON" "$RUNNER" "$EVALUATOR" "$MODEL_CODE" "$CONFIG" \
  "$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/noise_rain_full_development_audit_v1/result.json" \
  "$CHECKPOINT_DIR/deraining.pth" "$CHECKPOINT_DIR/real_denoising.pth" \
  "$CHECKPOINT_DIR/gaussian_color_denoising_blind.pth"; do
  [[ -f "$path" ]] || { echo "ERROR: missing task-native dependency $path" >&2; exit 1; }
done
[[ "$(stat -c %s "$CHECKPOINT_DIR/deraining.pth")" == 104700429 ]] || exit 1
[[ "$(stat -c %s "$CHECKPOINT_DIR/real_denoising.pth")" == 104611957 ]] || exit 1
[[ "$(stat -c %s "$CHECKPOINT_DIR/gaussian_color_denoising_blind.pth")" == 104611957 ]] || exit 1
[[ "$(sha256sum "$CHECKPOINT_DIR/deraining.pth" | cut -d' ' -f1)" == 225ed9acf80b3527d7080e5182918f642a23a78ef8593ad700b7e924bd660681 ]] || exit 1
[[ "$(sha256sum "$CHECKPOINT_DIR/real_denoising.pth" | cut -d' ' -f1)" == 4cae18bed8a291b9a5deeeab48756bbe6f61fcf7f31a4cca0969b24608601387 ]] || exit 1
[[ "$(sha256sum "$CHECKPOINT_DIR/gaussian_color_denoising_blind.pth" | cut -d' ' -f1)" == 4e2314dae77bb907184da6e2ac625b921491d6b33de26a52512eff1c6bb37fbd ]] || exit 1

"$PYTHON" -m py_compile "$EVALUATOR" "$MODEL_CODE"
bash -n "$RUNNER" "$0"
"$PYTHON" -u "$EVALUATOR" --config "$CONFIG" --preflight
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_LAUNCHER_CHECK_PASS"
  exit 0
fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }

mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: task-native capacity audit exited during launch; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_NOISE_RAIN_TASK_NATIVE_CAPACITY_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "GPU telemetry: $GPU_LOG"
echo "Stop safely: kill -- -$pid"
