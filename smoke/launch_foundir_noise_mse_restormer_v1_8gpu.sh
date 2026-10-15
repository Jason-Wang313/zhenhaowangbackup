#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
CONFIG="$ROOT/configs/foundir_noise_mse_restormer_v1.json"
RUNNER="$ROOT/smoke/run_foundir_noise_mse_restormer_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_noise_mse_restormer_v1.log"
PID_FILE="$ROOT/logs/foundir_noise_mse_restormer_v1.pid"

cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
"$PYTHON" -m py_compile "$TRAINER"
bash -n "$RUNNER" "$0"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
  tests/test_task_native_restormer.py tests/test_foundir_task_native_restormer_adaptation.py
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
echo "EXPECTED WALL TIME: H200 30-50 min; H100 45-70 min."
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FOUNDIR_NOISE_MSE_DETACHED process_group=$pid"
echo "Monitor: tail -f $LOG"
