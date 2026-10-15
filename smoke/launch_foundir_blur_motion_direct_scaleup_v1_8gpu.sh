#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_motion_direct_scaleup_v1.json"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_blur_motion_direct_scaleup_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_motion_direct_scaleup_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_motion_direct_scaleup_v1.pid"
RESULT="$ROOT/checkpoints/foundir_blur_motion_direct_scaleup_v1/development_selection.json"
cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check supported" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1; }
"$PYTHON" -m py_compile "$TRAINER"
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q tests/test_task_native_restormer.py tests/test_blur_restormer_refinement.py tests/test_foundir_task_native_restormer_adaptation.py
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
echo "TIME: first 1k stop gate 65-75 min; full 9k 9.5-11 h on one 8xH100 instance; full budget 76-88 GPU-hours."
if [[ "${1:-}" == "--check" ]]; then echo "BLUR_MOTION_DIRECT_SCALEUP_V1_LAUNCHER_CHECK_PASS"; exit 0; fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal selection already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "BLUR_MOTION_DIRECT_SCALEUP_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
