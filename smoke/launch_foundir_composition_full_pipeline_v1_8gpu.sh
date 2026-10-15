#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_composition_whole_image_full_pipeline_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_composition_full_pipeline_v1.py"
RUNNER="$ROOT/smoke/run_foundir_composition_full_pipeline_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_composition_whole_image_full_pipeline_v1.log"
PID_FILE="$ROOT/logs/foundir_composition_whole_image_full_pipeline_v1.pid"
RESULT="$ROOT/checkpoints/foundir_composition_whole_image_full_pipeline_v1/result.json"
cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check is supported" >&2; exit 1; }
"$PYTHON" -m py_compile "$EVALUATOR"
bash -n "$RUNNER" "$0"
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --mode preflight
if [[ "${1:-}" == "--check" ]]; then echo "FOUNDIR_COMPOSITION_FULL_PIPELINE_LAUNCHER_CHECK_PASS"; exit 0; fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists at $RESULT" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!; printf '%s\n' "$pid" >"$PID_FILE"; sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FOUNDIR_COMPOSITION_FULL_PIPELINE_DETACHED process_group=$pid"
echo "LOG=$LOG"
echo "MONITOR=tail -f $LOG"
