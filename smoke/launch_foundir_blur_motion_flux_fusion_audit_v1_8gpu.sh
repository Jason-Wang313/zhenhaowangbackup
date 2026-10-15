#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_motion_flux_fusion_audit_v1.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_blur_motion_flux_fusion_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_blur_motion_flux_fusion_audit_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_motion_flux_fusion_audit_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_motion_flux_fusion_audit_v1.pid"
RESULT="$ROOT/checkpoints/foundir_blur_motion_flux_fusion_audit_v1/result.json"
cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check supported" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1; }
"$PYTHON" -m py_compile "$EVALUATOR"
bash -n "$RUNNER" "$0"
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --preflight
echo "TIME: 10-25 min on one 8xH100/H200 instance; 1.3-3.3 GPU-hours. Development only; no official/sealed test."
if [[ "${1:-}" == "--check" ]]; then echo "BLUR_MOTION_FLUX_FUSION_AUDIT_V1_LAUNCHER_CHECK_PASS"; exit 0; fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "BLUR_MOTION_FLUX_FUSION_AUDIT_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
