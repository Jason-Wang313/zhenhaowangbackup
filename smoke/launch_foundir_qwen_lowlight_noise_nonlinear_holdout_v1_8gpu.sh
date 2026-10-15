#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
RUNNER="$ROOT/smoke/run_foundir_qwen_lowlight_noise_nonlinear_holdout_v1_8gpu.sh"
CONFIG="$ROOT/configs/foundir_qwen_lowlight_noise_nonlinear_holdout_v1.json"
LOG="$ROOT/logs/foundir_qwen_lowlight_noise_nonlinear_holdout_v1_launcher.log"
PID_FILE="$ROOT/logs/foundir_qwen_lowlight_noise_nonlinear_holdout_v1.pid"
cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
bash -n "$RUNNER" "$0"; "$PYTHON" -m py_compile smoke/evaluate_foundir_qwen_lowlight_noise_nonlinear_holdout_v1_ddp.py; "$PYTHON" smoke/evaluate_foundir_qwen_lowlight_noise_nonlinear_holdout_v1_ddp.py --config "$CONFIG" --check-only; mkdir -p "$ROOT/logs"
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$RUNNER" >>"$LOG" 2>&1 </dev/null & pid=$!; printf '%s\n' "$pid" >"$PID_FILE"; sleep 3; kill -0 "$pid" 2>/dev/null || { echo "ERROR: launcher exited; inspect $LOG" >&2; exit 1; }
echo "FOUNDIR_QWEN_LOWLIGHT_NOISE_HOLDOUT_STARTED pid=$pid"; echo "Monitor: tail -f $ROOT/logs/foundir_qwen_lowlight_noise_nonlinear_holdout_v1.log"
