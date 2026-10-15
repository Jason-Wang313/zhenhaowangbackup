#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
RUNNER="$ROOT/smoke/run_foundir_qwen_noise_lowlight_physical_holdout_v3_8gpu.sh"
CONFIG="$ROOT/configs/foundir_qwen_noise_lowlight_physical_holdout_v3.json"
LOG="$ROOT/logs/foundir_qwen_noise_lowlight_physical_holdout_v3_launcher.log"
PID="$ROOT/logs/foundir_qwen_noise_lowlight_physical_holdout_v3.pid"
cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || exit 1
bash -n "$RUNNER" "$0"
"$PYTHON" -m py_compile smoke/evaluate_foundir_qwen_noise_lowlight_physical_holdout_v3_ddp.py lowlight_restoration/noise_lowlight_physical_guard.py
"$PYTHON" smoke/evaluate_foundir_qwen_noise_lowlight_physical_holdout_v3_ddp.py --config "$CONFIG" --check-only
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash "$RUNNER" >>"$LOG" 2>&1 </dev/null &
process=$!
printf '%s\n' "$process" >"$PID"
sleep 3
kill -0 "$process"
echo "FOUNDIR_PHYSICAL_HOLDOUT_STARTED pid=$process"
echo "Monitor: tail -f $ROOT/logs/foundir_qwen_noise_lowlight_physical_holdout_v3.log"
