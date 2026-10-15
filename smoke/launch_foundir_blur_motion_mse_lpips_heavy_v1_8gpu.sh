#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_blur_motion_mse_lpips_heavy_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_motion_mse_lpips_heavy_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_motion_mse_lpips_heavy_v1.pid"

cd "$ROOT"
mkdir -p "$ROOT/logs"
[[ ! -e "$PID_FILE" ]] || { echo "ERROR: stale or active PID file exists: $PID_FILE" >&2; exit 1; }
nohup bash "$RUNNER" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  wait "$pid" || true
  rm -f "$PID_FILE"
  echo "ERROR: Blur training exited during launch; inspect $LOG" >&2
  exit 1
fi
echo "FOUNDIR_BLUR_MOTION_MSE_LPIPS_LAUNCHED pid=$pid log=$LOG expected_wall_time=45-90_minutes"
