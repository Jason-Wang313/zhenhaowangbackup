#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/smoke/run_foundir_blur_fftformer_external_diagnostic_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_blur_fftformer_external_diagnostic_v1.log"
PID_FILE="$ROOT/logs/foundir_blur_fftformer_external_diagnostic_v1.pid"
RESULT="$ROOT/checkpoints/foundir_blur_fftformer_external_diagnostic_v1/result.json"

cd "$ROOT"
mkdir -p "$ROOT/logs"
if [[ -e "$RESULT" ]]; then
  echo "Refusing to overwrite terminal diagnostic result: $RESULT" >&2
  exit 1
fi
if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "FFTformer diagnostic is already running with PID $(cat "$PID_FILE")" >&2
  exit 1
fi
nohup bash "$RUNNER" >"$LOG" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "FFTformer diagnostic started: pid=$pid log=$LOG"
