#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_lowlight_gate3_bundle.sh"
LOG="$ROOT/logs/foundir_lowlight_gate3_bundle_v1.log"
PID_FILE="$ROOT/logs/foundir_lowlight_gate3_bundle_v1.pid"

bash "$RUNNER" --check
mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(tr -cd '0-9' <"$PID_FILE")"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "FOUNDIR_LOWLIGHT_GATE3_BUNDLE_ALREADY_ACTIVE pid=$existing_pid"
    echo "Persistent log: $LOG"
    exit 0
  fi
fi

(
  cd "$ROOT"
  nohup bash "$RUNNER" >>"$LOG" 2>&1 </dev/null &
  launched_pid="$!"
  printf '%s\n' "$launched_pid" >"$PID_FILE"
  echo "FOUNDIR_LOWLIGHT_GATE3_BUNDLE_DETACHED pid=$launched_pid"
  echo "Persistent log: $LOG"
  echo "Monitor: tail -n 100 -f $LOG"
)
