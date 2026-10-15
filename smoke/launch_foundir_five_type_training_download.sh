#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_five_type_training_download.sh"
LOG="$ROOT/logs/foundir_five_type_training_download_v1.log"
PIDFILE="$ROOT/data/foundir/_download_stage/state_v1/download.pid"

mkdir -p "$ROOT/logs" "$(dirname "$PIDFILE")"

if [[ -f "$PIDFILE" ]]; then
  prior_pid=$(awk -F= '$1 == "pid" {print $2}' "$PIDFILE")
  if [[ "$prior_pid" =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_ALREADY_RUNNING pid=$prior_pid log=$LOG"
    exit 0
  fi
fi

if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: download runner is not executable: $RUNNER" >&2
  exit 1
fi

nohup setsid "$RUNNER" >>"$LOG" 2>&1 </dev/null &
download_pid=$!
printf 'pid=%s\nstarted_utc=%s\nlog=%s\n' "$download_pid" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" >"$PIDFILE"
sleep 1
if ! kill -0 "$download_pid" 2>/dev/null; then
  echo "ERROR: detached download exited during startup; inspect $LOG" >&2
  tail -n 80 "$LOG" >&2 || true
  exit 1
fi

echo "FOUNDIR_FIVE_TYPE_DOWNLOAD_DETACHED pid=$download_pid"
echo "log=$LOG"
echo "monitor: tail -f $LOG"
