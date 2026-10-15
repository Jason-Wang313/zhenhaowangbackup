#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
WATCHER="$ROOT/smoke/watch_foundir_five_type_ready_install.sh"
LOG="$ROOT/logs/foundir_five_type_ready_install_watch_v1.log"
PIDFILE="$ROOT/data/foundir/_extract_stage/five_type_v1/state/ready_watch.pid"

mkdir -p "$ROOT/logs" "$(dirname "$PIDFILE")"
if [[ -f "$PIDFILE" ]]; then
  prior_pid=$(awk -F= '$1 == "pid" {print $2}' "$PIDFILE")
  if [[ "$prior_pid" =~ ^[0-9]+$ ]] && kill -0 "$prior_pid" 2>/dev/null; then
    echo "FOUNDIR_FIVE_TYPE_READY_WATCH_ALREADY_RUNNING pid=$prior_pid log=$LOG"
    exit 0
  fi
fi
if [[ ! -x "$WATCHER" ]]; then
  echo "ERROR: FoundIR ready-install watcher is not executable: $WATCHER" >&2
  exit 1
fi

nohup setsid "$WATCHER" >>"$LOG" 2>&1 </dev/null &
watch_pid=$!
printf 'pid=%s\nstarted_utc=%s\nlog=%s\n' "$watch_pid" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LOG" >"$PIDFILE"
sleep 1
if ! kill -0 "$watch_pid" 2>/dev/null; then
  echo "ERROR: ready-install watcher exited during startup; inspect $LOG" >&2
  tail -n 80 "$LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_FIVE_TYPE_READY_WATCH_DETACHED pid=$watch_pid log=$LOG"
