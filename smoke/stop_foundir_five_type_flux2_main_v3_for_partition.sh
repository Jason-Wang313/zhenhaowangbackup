#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PID_FILE="$ROOT/logs/foundir_five_type_flux2_main_v3.pid"
EXPECTED="$ROOT/smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh"

cd "$ROOT"
[[ -f "$PID_FILE" ]] || {
  echo "FOUNDIR_FIVE_TYPE_MAIN_V3_ALREADY_INACTIVE no_pid_file"
  exit 0
}
pipeline_pid=$(tr -dc '0-9' <"$PID_FILE")
[[ -n "$pipeline_pid" ]] || { echo "ERROR: v3 PID file is malformed" >&2; exit 1; }
if ! kill -0 "$pipeline_pid" 2>/dev/null; then
  echo "FOUNDIR_FIVE_TYPE_MAIN_V3_ALREADY_INACTIVE pid=$pipeline_pid"
  exit 0
fi
command_line=$(ps -p "$pipeline_pid" -o args=)
[[ "$command_line" == *"bash $EXPECTED"* ]] || {
  echo "ERROR: refusing to signal unrelated PID $pipeline_pid: $command_line" >&2
  exit 1
}

mapfile -t children < <(pgrep -P "$pipeline_pid" || true)
for child in "${children[@]}"; do
  kill -TERM "$child" 2>/dev/null || true
done
if [[ "${#children[@]}" -eq 0 ]]; then
  kill -TERM "$pipeline_pid" 2>/dev/null || true
fi
for _ in $(seq 1 60); do
  if ! kill -0 "$pipeline_pid" 2>/dev/null; then
    echo "FOUNDIR_FIVE_TYPE_MAIN_V3_STOPPED_FOR_PARTITION pid=$pipeline_pid"
    exit 0
  fi
  sleep 1
done
echo "ERROR: v3 pipeline did not stop cleanly within 60 seconds; no force kill was used" >&2
exit 1
