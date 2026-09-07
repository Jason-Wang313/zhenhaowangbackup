#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
RUNNER="$ROOT/smoke/run_foundir_qwen_export_partition_group_h200.sh"
GROUP="${1:-}"
PID_FILE="$ROOT/logs/foundir_qwen_export_partition_group_${GROUP}.pid"
SUPERVISOR_LOG="$ROOT/logs/foundir_qwen_export_partition_group_${GROUP}_supervisor.log"

cd "$ROOT"
[[ "$GROUP" == "0" || "$GROUP" == "1" ]] || {
  echo "ERROR: provide partition group 0 on the first H200 instance or 1 on the second" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
}
bash -n "$RUNNER" "$0"
if [[ "$GROUP" == "0" ]]; then
  bash "$ROOT/smoke/stop_foundir_five_type_flux2_main_v3_for_partition.sh"
fi
for relative in \
  agents.md \
  smoke/export_foundir_five_type_physical_v8_conditions_ddp.py \
  smoke/train_qwen_multidegradation_lora_ddp.py \
  smoke/run_foundir_qwen_export_partition_group_h200.sh \
  smoke/launch_foundir_qwen_export_partition_group_h200.sh \
  smoke/stop_foundir_five_type_flux2_main_v3_for_partition.sh \
  tests/test_foundir_five_type_main_v3.py; do
  git ls-files --error-unmatch "$relative" >/dev/null 2>&1 || {
    echo "ERROR: required file is not committed: $relative" >&2
    exit 1
  }
  git diff --quiet HEAD -- "$relative" || {
    echo "ERROR: required file differs from committed revision: $relative" >&2
    exit 1
  }
done

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(tr -dc '0-9' <"$PID_FILE")
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: partition group $GROUP is already active as PID $old_pid" >&2
    exit 1
  fi
fi
nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" "$GROUP" \
  >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "ERROR: partition group exited during launch; inspect $SUPERVISOR_LOG" >&2
  tail -100 "$SUPERVISOR_LOG" >&2 || true
  exit 1
fi
echo "FOUNDIR_QWEN_PARTITION_GROUP_DETACHED group=$GROUP pid=$pid"
echo "Supervisor log: $SUPERVISOR_LOG"
echo "Worker logs: logs/foundir_qwen_export_partition_*_of_8.log"
