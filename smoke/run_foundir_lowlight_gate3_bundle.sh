#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
COMPACT="$ROOT/smoke/run_foundir_lowlight_gate3_compact.sh"
QWEN="$ROOT/smoke/run_qwen_lowlight_real_exposure_ordering.sh"
LOCK="$ROOT/logs/foundir_lowlight_gate3_bundle_v1.lock"

check_bundle() {
  CUDA_VISIBLE_DEVICES=2,3 bash "$COMPACT" --check
  CUDA_VISIBLE_DEVICES=0,1 bash "$QWEN" --check
  echo "FOUNDIR_LOWLIGHT_GATE3_BUNDLE_CHECK_PASS"
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  check_bundle
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected Gate 3 bundle argument: $1" >&2
  exit 1
fi

check_bundle
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: Gate 3 bundle is already active" >&2
  exit 1
fi

GPU_LOG="$ROOT/logs/foundir_lowlight_gate3_bundle_v1_gpu.csv"
monitor_pid=""
compact_pid=""
qwen_pid=""
stop_children() {
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$compact_pid" ]]; then kill "$compact_pid" 2>/dev/null || true; fi
  if [[ -n "$qwen_pid" ]]; then kill "$qwen_pid" 2>/dev/null || true; fi
}
trap stop_children EXIT INT TERM

monitor_gpus() {
  while true; do
    utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while IFS= read -r row; do printf '%s,%s\n' "$utc" "$row"; done < <(
      nvidia-smi --id=0,1,2,3 \
        --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm \
        --format=csv,noheader,nounits
    )
    sleep 2
  done
}

echo "sample_utc,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,power_draw_w,sm_clock_mhz" >>"$GPU_LOG"
monitor_gpus >>"$GPU_LOG" &
monitor_pid="$!"
echo "FOUNDIR_LOWLIGHT_GATE3_BUNDLE_BEGIN utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
(cd "$ROOT" && CUDA_VISIBLE_DEVICES=0,1 bash "$QWEN") &
qwen_pid="$!"
(cd "$ROOT" && CUDA_VISIBLE_DEVICES=2,3 bash "$COMPACT") &
compact_pid="$!"
wait "$qwen_pid"
qwen_status="$?"
qwen_pid=""
wait "$compact_pid"
compact_status="$?"
compact_pid=""
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
monitor_pid=""

terminal_status=0
if [[ "$compact_status" -ne 0 ]]; then
  terminal_status=1
elif [[ "$qwen_status" -eq 2 ]]; then
  terminal_status=2
elif [[ "$qwen_status" -ne 0 ]]; then
  terminal_status=1
fi
echo "FOUNDIR_LOWLIGHT_GATE3_BUNDLE_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) compact_status=$compact_status qwen_status=$qwen_status status=$terminal_status"
exit "$terminal_status"
