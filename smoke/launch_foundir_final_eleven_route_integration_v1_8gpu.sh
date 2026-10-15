#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
BUILDER="$ROOT/smoke/build_foundir_final_eleven_route_integration_v1.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_mixed_restoration_pilot_ddp.py"
COMPARE="$ROOT/smoke/compare_foundir_final_eleven_route_integration_v1.py"
RUNNER="$ROOT/smoke/run_foundir_final_eleven_route_integration_v1_8gpu.sh"
ROUTED="$ROOT/configs/foundir_final_eleven_route_integration_routed_v1.json"
DIRECT="$ROOT/configs/foundir_final_eleven_route_integration_direct_v1.json"
LOG="$ROOT/logs/foundir_final_eleven_route_integration_v1.log"
PID_FILE="$ROOT/logs/foundir_final_eleven_route_integration_v1.pid"
RESULT="$ROOT/checkpoints/foundir_final_eleven_route_integration_v1/result.json"
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check supported" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export eight visible GPUs" >&2; exit 1; }
"$PYTHON" -m py_compile "$BUILDER" "$EVALUATOR" "$COMPARE"
bash -n "$RUNNER" "$0"
"$PYTHON" "$BUILDER" >/dev/null
"$PYTHON" "$EVALUATOR" --config "$ROUTED" --preflight
"$PYTHON" "$EVALUATOR" --config "$DIRECT" --preflight
echo "TOTAL END-TO-END TIME: H100 35-55 min (4.7-7.3 GPU-hours); H200 25-40 min (3.3-5.3 GPU-hours)."
if [[ "${1:-}" == "--check" ]]; then echo "FINAL_ELEVEN_ROUTE_INTEGRATION_LAUNCHER_CHECK_PASS"; exit 0; fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FINAL_ELEVEN_ROUTE_INTEGRATION_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
