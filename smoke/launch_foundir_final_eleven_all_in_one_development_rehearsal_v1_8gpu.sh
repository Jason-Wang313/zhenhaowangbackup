#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
BUILDER="$ROOT/smoke/build_foundir_final_eleven_all_in_one_development_rehearsal_v1.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_mixed_restoration_pilot_ddp.py"
SUMMARIZER="$ROOT/smoke/summarize_foundir_final_eleven_all_in_one_development_rehearsal_v1.py"
CONFIG="$ROOT/configs/foundir_final_eleven_all_in_one_development_rehearsal_v1.json"
RUNNER="$ROOT/smoke/run_foundir_final_eleven_all_in_one_development_rehearsal_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_final_eleven_all_in_one_development_rehearsal_v1.log"
PID_FILE="$ROOT/logs/foundir_final_eleven_all_in_one_development_rehearsal_v1.pid"
TERMINAL="$ROOT/checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v1/terminal_result.json"
cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check supported" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export eight visible GPUs" >&2; exit 1; }
"$PYTHON" -m py_compile "$BUILDER" "$EVALUATOR" "$SUMMARIZER"
bash -n "$RUNNER" "$0"
"$PYTHON" "$BUILDER" >/dev/null
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --preflight
echo "TOTAL END-TO-END TIME: H100 45-65 min (6.0-8.7 GPU-hours); H200 30-45 min (4.0-6.0 GPU-hours)."
if [[ "${1:-}" == "--check" ]]; then echo "FINAL_ELEVEN_ALL_IN_ONE_LAUNCHER_CHECK_PASS"; exit 0; fi
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$TERMINAL" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!; printf '%s\n' "$pid" >"$PID_FILE"; sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FINAL_ELEVEN_ALL_IN_ONE_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
