#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_lowlight_haze_restormer_v1.json"
TRAINER="$ROOT/smoke/train_foundir_lowlight_haze_restormer_ddp.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_lowlight_haze_restormer_ddp.py"
BUILDER="$ROOT/smoke/build_foundir_lowlight_haze_restormer_v1.py"
RUNNER="$ROOT/smoke/run_foundir_lowlight_haze_restormer_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_lowlight_haze_restormer_v1.log"
PID_FILE="$ROOT/logs/foundir_lowlight_haze_restormer_v1.pid"
RESULT="$ROOT/checkpoints/foundir_lowlight_haze_restormer_v1/final_comparison/result.json"
cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only --check supported" >&2; exit 1; }
"$PYTHON" -m py_compile "$BUILDER" "$TRAINER" "$EVALUATOR"
bash -n "$RUNNER" "$0"
"$PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$PYTHON" "$TRAINER" --config "$CONFIG" --preflight
echo "TOTAL END-TO-END TIME: H200 100-160 min (center 125), 13.3-21.3 GPU-hours (center 16.7); H100 130-200 min (center 160), 17.3-26.7 GPU-hours (center 21.3)."
echo "CONCURRENCY: one homogeneous node, 8 GPUs, DDP training followed sequentially by 3 milestone evaluations and one final 60-image comparison."
echo "MAIN UNCERTAINTY: full-resolution tiled validation/LPIPS and shared-storage image hashing throughput."
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_LOWLIGHT_HAZE_RESTORMER_V1_LAUNCHER_CHECK_PASS profiles=H100,H200"
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal comparison already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: startup failed; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FOUNDIR_LOWLIGHT_HAZE_RESTORMER_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
