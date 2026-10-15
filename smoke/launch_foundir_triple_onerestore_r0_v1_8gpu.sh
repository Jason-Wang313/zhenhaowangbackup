#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
TRAIN_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
METRIC_PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
TRAINER="$ROOT/smoke/train_foundir_triple_onerestore_r0_ddp.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_triple_onerestore_r0_pyiqa_ddp.py"
RUNNER="$ROOT/smoke/run_foundir_triple_onerestore_r0_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_triple_onerestore_r0_v1.log"
PID_FILE="$ROOT/logs/foundir_triple_onerestore_r0_v1.pid"
RESULT="$ROOT/checkpoints/foundir_triple_onerestore_r0_v1/result.json"
REPORT="$ROOT/reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only --check is supported" >&2
  exit 1
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
"$TRAIN_PYTHON" -m py_compile "$TRAINER" "$EVALUATOR"
bash -n "$RUNNER" "$0"
"$TRAIN_PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$TRAIN_PYTHON" "$TRAINER" --config "$ROOT/configs/foundir_triple_onerestore_r0_v1.json" --preflight
TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$METRIC_PYTHON" "$EVALUATOR" --config "$ROOT/configs/foundir_triple_onerestore_r0_v1.json" --preflight

echo "TOTAL NORMAL TIME: H200 0.75-2.0 hours (center 1.25); H100 1.0-2.5 hours (center 1.5)."
echo "AGGREGATE GPU-HOURS: H200 6-16 (center 10); H100 8-20 (center 12)."
echo "CONCURRENCY: one homogeneous node, exactly 8 H100 or 8 H200 GPUs; three arms run sequentially, each arm uses 8-way DDP."
echo "INCLUDED: 3x1200-step training, three-arm full-image export, LPIPS/PSNR, aggregation, sidecars, and terminal report."
echo "MAIN UNCERTAINTY: no GPU is visible in this shell, so no external-node no-update benchmark was possible; VGG16 CDRL backward and DDP/shared-storage throughput dominate the range."

if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_TRIPLE_ONERESTORE_R0_V1_LAUNCHER_CHECK_PASS profiles=H100,H200"
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 1
}
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -e "$RESULT" ]] || {
  echo "ERROR: terminal R0 result already exists; frozen R0 must not be rerun" >&2
  exit 1
}
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: startup failed; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_TRIPLE_ONERESTORE_R0_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
