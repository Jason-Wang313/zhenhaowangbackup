#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
RUNNER="$ROOT/smoke/run_foundir_lowlight_luma_rescue_gate_v1_8gpu.sh"
EVALUATOR="$ROOT/smoke/evaluate_foundir_lowlight_luma_rescue_gate_ddp.py"
CONFIG="$ROOT/configs/foundir_lowlight_luma_rescue_gate_v1.json"
LOG="$ROOT/logs/foundir_lowlight_luma_rescue_gate_v1.log"
GPU_LOG="$ROOT/logs/foundir_lowlight_luma_rescue_gate_v1_gpu.csv"
PID_FILE="$ROOT/logs/foundir_lowlight_luma_rescue_gate_v1.pid"

cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}
for command in setsid flock sha256sum; do
  command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }
done
for path in "$PYTHON" "$RUNNER" "$EVALUATOR" "$CONFIG" \
  "$ROOT/configs/foundir_five_type_qwen_routed_integrated_v2.json" \
  "$ROOT/manifests/foundir_five_type_all_qwen_conditions_v3.csv" \
  "$ROOT/third_party/foundir/premodel/model-2000.pt" \
  "$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/protocol_v4_isolated/blur_generalist_specialization_gate_v2/step_003000.pt" \
  "$ROOT/third_party/foundir/specialist_model/ckpt/lowlight.pth"; do
  [[ -f "$path" ]] || { echo "ERROR: missing low-light rescue dependency $path" >&2; exit 1; }
done
"$PYTHON" -m py_compile "$EVALUATOR"
"$PYTHON" -m json.tool "$CONFIG" >/dev/null
bash -n "$RUNNER" "$0"
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_LOWLIGHT_LUMA_RESCUE_GATE_V1_FAST_LAUNCHER_CHECK_PASS"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
"$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise RuntimeError("low-light luma rescue gate requires exactly eight visible GPUs")
for index in range(8):
    properties = torch.cuda.get_device_properties(index)
    if not any(name in properties.name for name in ("H100", "H200")):
        raise RuntimeError(f"GPU {index} is not H100/H200: {properties.name}")
    if properties.total_memory < 75 * 1024**3:
        raise RuntimeError(f"GPU {index} has insufficient memory")
print("FOUNDIR_LOWLIGHT_LUMA_RESCUE_GATE_V1_GPU_CHECK_PASS", flush=True)
PY

mkdir -p "$ROOT/logs"
if [[ -f "$PID_FILE" ]]; then
  previous_pid=$(<"$PID_FILE")
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "ERROR: existing low-light rescue coordinator is live at PID $previous_pid" >&2
    exit 1
  fi
fi
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" \
  >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 5
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: low-light rescue gate exited during launch; inspect $LOG" >&2
  tail -120 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_LOWLIGHT_LUMA_RESCUE_GATE_V1_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "GPU telemetry: $GPU_LOG"
echo "Stop safely: kill -- -$pid"
