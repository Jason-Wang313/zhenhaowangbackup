#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
SELF="$ROOT/smoke/launch_zimage_deterministic_bridge_tiny_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1.json"
SCRIPT="$ROOT/smoke/train_zimage_deterministic_bridge_tiny.py"
OUT="$ROOT/checkpoints/foundir_blur_only_zimage_deterministic_bridge_tiny_v1"
LOG="$ROOT/logs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1_gpu.csv"
STARTUP="$ROOT/logs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1_startup.log"
PID_FILE="$ROOT/logs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1.pid"
SESSION=foundir-zimage-layer1-bridge-tiny-v1

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_zimage_layer1_bridge_v1"

[[ $# -eq 0 || ( $# -eq 1 && ( "$1" == --check || "$1" == --run ) ) ]] || {
  echo "usage: $0 [--check|--run]" >&2; exit 2;
}
[[ -x "$PYTHON" && -f "$CONFIG" && -f "$SCRIPT" ]] || { echo 'ERROR: missing Layer-1 runtime input' >&2; exit 1; }
bash -n "$SELF"
"$PYTHON" -m py_compile "$SCRIPT" lowlight_restoration/zimage_deterministic_bridge.py

if [[ "${1:-}" == --check ]]; then
  DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    "$PYTHON" -m pytest -q tests/test_zimage_deterministic_bridge.py
  "$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
  exit 0
fi

[[ "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]] || {
  echo 'ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' >&2; exit 1;
}
command -v flock >/dev/null
command -v nvidia-smi >/dev/null
mkdir -p "$ROOT/logs" "$TRITON_CACHE_DIR"

if [[ "${1:-}" != --run ]]; then
  exec 8>>"$ROOT/logs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1_launch.lock"
  flock -n 8 || { echo 'ERROR: concurrent Layer-1 launch' >&2; exit 1; }
  [[ ! -e "$LOG" && ! -e "$OUT" && ! -e "$STARTUP" ]] || {
    echo 'ERROR: prior Layer-1 evidence exists; inspect it instead of overwriting' >&2; exit 1;
  }
  TMUX=$(command -v tmux || true)
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" new-session -d -s "$SESSION" -c "$ROOT" 8>&-
    "$TMUX" set-option -t "$SESSION" remain-on-exit on
    "$TMUX" send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --run" C-m
    echo "Started tmux session: $SESSION"
    echo "Attach: $TMUX attach -t $SESSION"
  else
    nohup bash "$SELF" --run >"$STARTUP" 2>&1 </dev/null 8>&- &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    kill -0 "$pid" || { echo "ERROR: startup exited; inspect $STARTUP" >&2; exit 1; }
    echo "Started nohup PID: $pid"
  fi
  echo "Monitor: tail -f $LOG"
  exit 0
fi

exec 9>>"$ROOT/logs/foundir_blur_only_zimage_deterministic_bridge_tiny_v1.lock"
flock -n 9 || { echo 'ERROR: Layer-1 job already running' >&2; exit 1; }
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior Layer-1 evidence exists' >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "ZIMAGE_LAYER1_BRIDGE_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT

echo "ZIMAGE_LAYER1_BRIDGE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD) world=8 steps=1000"
echo 'SCOPE pure_Z-Image_Base6B Layer1 deterministic_LQ-latent_to_GT-latent_bridge multiscale_multidepth_LQ_condition latent_RGB_fidelity Blur_only no_independent_restoration_branch no_distillation no_decoder_tuning no_Qwen_degradation no_mixed no_sealed no_official_test'
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  "$SCRIPT" --config "$CONFIG"

"$PYTHON" - "$OUT/result.json" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "zimage_deterministic_bridge_tiny_complete"
assert p["classification"] == "pure_zimage_layer1_teacher_not_independent_restoration_branch"
assert p["steps"] == 1000 and p["visual_review_required"] is True
for key in ("independent_restoration_branch", "distillation", "decoder_finetuning",
            "qwen_degradation_conditioning", "mixed_degradations", "sealed_internal_accessed",
            "official_foundir_test_accessed", "development_gt_optimization"):
    assert p[key] is False
print("ZIMAGE_LAYER1_BRIDGE_RESULT_VERIFY_PASS", p["automatic_metric_gate_pass"])
PY
echo 'ZIMAGE_LAYER1_BRIDGE_COMPLETE_STOP_FOR_METRIC_AND_VISUAL_REVIEW'
