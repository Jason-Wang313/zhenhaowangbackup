#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
TRAIN_CONFIG="$ROOT/configs/foundir_blur_only_flux2_continuation_v1.json"
EVAL_CONFIG="$ROOT/configs/foundir_blur_only_flux2_continuation_development_step14_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_continuation_v1"
SUMMARY="$OUTPUT/training_summary.json"
RESULT="$OUTPUT/full_development_step_014000/result.json"
LOG="$ROOT/logs/foundir_blur_only_flux2_continuation_step14_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_continuation_step14_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_continuation_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
"$PYTHON" smoke/check_foundir_blur_only_flux2_continuation_step14_gate.py
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_blur_flux2_continuation_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another continuation run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_flux2_continuation_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

if [[ ! -f "$SUMMARY" ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/train_foundir_blur_only_flux2_continuation_ddp.py --config "$TRAIN_CONFIG" --stop-step 14000
fi

"$PYTHON" - "$TRAIN_CONFIG" "$SUMMARY" <<'PY'
import hashlib, json, pathlib, sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
config, summary = map(pathlib.Path, sys.argv[1:])
p=json.loads(summary.read_text())
assert p["status"] == "training_complete_awaiting_human_visual_review"
assert p["optimizer_steps"] == 14000 and p["global_batch_size"] == 32
assert p["config_sha256"] == sha(config)
assert p["qwen_degradation_conditioning"] is False and p["mixed_degradations"] is False
assert p["pixel_refiner"] is False and p["sealed_internal_accessed"] is False
assert p["protected_foundir_test_accessed"] is False
checkpoint=pathlib.Path.cwd()/p["final_checkpoint"]
assert checkpoint.name == "step_014000.safetensors" and sha(checkpoint) == p["final_checkpoint_sha256"]
print("FOUNDIR_BLUR_FLUX2_CONTINUATION_STAGE14_VERIFY_PASS")
PY

if [[ ! -f "$RESULT" ]]; then
  "$PYTHON" smoke/evaluate_foundir_blur_only_flux2_continuation_ddp.py --config "$EVAL_CONFIG" --preflight
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_blur_only_flux2_continuation_ddp.py --config "$EVAL_CONFIG"
fi

"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["status"] == "blur_only_full_development_evaluation_complete"
assert p["records"] == 2595 and p["source_groups"] == 165
assert p["checkpoint_curve"][0]["checkpoint_step"] == 14000
assert p["sealed_internal_accessed"] is False and p["protected_foundir_test_accessed"] is False
print("FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_FULL_DEVELOPMENT_VERIFY_PASS")
PY

echo "FOUNDIR_BLUR_FLUX2_CONTINUATION_STEP14_COMPLETE result=$RESULT"
