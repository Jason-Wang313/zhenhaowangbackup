#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_blur_only_flux2_sanity_v1.json"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_sanity_v1"
LOG="$ROOT/logs/foundir_blur_only_flux2_sanity_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_sanity_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_sanity_v1.lock"

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_blur_only_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_BLUR_ONLY_SANITY_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Blur-only sanity run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing workspace FLUX Python: $PYTHON" >&2; exit 1; }

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_only_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_BLUR_ONLY_SANITY_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE FLUX.2_4B Blur_only DiT_LoRA_only no_Qwen_degradation no_mixed no_pixel_refiner no_sealed no_official_test"
/usr/bin/python3 smoke/build_foundir_blur_only_sanity_manifest.py --config "$CONFIG"
"$PYTHON" smoke/train_foundir_blur_only_flux2_overfit_ddp.py --config "$CONFIG" --preflight

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

resume_args=()
if compgen -G "$OUTPUT/step_*.safetensors" >/dev/null && [[ ! -f "$OUTPUT/result.json" ]]; then
  resume_args+=(--resume)
  echo "FOUNDIR_BLUR_ONLY_SANITY_RESUME"
fi

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_only_flux2_overfit_ddp.py --config "$CONFIG" "${resume_args[@]}"
torchrun_status=$?
set -e

if [[ -f "$OUTPUT/result.json" ]]; then
  set +e
  "$PYTHON" - "$OUTPUT/result.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["protected_foundir_test_accessed"] is False
assert p["sealed_internal_accessed"] is False
if p["automatic_gate_pass"] is True and p["status"] == "awaiting_human_visual_review":
    raise SystemExit(0)
if p["automatic_gate_pass"] is False and p["status"] == "failed":
    raise SystemExit(2)
raise SystemExit(1)
PY
  normalized_status=$?
  set -e
else
  normalized_status=1
fi

if [[ "$normalized_status" -eq 0 ]]; then
  echo "FOUNDIR_BLUR_ONLY_SANITY_AUTOMATIC_PASS_STOP_FOR_VISUAL_REVIEW"
elif [[ "$normalized_status" -eq 2 ]]; then
  echo "FOUNDIR_BLUR_ONLY_SANITY_SCIENTIFIC_FAILURE_STOP_FOR_ADVISOR"
else
  echo "FOUNDIR_BLUR_ONLY_SANITY_RUNTIME_FAILURE torchrun_status=$torchrun_status" >&2
fi
exit "$normalized_status"
