#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.json"
SCRIPT="$ROOT/smoke/train_foundir_blur_only_flux2_pixel_fidelity_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_blur_only_flux2_trajectory_endpoint_gate_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_only_flux2_trajectory_endpoint_gate_v1.lock"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_blur_flux2_trajectory_endpoint_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another trajectory-endpoint run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing workspace FLUX Python: $PYTHON" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal trajectory-endpoint result already exists" >&2; exit 1; }

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_blur_flux2_trajectory_endpoint_v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD)"
echo "SCOPE FLUX.2_4B isolated_Blur step10k_parent exact_50step_trajectory prefix46_detached suffix4_differentiable frozen_VAE RGB1 LPIPS0.02 no_Qwen_degradation no_mixed no_refiner no_decoder_tuning no_distillation no_sealed no_official_test"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
resume_args=()
if [[ -d "$OUTPUT" ]] && find "$OUTPUT" -mindepth 1 -print -quit | grep -q .; then
  resume_args+=(--resume)
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_RESUME"
fi

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$SCRIPT" --config "$CONFIG" "${resume_args[@]}"
torchrun_status=$?
set -e

if [[ ! -f "$RESULT" ]]; then
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_RUNTIME_FAILURE torchrun_status=$torchrun_status" >&2
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
    --gate flux2_blur_trajectory_endpoint_runtime \
    --evidence reports/flux2_blur_trajectory_endpoint_authorization_v1.json \
    --config "$CONFIG" \
    --model "FLUX.2 Klein Base 4B exact 50-step trajectory-endpoint tiny gate" \
    --status "$torchrun_status" \
    --log "${LOG#$ROOT/}" \
    --diagnostic-only
  exit 1
fi

set +e
"$PYTHON" - "$RESULT" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p["world_size"] == 8 and p["global_batch_size"] == 8
assert set(p["arms"]) == {"flow_control", "trajectory_endpoint"}
assert p["development_used_for_selection"] is False
assert p["full_scale_automatic"] is False
for key in ("qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner",
            "decoder_tuning", "distillation", "development_gt_optimization", "sealed_internal_accessed",
            "official_foundir_test_accessed"):
    assert p[key] is False
if p["automatic_metric_gate_pass"] is True and p["status"] == "awaiting_human_visual_review":
    raise SystemExit(0)
if p["automatic_metric_gate_pass"] is False and p["status"] == "failed":
    raise SystemExit(2)
raise SystemExit(1)
PY
normalized_status=$?
set -e

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_blur_trajectory_endpoint_tiny_result \
  --evidence "$RESULT" \
  --config "$CONFIG" \
  --model "FLUX.2 Klein Base 4B isolated-Blur step-10000 parent paired flow/trajectory-endpoint tiny gate" \
  --split "16 frozen training crops plus 16 source-group-disjoint descriptive development crops; no protected split" \
  --outcome auto \
  --root-cause-or-status "Exact 50-step trajectory endpoint gate finished; inspect paired metrics, full-resolution rehearsal, and fixed Input|Output|GT visuals. No automatic full-scale action." \
  --advisor-decision-question "若metric与visual gate均通过，是否授权把同一trajectory-endpoint objective扩到完整isolated-Blur training？" \
  --hold-on-failure

if [[ "$normalized_status" -eq 0 ]]; then
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_METRIC_PASS_STOP_FOR_VISUAL_REVIEW"
elif [[ "$normalized_status" -eq 2 ]]; then
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_SCIENTIFIC_FAILURE_STOP"
else
  echo "FOUNDIR_BLUR_TRAJECTORY_ENDPOINT_RESULT_INVALID" >&2
fi
exit "$normalized_status"
