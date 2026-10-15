#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_haze_flux2_source_diverse_pilot_v1.json"
TRAINER="$ROOT/smoke/train_foundir_blur_only_flux2_overfit_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_haze_flux2_source_diverse_pilot_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_haze_flux2_source_diverse_pilot_v1.log"
GPU_LOG="$ROOT/logs/foundir_haze_flux2_source_diverse_pilot_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_haze_flux2_source_diverse_pilot_v1.lock"
MODEL="FLUX.2 Klein Base 4B fresh rank32 LoRA isolated-Haze source-diverse pilot"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_haze_source_diverse_v1"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_HAZE_SOURCE_DIVERSE_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Haze source-diverse pilot holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing workspace FLUX Python: $PYTHON" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal pilot result already exists" >&2; exit 1; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models are forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
done

export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_haze_source_diverse_v1"

echo "FOUNDIR_HAZE_SOURCE_DIVERSE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD) gpu=$first_name"
echo "SCOPE FLUX.2_4B isolated_Haze 1580_sources fresh_LoRA 2000_steps global_batch32 frozen_128_validation no_Qwen_degradation no_mixed no_pixel_refiner no_sealed no_official_test no_automatic_full_development_launch"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$TRAINER" --config "$CONFIG" --resume
trainer_status=$?
set -e
if [[ "$trainer_status" -ne 0 && "$trainer_status" -ne 2 ]]; then
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure \
    --gate haze_source_diverse_pilot_runtime --evidence "$CONFIG" \
    --config "$CONFIG" --model "$MODEL" --status "$trainer_status" --log "$LOG" || true
  exit "$trainer_status"
fi
[[ -f "$RESULT" ]] || { echo "ERROR: trainer ended without terminal result" >&2; exit 1; }

"$PYTHON" - "$CONFIG" "$RESULT" <<'PY'
import hashlib,json,pathlib,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
config_path,result_path=map(pathlib.Path,sys.argv[1:])
p=json.loads(result_path.read_text())
assert p["config_sha256"] == sha(config_path)
assert p["task"] == "08Haze"
assert p["validation_metrics_are_formal_gate"] is True
assert p["automatic_full_development_launch"] is False
assert p["descriptive_validation_metrics"]["records"] == 128
assert p["qwen_degradation_conditioning"] is False and p["mixed_degradations"] is False
assert p["pixel_refiner"] is False and p["protected_foundir_test_accessed"] is False
checkpoint=pathlib.Path.cwd()/p["checkpoint"]
assert sha(checkpoint) == p["checkpoint_sha256"]
print("FOUNDIR_HAZE_SOURCE_DIVERSE_RESULT_VERIFY_PASS")
PY

/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_haze_source_diverse_pilot_result --evidence "$RESULT" --config "$CONFIG" \
  --model "$MODEL" --split "1580 isolated-Haze training source groups and frozen 128 source-group-disjoint validation records" \
  --outcome auto --require-decision \
  --root-cause-or-status "The bounded Haze source-diverse pilot completed. Mean validation PSNR/LPIPS and fixed visuals decide whether the frozen checkpoint may proceed to the 1,580-record full-development validation; there is no automatic continuation." \
  --advisor-decision-question "请检查 Haze 128-record PSNR/LPIPS 与固定视觉；是否允许保持当前 1,580-source training scale 并进行一次完整 1,580-record development validation？"

if [[ "$trainer_status" -eq 0 ]]; then
  echo "FOUNDIR_HAZE_SOURCE_DIVERSE_METRIC_PASS_STOP_FOR_VISUAL_REVIEW"
else
  echo "FOUNDIR_HAZE_SOURCE_DIVERSE_SCIENTIFIC_FAILURE_STOP"
fi
exit "$trainer_status"
