#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_rain_flux2_strength_identity_ratio_gate_v1.json"
TRAINER="$ROOT/smoke/train_foundir_blur_only_flux2_overfit_ddp.py"
OUTPUT="$ROOT/checkpoints/foundir_rain_flux2_strength_identity_ratio_gate_v1"
RESULT="$OUTPUT/result.json"
LOG="$ROOT/logs/foundir_rain_flux2_strength_identity_ratio_gate_v1.log"
GPU_LOG="$ROOT/logs/foundir_rain_flux2_strength_identity_ratio_gate_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_rain_flux2_strength_identity_ratio_gate_v1.lock"
MODEL="FLUX.2 Klein Base 4B exact Rain step-2000 rank32 LoRA 3:2:1 ratio gate"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_rain_strength_identity_ratio_v1"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  echo "FOUNDIR_RAIN_RATIO_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Rain ratio gate holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal ratio-gate result already exists" >&2; exit 1; }
[[ -z "$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit)" ]] || { echo "ERROR: ratio-gate output directory is nonempty" >&2; exit 1; }
mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models are forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
done
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TRACE_BUFFER_SIZE=1048576 TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DESYNC_DEBUG=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_rain_strength_identity_ratio_v1"
echo "FOUNDIR_RAIN_RATIO_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) git=$(git rev-parse HEAD) gpu=$first_name"
echo "SCOPE exact_step2000_parent exact_optimizer LR_2.5e-5 250_steps ratio_3_2_1 near_clean_LQ_identity heavy_moderate_GT same_balanced_48_development no_new_module no_new_loss no_alpha no_refiner no_routing no_sealed no_official_test no_automatic_continuation"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
set +e
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" --config "$CONFIG"
trainer_status=$?
set -e
if [[ "$trainer_status" -ne 0 ]]; then
  /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate rain_strength_identity_ratio_runtime --evidence "$CONFIG" --config "$CONFIG" --model "$MODEL" --status "$trainer_status" --log "$LOG" || true
  exit "$trainer_status"
fi
[[ -f "$RESULT" ]] || { echo "ERROR: trainer ended without terminal result" >&2; exit 1; }
"$PYTHON" - "$CONFIG" "$RESULT" <<'PY'
import hashlib,json,pathlib,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
config_path,result_path=map(pathlib.Path,sys.argv[1:]); p=json.loads(result_path.read_text())
assert p["config_sha256"] == sha(config_path) and p["task"] == "10Rain"
assert p["directional_gate_only"] is True and p["automatic_continuation"] is False
assert p["parent_development_metrics"]["records"] == p["descriptive_validation_metrics"]["records"] == 48
assert all(p["parent_strength_metrics"][name]["records"] == 16 for name in ("heavy","moderate","near_clean"))
assert all(p["candidate_strength_metrics"][name]["records"] == 16 for name in ("heavy","moderate","near_clean"))
checkpoint=pathlib.Path.cwd()/p["checkpoint"]; assert sha(checkpoint) == p["checkpoint_sha256"]
print("FOUNDIR_RAIN_RATIO_RESULT_VERIFY_PASS")
PY
/usr/bin/python3 smoke/advisor_governance.py emit \
  --gate flux2_rain_strength_identity_ratio_result --evidence "$RESULT" --config "$CONFIG" --model "$MODEL" \
  --split "exact 1580 Rain parent training crops; frozen source-disjoint train-split development with 16 heavy, 16 moderate, 16 near-clean" \
  --outcome auto --require-decision \
  --root-cause-or-status "The one-change 3:2:1 Rain ratio gate completed; it decides whether reduced identity exposure retains near-clean gains while recovering heavy PSNR and LPIPS. There is no automatic continuation." \
  --advisor-decision-question "请检查五项 directional checks 与固定视觉；是否允许正式 Rain validation，或停止并重新诊断？"
if [[ "$(jq -r .automatic_gate_pass "$RESULT")" == "true" ]]; then echo "FOUNDIR_RAIN_RATIO_METRIC_PASS_STOP_FOR_VISUAL_REVIEW"; else echo "FOUNDIR_RAIN_RATIO_SCIENTIFIC_FAILURE_STOP"; fi
