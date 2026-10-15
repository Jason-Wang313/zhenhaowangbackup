#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v3.json"
MANIFEST="$ROOT/manifests/foundir_five_type_pairs_v1.csv"
MANIFEST_JSON="$ROOT/manifests/foundir_five_type_pairs_v1.json"
MANIFEST_CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v1.json"
FLUX_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
OUTPUT="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3"
MODEL="VLM-guided FLUX.2 Klein 4B; frozen physical-v8 Qwen; DiT LoRA + pixel refiner"
LOG="$ROOT/logs/foundir_five_type_flux2_dit_v3.log"
GPU_LOG="$ROOT/logs/foundir_five_type_flux2_dit_v3_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_flux2_dit_v3.lock"

cd "$ROOT"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
mkdir -p "$ROOT/logs" "$OUTPUT" "$ROOT/.cache/triton/foundir_five_type_v3" \
  "$ROOT/.cache/torch_extensions/foundir_five_type_v3"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  if [[ "$status" -ne 0 ]]; then
    /usr/bin/python3 "$ROOT/smoke/advisor_governance.py" runtime-failure \
      --gate stage0_v3_runtime_failure --evidence "$CONFIG" --config "$CONFIG" \
      --model "$MODEL" --status "$status" --log "$LOG" || true
  fi
  echo "FOUNDIR_FIVE_TYPE_STAGE0_V3_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another five-type v3 run holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$FLUX_PYTHON" && -x "$QWEN_PYTHON" ]] || {
  echo "ERROR: workspace Python environments are missing" >&2; exit 1;
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type_v3"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type_v3"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_FIVE_TYPE_STAGE0_V3_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FIVE_TYPE_STAGE0_V3_SCOPE blur,noise,rain,haze,low_light no_jpeg no_condition_ablation"
echo "FOUNDIR_FIVE_TYPE_STAGE0_V3_PROTECTED_TEST not_accessed"

"$FLUX_PYTHON" smoke/foundir_five_type_runtime_preflight.py \
  --config "$CONFIG" --require-cuda --full-deserialization
"$QWEN_PYTHON" smoke/build_foundir_five_type_manifest.py \
  --config "$MANIFEST_CONFIG" --output-csv "$MANIFEST" \
  --output-json "$MANIFEST_JSON" --verify-existing
"$QWEN_PYTHON" smoke/build_foundir_five_type_physical_v8_conditions.py \
  --config "$CONFIG"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
sleep 1
kill -0 "$telemetry_pid" 2>/dev/null || {
  echo "ERROR: nvidia-smi telemetry failed to stay active" >&2; exit 1;
}

OVERFIT_RESULT="$OUTPUT/native_reference_overfit_sow/result.json"
HISTORICAL_CHECKPOINT="$OUTPUT/native_reference_overfit/step_000200.safetensors"
if [[ ! -f "$OVERFIT_RESULT" ]]; then
  set +e
  if [[ -f "$HISTORICAL_CHECKPOINT" ]]; then
    "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/recheck_foundir_five_type_flux2_overfit_sow_ddp.py --config "$CONFIG"
  else
    "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
      smoke/train_foundir_five_type_flux2_overfit_ddp.py --config "$CONFIG"
  fi
  torchrun_status=$?
  set -e
else
  torchrun_status=0
fi
if [[ -f "$OVERFIT_RESULT" ]]; then
  set +e
  "$FLUX_PYTHON" - "$CONFIG" "$OVERFIT_RESULT" <<'PY'
import hashlib,json,pathlib,sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
p=json.loads(pathlib.Path(sys.argv[2]).read_text())
assert p["config_sha256"] == sha(sys.argv[1])
assert p["protected_foundir_test_accessed"] is False
assert set(p["arm_metrics"]) == {"qwen"}
print("FOUNDIR_FIVE_TYPE_V3_OVERFIT_VERIFY_PASS", p["status"])
raise SystemExit(0 if p["automatic_gate_pass"] else 2)
PY
  gate_status=$?
  set -e
  /usr/bin/python3 "$ROOT/smoke/advisor_governance.py" emit \
    --gate stage0_overfit --evidence "$OVERFIT_RESULT" --config "$CONFIG" \
    --model "$MODEL" --split foundir_training_only_overfit \
    --outcome auto --hold-on-failure \
    --root-cause-or-status "Stage-0 overfit learnability gate completed." \
    --advisor-decision-question "若失败，请判断是否修复同一 FLUX.2 mainline；若通过，记录后继续 full training。"
  exit "$gate_status"
fi
exit "$torchrun_status"
