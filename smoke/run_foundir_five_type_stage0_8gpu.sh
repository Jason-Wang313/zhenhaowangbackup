#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v1.json"
QWEN_CONFIG="$ROOT/configs/qwen_multidegradation_100k_rank64_joint_v1.json"
PRELAUNCH="$ROOT/configs/foundir_five_type_prelaunch_gate_v1.json"
QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
FLUX_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
MANIFEST="$ROOT/manifests/foundir_five_type_pairs_v1.csv"
MANIFEST_JSON="$ROOT/manifests/foundir_five_type_pairs_v1.json"
CONDITIONS="$ROOT/manifests/foundir_five_type_overfit_qwen_conditions_v1.csv"
CONDITIONS_JSON="$ROOT/manifests/foundir_five_type_overfit_qwen_conditions_v1.json"
OUTPUT="$ROOT/checkpoints/foundir_five_type_flux2_dit_v1"
LOG="$ROOT/logs/foundir_five_type_flux2_dit_v1.log"
GPU_LOG="$ROOT/logs/foundir_five_type_flux2_dit_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_flux2_dit_v1.lock"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs" "$ROOT/.cache/triton/foundir_five_type" \
  "$ROOT/.cache/torch_extensions/foundir_five_type"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_FIVE_TYPE_STAGE0_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: another five-type stage-0 run holds $LOCK" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
if [[ ! -x "$QWEN_PYTHON" || ! -x "$FLUX_PYTHON" ]]; then
  echo "ERROR: workspace Qwen or FLUX Python environment is missing" >&2
  exit 1
fi
for executable in flock nvidia-smi; do
  command -v "$executable" >/dev/null 2>&1 || {
    echo "ERROR: required runtime executable is missing: $executable" >&2
    exit 1
  }
done

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions/foundir_five_type"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "FOUNDIR_FIVE_TYPE_STAGE0_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FIVE_TYPE_STAGE0_HARDWARE exactly_8x_H100_80GB_or_H200"
echo "FOUNDIR_FIVE_TYPE_STAGE0_PROTECTED_TEST not_accessed"

"$FLUX_PYTHON" smoke/foundir_five_type_runtime_preflight.py \
  --config "$CONFIG" --require-cuda --full-deserialization
"$QWEN_PYTHON" smoke/evaluate_foundir_five_type_qwen_training_gate.py \
  --prelaunch-config "$PRELAUNCH" --require-selected joint
"$QWEN_PYTHON" smoke/foundir_five_type_preflight.py --config "$CONFIG" --require-data
if [[ -f "$MANIFEST" || -f "$MANIFEST_JSON" ]]; then
  "$QWEN_PYTHON" smoke/build_foundir_five_type_manifest.py \
    --config "$CONFIG" --output-csv "$MANIFEST" --output-json "$MANIFEST_JSON" --verify-existing
else
  "$QWEN_PYTHON" smoke/build_foundir_five_type_manifest.py \
    --config "$CONFIG" --output-csv "$MANIFEST" --output-json "$MANIFEST_JSON" --workers 64
fi

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
sleep 1
if ! kill -0 "$telemetry_pid" 2>/dev/null; then
  echo "ERROR: nvidia-smi telemetry failed to stay active" >&2
  exit 1
fi

condition_args=(
  --config "$CONFIG" --qwen-config "$QWEN_CONFIG" --manifest "$MANIFEST"
  --output-csv "$CONDITIONS" --output-json "$CONDITIONS_JSON" --records 128
)
if [[ ! -f "$CONDITIONS" || ! -f "$CONDITIONS_JSON" ]]; then
  echo "ERROR: corrected joint-Qwen pilot is missing; run the prelaunch gate first" >&2
  exit 1
fi
"$QWEN_PYTHON" smoke/export_foundir_five_type_qwen_conditions.py \
  "${condition_args[@]}" --verify-existing

VAE_RESULT="$OUTPUT/vae_ceiling_result.json"
if [[ -f "$VAE_RESULT" ]]; then
  "$FLUX_PYTHON" - "$CONFIG" "$MANIFEST" "$VAE_RESULT" <<'PY'
import hashlib, json, pathlib, sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload=json.loads(pathlib.Path(sys.argv[3]).read_text())
assert payload["config_sha256"] == sha(sys.argv[1])
assert payload["manifest_sha256"] == sha(sys.argv[2])
assert payload["protected_foundir_test_accessed"] is False
print("FOUNDIR_FIVE_TYPE_VAE_RESULT_VERIFY_PASS")
PY
else
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/evaluate_foundir_five_type_vae_ceiling_ddp.py --config "$CONFIG" --manifest "$MANIFEST"
fi

OVERFIT_RESULT="$OUTPUT/native_reference_overfit/result.json"
if [[ ! -f "$OVERFIT_RESULT" ]]; then
  set +e
  "$FLUX_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/train_foundir_five_type_flux2_overfit_ddp.py --config "$CONFIG"
  torchrun_status=$?
  set -e
else
  torchrun_status=0
fi

if [[ -f "$OVERFIT_RESULT" ]]; then
  set +e
  "$FLUX_PYTHON" - "$CONFIG" "$CONDITIONS" "$OVERFIT_RESULT" <<'PY'
import hashlib, json, pathlib, sys
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
payload=json.loads(pathlib.Path(sys.argv[3]).read_text())
assert payload["config_sha256"] == sha(sys.argv[1])
assert payload["conditions_sha256"] == sha(sys.argv[2])
assert payload["protected_foundir_test_accessed"] is False
print("FOUNDIR_FIVE_TYPE_OVERFIT_RESULT_VERIFY_PASS", payload["status"])
raise SystemExit(0 if payload["automatic_gate_pass"] else 2)
PY
  gate_status=$?
  set -e
  exit "$gate_status"
fi
exit "$torchrun_status"
