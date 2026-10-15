#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
TRAIN_PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
METRIC_PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
CONFIG="$ROOT/configs/foundir_triple_onerestore_r0_v1.json"
TRAINER="$ROOT/smoke/train_foundir_triple_onerestore_r0_ddp.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_triple_onerestore_r0_pyiqa_ddp.py"
RENDERER="$ROOT/smoke/render_foundir_triple_onerestore_r0_v1.py"
OUTPUT="$ROOT/checkpoints/foundir_triple_onerestore_r0_v1"
RESULT="$OUTPUT/result.json"
REPORT="$ROOT/reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md"
LOG="$ROOT/logs/foundir_triple_onerestore_r0_v1.log"
GPU_LOG="$ROOT/logs/foundir_triple_onerestore_r0_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_triple_onerestore_r0_v1.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_TRIPLE_ONERESTORE_R0_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return "$status"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: FoundIR triple OneRestore R0 already running" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set exactly CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 1
}
[[ ! -e "$RESULT" ]] || {
  echo "ERROR: terminal R0 result already exists; frozen R0 must not be rerun" >&2
  exit 1
}
mapfile -t names < <(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
[[ "${#names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs required" >&2; exit 1; }
profile=
for name in "${names[@]}"; do
  current=
  [[ "$name" == *H100* ]] && current=H100
  [[ "$name" == *H200* ]] && current=H200
  [[ -n "$current" ]] || { echo "ERROR: unsupported GPU $name" >&2; exit 1; }
  [[ -z "$profile" || "$profile" == "$current" ]] || {
    echo "ERROR: mixed H100/H200 node rejected" >&2
    exit 1
  }
  profile=$current
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_P2P_DISABLE=0 NCCL_NVLS_ENABLE=0

echo "FOUNDIR_TRIPLE_ONERESTORE_R0_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT profile=$profile"
echo "IDENTITY config=$CONFIG config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}') git_revision=$(git rev-parse HEAD) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SCOPE synthetic_R0_only A_unconditioned B_fixed_categorical C_fixed_categorical_cdrl final_step_only heldout_after_all_arms_frozen no_official_test no_FoundIR60"
echo "TRANSPORT NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"
echo "BUDGET H200_wall_hours=0.75-2.0_center=1.25 GPU_hours=6-16_center=10; H100_wall_hours=1.0-2.5_center=1.5 GPU_hours=8-20_center=12"
echo "UNCERTAINTY no external-node benchmark; dominant uncertainty is CDRL VGG16 backward plus DDP/shared-storage throughput"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!

"$TRAIN_PYTHON" "$ROOT/smoke/advisor_governance.py" check
"$TRAIN_PYTHON" "$RENDERER" --check
"$TRAIN_PYTHON" "$TRAINER" --config "$CONFIG" --preflight
"$METRIC_PYTHON" "$EVALUATOR" --config "$CONFIG" --preflight
"$TRAIN_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" --config "$CONFIG"
"$METRIC_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG"

[[ -s "$RESULT" && -s "$REPORT" ]] || { echo "ERROR: terminal result/report missing" >&2; exit 1; }
"$TRAIN_PYTHON" - <<'PY'
import json
from pathlib import Path
path = Path("checkpoints/foundir_triple_onerestore_r0_v1/result.json")
result = json.loads(path.read_text(encoding="utf-8"))
assert result["status"] == "foundir_triple_onerestore_r0_v1_terminal"
assert result["official_foundir_test_accessed"] is False
assert result["foundir_60_image_triple_accessed"] is False
assert result["heldout_used_for_checkpoint_selection"] is False
print("FOUNDIR_TRIPLE_ONERESTORE_R0_V1_TERMINAL_VERIFIED stop_after_r0=true")
PY
