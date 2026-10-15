#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_five_type_qwen_routed_integrated_v2.json"
LOG="$ROOT/logs/foundir_five_type_qwen_routed_integrated_v2_internal.log"
GPU_LOG="$ROOT/logs/foundir_five_type_qwen_routed_integrated_v2_internal_gpu.csv"
LOCK="$ROOT/logs/foundir_five_type_qwen_routed_integrated_v2_internal.lock"
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  echo "FOUNDIR_QWEN_ROUTED_INTEGRATED_V2_INTERNAL_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another integrated v2 internal evaluation holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly eight GPUs" >&2; exit 1; }
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FOUNDIR_EVALUATION_COORDINATION_TOKEN="qwen-routed-integrated-v2-internal"
echo "FOUNDIR_QWEN_ROUTED_INTEGRATED_V2_INTERNAL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_QWEN_ROUTED_INTEGRATED_INTERNAL_REUSE_DISCLOSURE previously_consumed_internal_records"
echo "FOUNDIR_QWEN_ROUTED_INTEGRATED_PROTECTED_OFFICIAL_TEST not_accessed"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/evaluate_foundir_five_type_qwen_routed_integrated_ddp.py \
  --router-config "$CONFIG" --split internal_test
