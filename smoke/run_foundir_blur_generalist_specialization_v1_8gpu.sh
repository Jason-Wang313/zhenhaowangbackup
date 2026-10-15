#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_blur_generalist_specialization_gate_v1.json"
LOG="$ROOT/logs/foundir_blur_generalist_specialization_v1.log"
GPU_LOG="$ROOT/logs/foundir_blur_generalist_specialization_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_blur_generalist_specialization_v1.lock"
cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  echo "FOUNDIR_BLUR_SPECIALIZATION_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another Blur specialization holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly eight GPUs" >&2; exit 1; }
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "FOUNDIR_BLUR_SPECIALIZATION_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_BLUR_SPECIALIZATION_SCOPE pretrained_backbone_train_Blur_validation_Blur_Noise"
echo "FOUNDIR_BLUR_SPECIALIZATION_PROTECTED_OFFICIAL_TEST not_accessed"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/train_foundir_blur_generalist_specialization_ddp.py --config "$CONFIG"
