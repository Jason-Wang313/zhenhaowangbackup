#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
export FOUNDIR_OFFICIAL_NNODES=4
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: <prepare|finish|render node0-3>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" == render && $# == 2 && "$NODE" =~ ^[0-3]$ || "$PHASE" =~ ^(prepare|finish)$ && $# == 1 ]] || exit 2
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified runtime missing'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
OUTPUT="$ROOT/checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2"
SCRIPT=smoke/official_vlm_v4_metric_identity_fix.py
if [[ "$PHASE" == render ]]; then
  export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
  KEY="render_node$NODE"; GPUS=2
  exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
else
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  KEY="$PHASE"; GPUS=0
  exec 9>/tmp/zhenhaowang_frozen_image_router_cpu.lock
fi
flock -n 9 || { echo 'ERROR: local task active; do not kill it'; exit 3; }
[[ -d "$OUTPUT/attempts" ]] || { echo 'ERROR: existing official run missing'; exit 2; }
exec 8>"$OUTPUT/$KEY.lock"
flock -n 8 || { echo 'ERROR: same stage/shard active elsewhere'; exit 3; }
mkdir -p logs
exec > >(tee -a "logs/foundir_official_vlm_router_v4_4x2_${KEY}_metricfix.log") 2>&1
if [[ "$PHASE" == prepare ]]; then
  "$PYTHON" -u "$SCRIPT" prepare
  exit 0
fi
if [[ "$PHASE" == render ]]; then
  "$PYTHON" -S "$SCRIPT" account --node "$NODE"
else
  "$PYTHON" -S "$SCRIPT" verify
  "$PYTHON" -S smoke/official_vlm_v4_unlimited_execution.py account --key finish
fi
ATTEMPT="${KEY}_metricfix_$(date -u +%Y%m%dT%H%M%S)_$$"; START=$SECONDS
printf '{"gpus":%s,"start_utc":"%s","time_limit_disabled":true,"metric_identity_fix":true}\n' "$GPUS" "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish_attempt() {
  status=$?; trap - EXIT; elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"gpus":%s,"used_gpu_seconds":%s,"time_limit_disabled":true,"metric_identity_fix":true}\n' "$status" "$elapsed" "$GPUS" "$((elapsed*GPUS))" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'OFFICIAL_VLM_V4_END phase=%s node=%s status=%s seconds=%s gpus=%s time_limit_disabled=true metric_identity_fix=true\n' "$PHASE" "$NODE" "$status" "$elapsed" "$GPUS"
  exit "$status"
}
trap finish_attempt EXIT
echo "OFFICIAL_VLM_V4_START phase=$PHASE node=$NODE time_limit_disabled=true gpus=$GPUS metric_identity_fix=true"
if [[ "$PHASE" == render ]]; then
  "$PYTHON" -u "$SCRIPT" hardware --node "$NODE"
  "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 "$SCRIPT" render --node "$NODE"
else
  "$PYTHON" -S "$SCRIPT" finish
fi
