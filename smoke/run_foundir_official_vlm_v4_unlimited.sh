#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
export FOUNDIR_OFFICIAL_NNODES=4
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: <authorize|finish|render node0-3>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" == render && $# == 2 && "$NODE" =~ ^[0-3]$ || "$PHASE" =~ ^(authorize|finish)$ && $# == 1 ]] || exit 2
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified runtime missing'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
OUTPUT="$ROOT/checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1_4x2"
[[ -f "$OUTPUT/render_plan.json" ]] || { echo 'ERROR: existing official plan missing; do not regenerate'; exit 2; }
HELPER=smoke/official_vlm_v4_unlimited_execution.py
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
exec 8>"$OUTPUT/$KEY.lock"
flock -n 8 || { echo 'ERROR: same stage/shard active elsewhere'; exit 3; }
mkdir -p "$OUTPUT/attempts" logs
exec > >(tee -a "logs/foundir_official_vlm_router_v4_4x2_$KEY.log") 2>&1
if [[ "$PHASE" == authorize ]]; then
  "$PYTHON" -S "$HELPER" authorize
  exit 0
fi
"$PYTHON" -S "$HELPER" account --key "$KEY"
ATTEMPT="${KEY}_$(date -u +%Y%m%dT%H%M%S)_$$"; START=$SECONDS
printf '{"gpus":%s,"start_utc":"%s","time_limit_disabled":true}\n' "$GPUS" "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish_attempt() {
  status=$?; trap - EXIT; elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"gpus":%s,"used_gpu_seconds":%s,"time_limit_disabled":true}\n' "$status" "$elapsed" "$GPUS" "$((elapsed*GPUS))" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'OFFICIAL_VLM_V4_END phase=%s node=%s status=%s seconds=%s gpus=%s time_limit_disabled=true\n' "$PHASE" "$NODE" "$status" "$elapsed" "$GPUS"
  exit "$status"
}
trap finish_attempt EXIT
SCRIPT=smoke/evaluate_foundir_official_vlm_router_v4.py
echo "OFFICIAL_VLM_V4_START phase=$PHASE node=$NODE time_limit_disabled=true gpus=$GPUS"
# No timeout command and no fabricated large finite cap. Scientific producer unchanged.
if [[ "$PHASE" == render ]]; then
  "$PYTHON" -u "$SCRIPT" hardware --node "$NODE"
  "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 "$SCRIPT" render --node "$NODE"
else
  "$PYTHON" -S "$SCRIPT" finish
fi
