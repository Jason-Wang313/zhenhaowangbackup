#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: destination only'; exit 2; }
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: runner <features node_rank_0_to_3 | train>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" == features && $# == 2 && "$NODE" =~ ^[0-3]$ || "$PHASE" == train && $# == 1 ]] || exit 2
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified Python missing'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
OUTPUT="$ROOT/checkpoints/foundir_frozen_image_router_v2/expanded_frozen_image_v2"
[[ -f "$OUTPUT/receipt.json" ]] || { echo 'ERROR: CPU prepare must pass first'; exit 2; }
mkdir -p "$OUTPUT/attempts" logs
if [[ "$PHASE" == features ]]; then
  export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
  KEY="features_node$NODE"; GPUS=2
  exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
else
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  KEY=cpu_train; GPUS=0
  exec 9>/tmp/zhenhaowang_frozen_image_router_cpu.lock
fi
flock -n 9 || { echo 'ERROR: local task active; do not kill it'; exit 3; }
exec 8>"$OUTPUT/$KEY.lock"
flock -n 8 || { echo 'ERROR: same shard/phase active on another node'; exit 3; }
exec > >(tee -a "logs/foundir_frozen_image_router_v2_$KEY.log") 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" "$KEY" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'attempts'; key=sys.argv[2]
starts=list(p.glob(key+'_*.started.json')); ends=list(p.glob(key+'_*.ended.json'))
if {f.name.removesuffix('.started.json') for f in starts}!={f.name.removesuffix('.ended.json') for f in ends}:
    raise SystemExit('ERROR: interrupted accounting; investigate before retry')
times=[json.loads(f.read_text())['seconds'] for f in ends]
if any(type(t) is not int or t<0 for t in times): raise SystemExit('ERROR: invalid accounting')
left=3600-sum(times)
if left<=0: raise SystemExit('ERROR: cumulative budget exhausted; no auto extension')
print(left)
PY
)
ATTEMPT="${KEY}_$(date -u +%Y%m%dT%H%M%S)_$$"; START=$SECONDS
printf '{"gpus":%s,"start_utc":"%s"}\n' "$GPUS" "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish() {
  status=$?; trap - EXIT; elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"gpus":%s,"used_gpu_seconds":%s}\n' "$status" "$elapsed" "$GPUS" "$((elapsed*GPUS))" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'FROZEN_IMAGE_V2_END phase=%s node=%s status=%s seconds=%s gpus=%s\n' "$PHASE" "$NODE" "$status" "$elapsed" "$GPUS"
  exit "$status"
}
trap finish EXIT
step() {
  local left=$((REMAINING-SECONDS+START)); ((left>0)) || return 124
  timeout --signal=TERM --kill-after=30s "${left}s" "$@"
}
SCRIPT=smoke/repair_foundir_frozen_image_router_v2.py
echo "FROZEN_IMAGE_V2_START phase=$PHASE node=$NODE cumulative_remaining=$REMAINING"
if [[ "$PHASE" == features ]]; then
  step "$PYTHON" -u "$SCRIPT" hardware --node "$NODE"
  # Independent feature workers, no rendezvous/NCCL and no old master IP.
  step "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 "$SCRIPT" extract --node "$NODE"
else
  step "$PYTHON" -u "$SCRIPT" train
  step "$PYTHON" -S "$SCRIPT" verify
fi
# Feature completion does NOT automatically start CPU training or restoration.
