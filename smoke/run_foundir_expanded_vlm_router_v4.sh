#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: <prepare|train|finish|features node0-3|render node0-3>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" =~ ^(features|render)$ && $# == 2 && "$NODE" =~ ^[0-3]$ || "$PHASE" =~ ^(prepare|train|finish)$ && $# == 1 ]] || exit 2
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified runtime missing'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
OUTPUT="$ROOT/checkpoints/foundir_expanded_vlm_router_v4/expanded_conditional_v4"
if [[ "$PHASE" != prepare ]]; then
  [[ -f "$OUTPUT/receipt.json" ]] || { echo 'ERROR: CPU prepare must finish first'; exit 2; }
fi
if [[ "$PHASE" == render || "$PHASE" == finish ]]; then
  [[ -f "$ROOT/reports/foundir_expanded_vlm_router_v4_training_2026-11-05.json" ]] || { echo 'ERROR: fixed CPU training/selection must finish first'; exit 2; }
fi
mkdir -p "$OUTPUT/attempts" logs
if [[ "$PHASE" == features || "$PHASE" == render ]]; then
  export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
  KEY="${PHASE}_node$NODE"; GPUS=2; CAP=5400
  [[ "$PHASE" == features ]] || CAP=2700
  exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
else
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  KEY="$PHASE"; GPUS=0; CAP=600
  [[ "$PHASE" != train ]] || CAP=3600
  exec 9>/tmp/zhenhaowang_frozen_image_router_cpu.lock
fi
flock -n 9 || { echo 'ERROR: local task active; do not kill it'; exit 3; }
exec 8>"$OUTPUT/$KEY.lock"
flock -n 8 || { echo 'ERROR: same stage/shard active elsewhere'; exit 3; }
exec > >(tee -a "logs/foundir_expanded_vlm_router_v4_$KEY.log") 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" "$KEY" "$CAP" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'attempts'; key=sys.argv[2]; cap=int(sys.argv[3])
starts=list(p.glob(key+'_*.started.json')); ends=list(p.glob(key+'_*.ended.json'))
if {f.name.removesuffix('.started.json') for f in starts}!={f.name.removesuffix('.ended.json') for f in ends}:
    raise SystemExit('ERROR: interrupted accounting; investigate before retry')
times=[json.loads(f.read_text())['seconds'] for f in ends]
if any(type(t) is not int or t<0 for t in times): raise SystemExit('ERROR: invalid accounting')
left=cap-sum(times)
if left<=0: raise SystemExit('ERROR: cumulative budget exhausted; no auto extension')
print(left)
PY
)
ATTEMPT="${KEY}_$(date -u +%Y%m%dT%H%M%S)_$$"; START=$SECONDS
printf '{"gpus":%s,"start_utc":"%s"}\n' "$GPUS" "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish_attempt() {
  status=$?; trap - EXIT; elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"gpus":%s,"used_gpu_seconds":%s}\n' "$status" "$elapsed" "$GPUS" "$((elapsed*GPUS))" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'EXPANDED_V4_END phase=%s node=%s status=%s seconds=%s gpus=%s\n' "$PHASE" "$NODE" "$status" "$elapsed" "$GPUS"
  exit "$status"
}
trap finish_attempt EXIT
step() {
  local left=$((REMAINING-SECONDS+START)); ((left>0)) || return 124
  timeout --signal=TERM --kill-after=30s "${left}s" "$@"
}
SCRIPT=smoke/run_foundir_expanded_vlm_router_v4.py
echo "EXPANDED_V4_START phase=$PHASE node=$NODE cumulative_remaining=$REMAINING gpus=$GPUS"
if [[ "$PHASE" == features || "$PHASE" == render ]]; then
  step "$PYTHON" -u "$SCRIPT" hardware --node "$NODE" --stage "$PHASE"
  step "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 "$SCRIPT" "$PHASE" --node "$NODE"
elif [[ "$PHASE" == train ]]; then
  step "$PYTHON" -u "$SCRIPT" train
else
  step "$PYTHON" -S "$SCRIPT" "$PHASE"
fi
# CPU agent coordinates next prespecified stage after all worker/exit receipts.
# No extra classification pass requirement; no budget increases or new recipe.
