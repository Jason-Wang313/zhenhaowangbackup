#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || exit 2
export FOUNDIR_OFFICIAL_NNODES=${FOUNDIR_OFFICIAL_NNODES:-4}
[[ "$FOUNDIR_OFFICIAL_NNODES" == 2 || "$FOUNDIR_OFFICIAL_NNODES" == 4 ]] || { echo "ERROR: FOUNDIR_OFFICIAL_NNODES must be 2 or 4"; exit 2; }
TOTAL_GPUS=$((FOUNDIR_OFFICIAL_NNODES*2))
[[ $# == 1 || $# == 2 ]] || { echo 'Usage: <prepare|plan|finish|route node0..N-1|render node0..N-1>'; exit 2; }
PHASE=$1 NODE=${2:-0}
[[ "$PHASE" =~ ^(route|render)$ && $# == 2 && "$NODE" =~ ^[0-3]$ && "$NODE" -lt "$FOUNDIR_OFFICIAL_NNODES" || "$PHASE" =~ ^(prepare|plan|finish)$ && $# == 1 ]] || exit 2
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified runtime missing'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_HOME="$ROOT/models/torch"
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
RUN_SUFFIX=""
[[ "$FOUNDIR_OFFICIAL_NNODES" != 4 ]] || RUN_SUFFIX="_4x2"
OUTPUT="$ROOT/checkpoints/foundir_official_vlm_router_v4/official_vlm_v4_frozen4800_v1$RUN_SUFFIX"
if [[ "$PHASE" != prepare ]]; then [[ -f "$OUTPUT/receipt.json" ]] || { echo 'ERROR: CPU prepare required'; exit 2; }; fi
mkdir -p "$OUTPUT/attempts" logs
if [[ "$PHASE" == route || "$PHASE" == render ]]; then
  export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
  KEY="${PHASE}_node$NODE"; GPUS=2; CAP=$((12*3600/TOTAL_GPUS))
  if [[ "$PHASE" == render ]]; then
    [[ -f "$OUTPUT/render_budget.json" ]] || { echo 'ERROR: full-resolution cost requires explicit user approval; run plan first'; exit 2; }
    CAP=$("$PYTHON" -S - "$OUTPUT" "$TOTAL_GPUS" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); b=json.loads((p/'render_budget.json').read_text())
assert b['render_plan_sha256']==hashlib.sha256((p/'render_plan.json').read_bytes()).hexdigest()
assert b['authority']=='explicit_user_approval_recorded_by_CPU_agent' and len(b['approval_note'].strip())>=12
assert b['world']==int(sys.argv[2])
assert 0<b['approved_render_GPU_hours']<=400 and b['cap_seconds_per_node']==int(b['approved_render_GPU_hours']*3600/int(sys.argv[2]))
print(b['cap_seconds_per_node'])
PY
)
  fi
  exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
else
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
  KEY="$PHASE"; GPUS=0; CAP=1800
  [[ "$PHASE" != finish ]] || CAP=900
  exec 9>/tmp/zhenhaowang_frozen_image_router_cpu.lock
fi
flock -n 9 || { echo 'ERROR: local task active; do not kill it'; exit 3; }
exec 8>"$OUTPUT/$KEY.lock"
flock -n 8 || { echo 'ERROR: same stage/shard active elsewhere'; exit 3; }
exec > >(tee -a "logs/foundir_official_vlm_router_v4${RUN_SUFFIX}_$KEY.log") 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" "$KEY" "$CAP" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'attempts'; key=sys.argv[2]; cap=int(sys.argv[3])
starts=list(p.glob(key+'_*.started.json')); ends=list(p.glob(key+'_*.ended.json'))
if {f.name.removesuffix('.started.json') for f in starts}!={f.name.removesuffix('.ended.json') for f in ends}:
    raise SystemExit('ERROR: interrupted accounting; reconcile elapsed cost before retry')
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
  printf 'OFFICIAL_VLM_V4_END phase=%s node=%s status=%s seconds=%s gpus=%s\n' "$PHASE" "$NODE" "$status" "$elapsed" "$GPUS"
  exit "$status"
}
trap finish_attempt EXIT
step() {
  local left=$((REMAINING-SECONDS+START)); ((left>0)) || return 124
  timeout --signal=TERM --kill-after=30s "${left}s" "$@"
}
SCRIPT=smoke/evaluate_foundir_official_vlm_router_v4.py
echo "OFFICIAL_VLM_V4_START phase=$PHASE node=$NODE remaining=$REMAINING gpus=$GPUS"
if [[ "$PHASE" == route || "$PHASE" == render ]]; then
  step "$PYTHON" -u "$SCRIPT" hardware --node "$NODE"
  step "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node=2 --max_restarts=0 "$SCRIPT" "$PHASE" --node "$NODE"
elif [[ "$PHASE" == finish || "$PHASE" == plan ]]; then
  step "$PYTHON" -S "$SCRIPT" "$PHASE"
else
  step "$PYTHON" -u "$SCRIPT" prepare
fi
