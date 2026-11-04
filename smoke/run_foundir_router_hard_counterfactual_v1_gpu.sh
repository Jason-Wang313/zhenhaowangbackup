#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: destination workspace only'; exit 2; }
[[ $# == 0 ]] || { echo 'ERROR: fixed four-image diagnostic takes no arguments'; exit 2; }
cd "$ROOT"
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified Python missing'; exit 2; }
export CUDA_VISIBLE_DEVICES=0,1
export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$ROOT/models/torch" PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
command -v flock >/dev/null
command -v timeout >/dev/null
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || { echo 'ERROR: another project GPU job is active; do not stop it'; exit 3; }
OUTPUT="$ROOT/checkpoints/foundir_router_hard_counterfactual_v1/predicted_hard_four_v1"
mkdir -p "$OUTPUT/attempts" logs
exec 8>"$OUTPUT/run.lock"
flock -n 8 || { echo 'ERROR: diagnostic already running'; exit 3; }
exec > >(tee -a logs/foundir_router_hard_counterfactual_v1.log) 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" <<'PY'
import json, pathlib, sys
folder=pathlib.Path(sys.argv[1])/'attempts'
starts=list(folder.glob('*.started.json')); ends=list(folder.glob('*.ended.json'))
if {p.name.removesuffix('.started.json') for p in starts} != {p.name.removesuffix('.ended.json') for p in ends}:
    raise SystemExit('ERROR: unfinished attempt accounting; inspect its log before manual recovery')
remaining=1800-sum(json.loads(p.read_text())['seconds'] for p in ends)
if remaining <= 0: raise SystemExit('ERROR: 30-minute cumulative cap exhausted; no auto expansion')
print(remaining)
PY
)
ATTEMPT=$(date -u +%Y%m%dT%H%M%S)_$$
START=$SECONDS
printf '{"start_utc":"%s","gpus":2}\n' "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish() {
  status=$?
  trap - EXIT
  elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"used_gpu_seconds":%s,"gpus":2}\n' "$status" "$elapsed" "$((2*elapsed))" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'ROUTER_HARD_END status=%s seconds=%s used_gpu_seconds=%s\n' "$status" "$elapsed" "$((2*elapsed))"
  exit "$status"
}
trap finish EXIT
step() {
  local remaining=$((REMAINING-SECONDS+START))
  ((remaining > 0)) || return 124
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@"
}
SCRIPT=smoke/diagnose_foundir_router_hard_counterfactual_v1.py
echo 'ROUTER_HARD_START nodes=1 gpus=2 renders=4 training=false cap_total_seconds=1800'
step "$PYTHON" -S "$SCRIPT" prepare
step "$PYTHON" -u "$SCRIPT" hardware
step "$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 "$SCRIPT" run
step "$PYTHON" -S "$SCRIPT" summarize
