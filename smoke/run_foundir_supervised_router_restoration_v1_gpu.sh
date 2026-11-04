#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: destination workspace only'; exit 2; }
[[ $# == 0 ]] || { echo 'ERROR: fixed four-render evaluation takes no arguments'; exit 2; }
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
OUTPUT="$ROOT/checkpoints/foundir_supervised_router_restoration_v1/matched_top1_four_missing_v1"
mkdir -p "$OUTPUT/attempts" logs
exec 8>"$OUTPUT/run.lock"
flock -n 8 || { echo 'ERROR: evaluation already running'; exit 3; }
exec > >(tee -a logs/foundir_supervised_router_restoration_v1.log) 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" <<'PY'
import json, pathlib, sys
folder=pathlib.Path(sys.argv[1])/'attempts'
starts=list(folder.glob('*.started.json')); ends=list(folder.glob('*.ended.json'))
if {p.name.removesuffix('.started.json') for p in starts} != {p.name.removesuffix('.ended.json') for p in ends}:
    raise SystemExit('ERROR: unfinished attempt accounting; inspect its log before manual recovery')
seconds=[json.loads(p.read_text())['seconds'] for p in ends]
if any(type(s) is not int or s < 0 for s in seconds):
    raise SystemExit('ERROR: invalid attempt accounting')
remaining=900-sum(seconds)
if remaining <= 0: raise SystemExit('ERROR: 15-minute cumulative cap exhausted; no auto expansion')
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
  printf 'MATCHED_RESTORATION_END status=%s seconds=%s used_gpu_seconds=%s\n' "$status" "$elapsed" "$((2*elapsed))"
  exit "$status"
}
trap finish EXIT
step() {
  local remaining=$((REMAINING-SECONDS+START))
  ((remaining > 0)) || return 124
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@"
}
SCRIPT=smoke/evaluate_foundir_supervised_router_restoration_v1.py
echo 'MATCHED_RESTORATION_START nodes=1 gpus=2 model=H200 renders=4 reused_rows=76 training=false cap_total_seconds=900'
step "$PYTHON" -S "$SCRIPT" prepare
step "$PYTHON" -u "$SCRIPT" hardware
step "$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 --max_restarts=0 "$SCRIPT" run
step "$PYTHON" -S "$SCRIPT" summarize
