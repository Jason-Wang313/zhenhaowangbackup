#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang && $# == 0 ]] || { echo 'ERROR: fixed destination CPU job only'; exit 2; }
cd "$ROOT"
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified Python missing'; exit 2; }
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
command -v flock >/dev/null
command -v timeout >/dev/null
OUTPUT="$ROOT/checkpoints/foundir_router_supervised_repair_v1/fresh_ce400_matched_v1"
mkdir -p "$OUTPUT/attempts" logs
exec 9>"$OUTPUT/run.lock"
flock -n 9 || { echo 'ERROR: CPU repair already running'; exit 3; }
exec > >(tee -a logs/foundir_router_supervised_repair_v1.log) 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" <<'PY'
import json, pathlib, sys
folder=pathlib.Path(sys.argv[1])/'attempts'
starts=list(folder.glob('*.started.json')); ends=list(folder.glob('*.ended.json'))
if {p.name.removesuffix('.started.json') for p in starts} != {p.name.removesuffix('.ended.json') for p in ends}:
    raise SystemExit('ERROR: interrupted accounting; inspect log and record elapsed time before retry')
remaining=1800-sum(json.loads(p.read_text())['seconds'] for p in ends)
if remaining <= 0: raise SystemExit('ERROR: cumulative 30-minute CPU cap exhausted; no auto increase')
print(remaining)
PY
)
ATTEMPT=$(date -u +%Y%m%dT%H%M%S)_$$
START=$SECONDS
printf '{"start_utc":"%s","GPU_used":false}\n' "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish() {
  status=$?
  trap - EXIT
  elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"used_gpu_seconds":0}\n' "$status" "$elapsed" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'SUPERVISED_ROUTER_CPU_END status=%s seconds=%s GPU_used=false\n' "$status" "$elapsed"
  exit "$status"
}
trap finish EXIT
step() {
  local remaining=$((REMAINING-SECONDS+START))
  ((remaining > 0)) || return 124
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@"
}
SCRIPT=smoke/train_foundir_router_supervised_repair_v1.py
echo 'SUPERVISED_ROUTER_CPU_START fresh_arms=2 steps_each=400 global_batch=32 GPU_used=false'
step "$PYTHON" -S "$SCRIPT" prepare
step "$PYTHON" -u "$SCRIPT" train
step "$PYTHON" -S "$SCRIPT" verify
# No automatic GPU restoration, formal training, or budget increase.
