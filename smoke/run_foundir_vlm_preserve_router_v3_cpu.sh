#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: destination only'; exit 2; }
[[ $# == 0 ]] || { echo 'ERROR: fixed CPU recipe; no override arguments'; exit 2; }
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
[[ -x "$PYTHON" ]] || { echo 'ERROR: qualified Python missing'; exit 2; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
OUTPUT="$ROOT/checkpoints/foundir_vlm_preserve_router_v3/preserve_correct_v3"
mkdir -p "$OUTPUT/attempts" logs
exec 9>/tmp/zhenhaowang_frozen_image_router_cpu.lock
flock -n 9 || { echo 'ERROR: CPU router task active; do not kill it'; exit 3; }
exec 8>"$OUTPUT/cpu_train.lock"
flock -n 8 || { echo 'ERROR: same run active on another node'; exit 3; }
exec > >(tee -a logs/foundir_vlm_preserve_router_v3_cpu_train.log) 2>&1
REMAINING=$("$PYTHON" -S - "$OUTPUT" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'attempts'
starts=list(p.glob('cpu_train_*.started.json')); ends=list(p.glob('cpu_train_*.ended.json'))
if {f.name.removesuffix('.started.json') for f in starts}!={f.name.removesuffix('.ended.json') for f in ends}:
    raise SystemExit('ERROR: interrupted accounting; investigate before retry')
times=[json.loads(f.read_text())['seconds'] for f in ends]
if any(type(t) is not int or t<0 for t in times): raise SystemExit('ERROR: invalid accounting')
left=3600-sum(times)
if left<=0: raise SystemExit('ERROR: cumulative one-hour budget exhausted')
print(left)
PY
)
ATTEMPT="cpu_train_$(date -u +%Y%m%dT%H%M%S)_$$"; START=$SECONDS
printf '{"gpus":0,"start_utc":"%s"}\n' "$(date -u +%FT%TZ)" > "$OUTPUT/attempts/$ATTEMPT.started.json"
finish() {
  status=$?; trap - EXIT; elapsed=$((SECONDS-START))
  printf '{"status":%s,"seconds":%s,"gpus":0,"used_gpu_seconds":0}\n' "$status" "$elapsed" > "$OUTPUT/attempts/$ATTEMPT.ended.json"
  printf 'VLM_PRESERVE_V3_END status=%s seconds=%s gpus=0\n' "$status" "$elapsed"
  exit "$status"
}
trap finish EXIT
step() {
  local left=$((REMAINING-SECONDS+START)); ((left>0)) || return 124
  timeout --signal=TERM --kill-after=30s "${left}s" "$@"
}
SCRIPT=smoke/repair_foundir_vlm_preserve_router_v3.py
echo "VLM_PRESERVE_V3_START cpu_only=true cumulative_remaining=$REMAINING"
step "$PYTHON" -S "$SCRIPT" prepare
step "$PYTHON" -u "$SCRIPT" train
step "$PYTHON" -S "$SCRIPT" verify
# No feature extraction, old-holdout rerun, restoration, or automatic next trial.
