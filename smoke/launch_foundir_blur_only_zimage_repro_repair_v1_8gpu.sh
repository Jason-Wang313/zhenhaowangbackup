#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
SELF="$ROOT/smoke/launch_foundir_blur_only_zimage_repro_repair_v1_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_repro_repair_v1.json
SCRIPT=smoke/repair_foundir_blur_only_zimage_repro_batch.py
LOG=logs/foundir_blur_only_zimage_repro_repair_v1.log
OUT=checkpoints/foundir_blur_only_zimage_repro_repair_v1
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 CUBLAS_WORKSPACE_CONFIG=:4096:8
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -eq 0 || ( $# -eq 1 && ( "$1" == --check || "$1" == --run ) ) ]]
[[ -x "$PYTHON" ]]
bash -n "$SELF"
"$PYTHON" -m py_compile "$SCRIPT" lowlight_restoration/zimage_repro_batch.py
if [[ "${1:-}" == --check ]]; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/test_zimage_padding_compat.py tests/test_zimage_repro_batch.py tests/test_zimage_lora_stability.py tests/test_zimage_optimization_audit.py
  "$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]]
mkdir -p logs
if [[ "${1:-}" != --run ]]; then
  [[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: existing batch evidence; inspect before relaunch'; exit 1; }
  TMUX=$(command -v tmux || true)
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" -V
    "$TMUX" new-session -d -s foundir-zimage-repro-repair-v1 -c "$ROOT"
    "$TMUX" set-option -t foundir-zimage-repro-repair-v1 remain-on-exit on
    "$TMUX" send-keys -t foundir-zimage-repro-repair-v1 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --run" C-m
    echo "Attach: $TMUX attach -t foundir-zimage-repro-repair-v1"
  else
    nohup bash "$SELF" --run >logs/foundir_blur_only_zimage_repro_repair_v1_startup.log 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >logs/foundir_blur_only_zimage_repro_repair_v1.pid
    sleep 3
    kill -0 "$pid"
  fi
  echo "Monitor: tail -f $LOG"
  echo "Worker logs: $OUT/worker_2.log, worker_3.log, worker_6.log, worker_7.log"
  exit 0
fi
exec 9>>logs/foundir_blur_only_zimage_repro_repair_v1.lock
flock -n 9 || { echo 'ERROR: batch already running'; exit 1; }
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior evidence exists'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f "$OUT/result.json" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate zimage_repro_repair_runtime --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image consolidated reproducibility audit' --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "REPRO_BATCH_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT
echo "REPRO_BATCH_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES new_fits=12 reused_fits=12 worker_ranks=2,3,6,7 independent_optimizers=true"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >logs/foundir_blur_only_zimage_repro_repair_v1_gpu.csv 2>&1 &
telemetry_pid=$!
"$PYTHON" smoke/preflight_foundir_blur_only_zimage_vae_ceiling_solver_sweep.py --config configs/foundir_blur_only_zimage_vae_ceiling_v1.json --require-cuda
batch_status=0
"$PYTHON" "$SCRIPT" --config "$CONFIG" || batch_status=$?
if [[ -f "$OUT/result.json" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py emit --gate zimage_repro_repair_v1 --evidence "$OUT/result.json" --config "$CONFIG" --model 'Z-Image repaired deterministic controls with immutable ordinary-fit reuse' --split two_existing_training_crops_only --outcome information --diagnostic-only --root-cause-or-status 'Deterministic padding repair supplement finished; inspect CUDA equivalence gates and combined controls. No automatic quality pass.' --advisor-decision-question 'Continue within the existing user-authorized two-round decision budget; no additional advisor round.'
fi
exit "$batch_status"
