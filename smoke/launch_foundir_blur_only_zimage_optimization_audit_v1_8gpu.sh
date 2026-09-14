#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
SELF="$ROOT/smoke/launch_foundir_blur_only_zimage_optimization_audit_v1_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_optimization_audit_v1.json
SCRIPT=smoke/audit_foundir_blur_only_zimage_optimization.py
LOG=logs/foundir_blur_only_zimage_optimization_audit_v1.log
OUT=checkpoints/foundir_blur_only_zimage_optimization_audit_v1
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -eq 0 || ( $# -eq 1 && ( "$1" == --check || "$1" == --run ) ) ]]
bash -n "$SELF"
"$PYTHON" -m py_compile "$SCRIPT"
if [[ "${1:-}" == --check ]]; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/test_zimage_optimization_audit.py
  "$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]]
mkdir -p logs
if [[ "${1:-}" != --run ]]; then
  [[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: existing audit evidence; inspect before relaunch'; exit 1; }
  TMUX=$(command -v tmux || true)
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" -V
    "$TMUX" new-session -d -s foundir-zimage-optaudit-v1 -c "$ROOT"
    "$TMUX" set-option -t foundir-zimage-optaudit-v1 remain-on-exit on
    "$TMUX" send-keys -t foundir-zimage-optaudit-v1 "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --run" C-m
    echo "Attach: $TMUX attach -t foundir-zimage-optaudit-v1"
  else
    nohup bash "$SELF" --run >logs/foundir_blur_only_zimage_optimization_audit_v1_startup.log 2>&1 </dev/null &
    pid=$!
    printf '%s\n' "$pid" >logs/foundir_blur_only_zimage_optimization_audit_v1.pid
    sleep 3
    kill -0 "$pid"
  fi
  echo "Monitor: tail -f $LOG"
  exit 0
fi
exec 9>>logs/foundir_blur_only_zimage_optimization_audit_v1.lock
flock -n 9 || { echo 'ERROR: audit already running'; exit 1; }
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior evidence exists'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f "$OUT/result.json" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate zimage_optimization_audit_runtime --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image optimization audit' --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "OPT_AUDIT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT
echo "OPT_AUDIT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES independent_optimizers=true"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >logs/foundir_blur_only_zimage_optimization_audit_v1_gpu.csv 2>&1 &
telemetry_pid=$!
"$PYTHON" smoke/preflight_foundir_blur_only_zimage_vae_ceiling_solver_sweep.py --config configs/foundir_blur_only_zimage_vae_ceiling_v1.json --require-cuda
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$SCRIPT" --config "$CONFIG"
/usr/bin/python3 smoke/advisor_governance.py emit --gate zimage_optimization_audit_v1 --evidence "$OUT/result.json" --config "$CONFIG" --model 'Z-Image training-only latent/LoRA objective probes' --split two_existing_training_crops_only --outcome information --diagnostic-only --root-cause-or-status 'Eight independent probes complete; interpret latent vs LoRA and RGB vs mixed before next design.' --advisor-decision-question 'Continue under existing user authorization without another advisor round.'
