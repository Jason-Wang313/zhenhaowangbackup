#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
SELF="$ROOT/smoke/launch_foundir_blur_only_zimage_latent_first_v2_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_latent_first_v2.json
SCRIPT=smoke/train_foundir_blur_only_zimage_latent_first_ddp.py
LOG=logs/foundir_blur_only_zimage_latent_first_v2.log
OUT=checkpoints/foundir_blur_only_zimage_latent_first_v2
SESSION=foundir-zimage-latent-first-v2
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -eq 0 || ( $# -eq 1 && ( "$1" == --check || "$1" == --run ) ) ]]
[[ -x "$PYTHON" && -f "$CONFIG" && -f "$SCRIPT" ]]
bash -n "$SELF"
"$PYTHON" -m py_compile "$SCRIPT" lowlight_restoration/zimage_latent_first.py
if [[ "${1:-}" == --check ]]; then
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/test_zimage_latent_first.py tests/test_zimage_padding_compat.py tests/test_zimage_repro_batch.py tests/test_zimage_endpoint_distillation.py tests/test_foundir_zimage_paired_flow_teacher.py
  "$PYTHON" "$SCRIPT" --config "$CONFIG" --preflight
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]]
command -v flock >/dev/null
command -v nvidia-smi >/dev/null
mkdir -p logs
if [[ "${1:-}" != --run ]]; then
  exec 8>>logs/foundir_blur_only_zimage_latent_first_v2_launch.lock
  flock -n 8 || { echo 'ERROR: concurrent launch'; exit 1; }
  [[ ! -e "$LOG" && ! -e "$OUT" && ! -e logs/foundir_blur_only_zimage_latent_first_v2_startup.log ]] || { echo 'ERROR: prior v2 evidence exists; inspect before relaunch'; exit 1; }
  TMUX=$(command -v tmux || true)
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" -V
    "$TMUX" new-session -d -s "$SESSION" -c "$ROOT" 8>&-
    "$TMUX" set-option -t "$SESSION" remain-on-exit on
    "$TMUX" send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --run" C-m
    echo "Attach: $TMUX attach -t $SESSION"
  else
    command -v nohup >/dev/null
    nohup bash "$SELF" --run >logs/foundir_blur_only_zimage_latent_first_v2_startup.log 2>&1 </dev/null 8>&- &
    pid=$!
    printf '%s\n' "$pid" >logs/foundir_blur_only_zimage_latent_first_v2.pid
    sleep 3
    kill -0 "$pid" || { echo 'ERROR: startup exited; inspect startup log'; exit 1; }
  fi
  echo "Monitor: tail -f $LOG"
  exit 0
fi
exec 9>>logs/foundir_blur_only_zimage_latent_first_v2.lock
flock -n 9 || { echo 'ERROR: v2 job already running'; exit 1; }
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior v2 evidence exists'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f "$OUT/result.json" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate zimage_latent_first_v2_runtime --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image latent-first v2 baseline-replay recovery' --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "LATENT_FIRST_V2_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT
echo "LATENT_FIRST_V2_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES world=8 global_batch=8 steps=1200 strict_baseline_repeats=2"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >logs/foundir_blur_only_zimage_latent_first_v2_gpu.csv 2>&1 &
telemetry_pid=$!
train_status=0
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$SCRIPT" --config "$CONFIG" || train_status=$?
if [[ -f "$OUT/result.json" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py emit --gate zimage_latent_first_v2 --evidence "$OUT/result.json" --config "$CONFIG" --model 'Z-Image latent-first supervised endpoint repair v2, original frozen VAE' --split fixed_16_train_16_source_disjoint_development_crops --outcome auto --diagnostic-only --root-cause-or-status 'Bounded round-2 recipe finished after exact strict baseline replay. Review both metrics and visuals; no successful teacher distillation or automatic full-data admission.' --advisor-decision-question 'Close the agreed two-round decision budget after visual review; no additional advisor request was required for this authorized recovery.' || { report_status=$?; if [[ "$train_status" -eq 0 ]]; then train_status=$report_status; fi; }
fi
exit "$train_status"
