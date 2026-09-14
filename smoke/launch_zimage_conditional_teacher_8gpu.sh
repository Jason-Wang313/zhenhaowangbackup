#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
SELF="$ROOT/smoke/launch_zimage_conditional_teacher_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_conditional_teacher_v1.json
SCRIPT=smoke/train_zimage_conditional_teacher.py
stage=tiny
mode=launch
scale_pilot=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="${2:?missing stage}"; shift 2 ;;
    --scale-pilot) scale_pilot=true; shift ;;
    --check) mode=check; shift ;;
    --run) mode=run; shift ;;
    *) echo "ERROR: unsupported argument $1"; exit 1 ;;
  esac
done
if [[ "$scale_pilot" == true ]]; then stage=pilot; fi
[[ "$stage" == tiny || "$stage" == pilot ]]
NAME="foundir_blur_only_zimage_conditional_teacher_v1_$stage"
LOG="logs/$NAME.log"
OUT="checkpoints/$NAME"
SESSION="zimage-conditional-$stage"
REVIEW=reports/zimage_conditional_teacher_tiny_visual_review_v1.json
AUTHORIZATION=reports/zimage_conditional_teacher_scale_authorization_v1.json
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple
[[ -x "$PYTHON" && -f "$CONFIG" && -f "$SCRIPT" ]]
/usr/bin/python3 smoke/advisor_governance.py check
bash -n "$SELF"
"$PYTHON" -m py_compile "$SCRIPT" lowlight_restoration/zimage_conditional_teacher.py smoke/prepare_zimage_conditional_pilot.py
args=(--config "$CONFIG" --stage "$stage")
if [[ "$stage" == pilot ]]; then
  if [[ "$scale_pilot" == true ]]; then
    args+=(--scale-authorization "$AUTHORIZATION")
  else
    args+=(--tiny-review "$REVIEW")
  fi
fi
"$PYTHON" "$SCRIPT" "${args[@]}" --admission-check
if [[ "$mode" == check ]]; then
  DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q tests/test_zimage_conditional_teacher.py tests/test_zimage_latent_first.py tests/test_zimage_padding_compat.py
  "$PYTHON" "$SCRIPT" "${args[@]}" --preflight
  exit 0
fi
[[ "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]]
command -v flock >/dev/null
command -v nvidia-smi >/dev/null
mkdir -p logs
if [[ "$mode" == launch ]]; then
  exec 8>>"logs/${NAME}_launch.lock"
  flock -n 8 || { echo 'ERROR: concurrent launch'; exit 1; }
  [[ ! -e "$LOG" && ! -e "$OUT" && ! -e "logs/${NAME}_startup.log" ]] || { echo 'ERROR: prior evidence exists; inspect before relaunch'; exit 1; }
  TMUX=$(command -v tmux || true)
  scale_arg=()
  if [[ "$scale_pilot" == true ]]; then scale_arg=(--scale-pilot); fi
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" -V
    "$TMUX" new-session -d -s "$SESSION" -c "$ROOT" 8>&-
    "$TMUX" set-option -t "$SESSION" remain-on-exit on
    "$TMUX" send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --stage '$stage' ${scale_arg[*]} --run" C-m
    echo "Attach: $TMUX attach -t $SESSION"
  else
    command -v nohup >/dev/null
    nohup bash "$SELF" --stage "$stage" "${scale_arg[@]}" --run >"logs/${NAME}_startup.log" 2>&1 </dev/null 8>&- &
    pid=$!
    printf '%s\n' "$pid" >"logs/$NAME.pid"
    sleep 3
    kill -0 "$pid" || { echo 'ERROR: startup exited; inspect startup log'; exit 1; }
  fi
  echo "Monitor: tail -f $LOG"
  exit 0
fi
exec 9>>logs/zimage_conditional_teacher.lock
flock -n 9 || { echo 'ERROR: conditional teacher already running'; exit 1; }
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior evidence exists'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f "$OUT/result.json" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate "zimage_conditional_${stage}_runtime" --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image Base conditional teacher' --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "CONDITIONAL_END stage=$stage status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT
echo "CONDITIONAL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES stage=$stage"
"$PYTHON" -c 'import torch; print("RUNTIME", torch.__version__, torch.version.cuda, torch.cuda.device_count())'
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >"logs/${NAME}_gpu.csv" 2>&1 &
telemetry_pid=$!
train_status=0
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$SCRIPT" "${args[@]}" || train_status=$?
if [[ -f "$OUT/result.json" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py emit --gate "zimage_conditional_teacher_$stage" --evidence "$OUT/result.json" --config "$CONFIG" --model 'Z-Image Base 6B conditional noise-to-GT teacher, persistent LQ, FP32 LoRA' --split "${stage}_training_archive_source_disjoint_development" --outcome auto --diagnostic-only --root-cause-or-status 'Conditional teacher stage finished. Review fixed images and metrics; tiny measures learnability, diverse pilot measures generalization.' --advisor-decision-question 'Complete visual review before the next authorized stage; no new advisor request.' || { report_status=$?; if [[ "$train_status" -eq 0 ]]; then train_status=$report_status; fi; }
fi
exit "$train_status"
