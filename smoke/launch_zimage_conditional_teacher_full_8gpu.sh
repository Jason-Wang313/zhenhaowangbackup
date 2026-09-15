#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
SELF="$ROOT/smoke/launch_zimage_conditional_teacher_full_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_conditional_teacher_full_v1.json
AUTHORIZATION=reports/zimage_conditional_teacher_full_scale_authorization_v1.json
TRAIN=smoke/train_zimage_conditional_teacher.py
EVALUATE=smoke/evaluate_zimage_conditional_teacher_full_ddp.py
NAME=foundir_blur_only_zimage_conditional_teacher_full_v1
LOG="logs/$NAME.log"
OUT="checkpoints/$NAME"
SESSION=zimage-conditional-full
mode=launch
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) mode=check; shift ;;
    --run) mode=run; shift ;;
    *) echo "ERROR: unsupported argument $1"; exit 1 ;;
  esac
done

export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring NCCL_PROTO=Simple

[[ -x "$PYTHON" && -f "$CONFIG" && -f "$AUTHORIZATION" && -f "$TRAIN" && -x "$EVALUATE" ]]
/usr/bin/python3 smoke/advisor_governance.py check
bash -n "$SELF"
"$PYTHON" -m py_compile "$TRAIN" "$EVALUATE" lowlight_restoration/zimage_conditional_teacher.py
args=(--config "$CONFIG" --stage full --full-authorization "$AUTHORIZATION")
"$PYTHON" "$TRAIN" "${args[@]}" --admission-check

if [[ "$mode" == check || "$mode" == launch ]]; then
  if [[ "$mode" == check ]]; then
    DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON" -m pytest -q \
      tests/test_zimage_conditional_teacher.py tests/test_zimage_latent_first.py tests/test_zimage_padding_compat.py
  fi
  "$PYTHON" "$TRAIN" "${args[@]}" --preflight
  if [[ "$mode" == check ]]; then exit 0; fi
fi

[[ "${CUDA_VISIBLE_DEVICES:-}" == 0,1,2,3,4,5,6,7 ]]
command -v flock >/dev/null
command -v nvidia-smi >/dev/null
mkdir -p logs

if [[ "$mode" == launch ]]; then
  exec 8>>"logs/${NAME}_launch.lock"
  flock -n 8 || { echo 'ERROR: concurrent launch'; exit 1; }
  [[ ! -e "$LOG" && ! -e "$OUT" && ! -e "logs/${NAME}_startup.log" ]] || {
    echo 'ERROR: prior full-scale evidence exists; inspect instead of relaunching'; exit 1;
  }
  TMUX=$(command -v tmux || true)
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" -V
    "$TMUX" new-session -d -s "$SESSION" -c "$ROOT" 8>&-
    "$TMUX" set-option -t "$SESSION" remain-on-exit on
    "$TMUX" send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --run" C-m
    echo "Attach: $TMUX attach -t $SESSION"
  else
    command -v nohup >/dev/null
    nohup bash "$SELF" --run >"logs/${NAME}_startup.log" 2>&1 </dev/null 8>&- &
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
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior full-scale evidence exists'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f "$OUT/full_development_result.json" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate zimage_conditional_teacher_full_runtime \
      --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image Base conditional teacher full scale' \
      --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "CONDITIONAL_FULL_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT

echo "CONDITIONAL_FULL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES"
"$PYTHON" -c 'import torch; print("RUNTIME", torch.__version__, torch.version.cuda, torch.cuda.device_count())'
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 \
  >"logs/${NAME}_gpu.csv" 2>&1 &
telemetry_pid=$!

train_status=0
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$TRAIN" "${args[@]}" || train_status=$?
if [[ "$train_status" -eq 0 && -f "$OUT/result.json" ]]; then
  "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$EVALUATE" \
    --config "$CONFIG" --full-authorization "$AUTHORIZATION" || train_status=$?
fi
if [[ -f "$OUT/full_development_result.json" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py emit --gate zimage_conditional_teacher_full \
    --evidence "$OUT/full_development_result.json" --config "$CONFIG" \
    --model 'Z-Image Base 6B conditional teacher, full Blur scale, FP32 LoRA' \
    --split '2595 training-archive source-disjoint Blur development records; diagnostic SOW proxy' \
    --outcome auto --diagnostic-only \
    --root-cause-or-status 'Maximum single-instance full-data continuation completed. Inspect all preselected full-development candidates and fixed triptychs before any distillation.' \
    --advisor-decision-question 'No new advisor request. User already authorized this full scale; visual review gates any distillation.' \
    || { report_status=$?; if [[ "$train_status" -eq 0 ]]; then train_status=$report_status; fi; }
fi
exit "$train_status"
