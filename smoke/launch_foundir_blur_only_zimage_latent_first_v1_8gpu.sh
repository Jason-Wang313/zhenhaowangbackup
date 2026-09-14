#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
SELF="$ROOT/smoke/launch_foundir_blur_only_zimage_latent_first_v1_8gpu.sh"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_latent_first_v1.json
SCRIPT=smoke/train_foundir_blur_only_zimage_latent_first_ddp.py
LOG=logs/foundir_blur_only_zimage_latent_first_v1.log
OUT=checkpoints/foundir_blur_only_zimage_latent_first_v1
SESSION=foundir-zimage-latent-first-v1
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
  exec 8>>logs/foundir_blur_only_zimage_latent_first_v1_launch.lock
  flock -n 8 || { echo 'ERROR: concurrent launch'; exit 1; }
  [[ ! -e "$LOG" && ! -e "$OUT" && ! -e logs/foundir_blur_only_zimage_latent_first_v1_startup.log ]] || { echo 'ERROR: prior evidence exists; inspect before relaunch'; exit 1; }
  TMUX=$(command -v tmux || true)
  if [[ -n "$TMUX" && -x "$TMUX" ]]; then
    "$TMUX" -V
    "$TMUX" new-session -d -s "$SESSION" -c "$ROOT" 8>&-
    "$TMUX" set-option -t "$SESSION" remain-on-exit on
    "$TMUX" send-keys -t "$SESSION" "export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; bash '$SELF' --run" C-m
    echo "Attach: $TMUX attach -t $SESSION"
  else
    command -v nohup >/dev/null
    nohup bash "$SELF" --run >logs/foundir_blur_only_zimage_latent_first_v1_startup.log 2>&1 </dev/null 8>&- &
    pid=$!
    printf '%s\n' "$pid" >logs/foundir_blur_only_zimage_latent_first_v1.pid
    sleep 3
    kill -0 "$pid" || { echo 'ERROR: startup exited; inspect startup log'; exit 1; }
  fi
  echo "Monitor: tail -f $LOG"
  exit 0
fi
exec 9>>logs/foundir_blur_only_zimage_latent_first_v1.lock
flock -n 9 || { echo 'ERROR: job already running'; exit 1; }
[[ ! -e "$LOG" && ! -e "$OUT" ]] || { echo 'ERROR: prior evidence exists'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f "$OUT/result.json" ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate zimage_latent_first_runtime --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image latent-first supervised endpoint repair' --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "LATENT_FIRST_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT
echo "LATENT_FIRST_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES world=8 global_batch=8 microbatch=1 accumulation=1 steps=1200"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 >logs/foundir_blur_only_zimage_latent_first_v1_gpu.csv 2>&1 &
telemetry_pid=$!
# Model/data hashes, native attention execution and two-stage rehearsal are
# checked inside the actual job. No separate GPU diagnostic or duplicate hash pass.
train_status=0
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 "$SCRIPT" --config "$CONFIG" || train_status=$?
if [[ -f "$OUT/result.json" ]]; then
  /usr/bin/python3 smoke/advisor_governance.py emit --gate zimage_latent_first_v1 --evidence "$OUT/result.json" --config "$CONFIG" --model 'Z-Image latent-first supervised endpoint repair, original frozen VAE' --split fixed_16_train_16_source_disjoint_development_crops --outcome auto --diagnostic-only --root-cause-or-status 'Bounded round-2 recipe finished. Inspect both metrics and fixed visuals; no successful teacher distillation or automatic full-training admission is claimed.' --advisor-decision-question 'Existing user authority needs no additional advisor round. Review final step 1200 and close the agreed two-round decision budget before further design.' || { report_status=$?; if [[ "$train_status" -eq 0 ]]; then train_status=$report_status; fi; }
fi
exit "$train_status"
