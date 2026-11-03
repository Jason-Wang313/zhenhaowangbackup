#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$ROOT" == /inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang ]] || { echo 'ERROR: use migrated GPU workspace'; exit 2; }
cd "$ROOT"
[[ $# == 4 ]] || { echo 'Usage: launcher <node_rank 0..3> <current_node0_address> <run_id> <fresh_shared_launch_id>'; exit 2; }
NODE_RANK=$1 MASTER_ADDR=$2 RUN_ID=$3 LAUNCH_ID=$4
[[ "$NODE_RANK" =~ ^[0-3]$ && "$MASTER_ADDR" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ ]] || exit 2
[[ "$RUN_ID" =~ ^[A-Za-z0-9_-]{1,80}$ && "$LAUNCH_ID" =~ ^[A-Za-z0-9_-]{1,80}$ ]] || exit 2
MASTER_PORT=${REPLAY_MASTER_PORT:-29698}
[[ "$MASTER_PORT" =~ ^[0-9]{4,5}$ ]] && ((10#$MASTER_PORT <= 65535)) || exit 2
export REPLAY_NODE_RANK="$NODE_RANK" REPLAY_MASTER_ADDR="$MASTER_ADDR" REPLAY_MASTER_PORT="$MASTER_PORT"
export CUDA_VISIBLE_DEVICES=0,1
PYTHON="${PROJECT_PYTHON:-$ROOT/.conda/gpu-train-stable/bin/python}"
[[ -x "$PYTHON" ]] || { echo 'ERROR: verified Python unavailable'; exit 2; }
export CUDNN_CONV_WSCAP_DBG=256 TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_NVLS_ENABLE=0
export TORCH_HOME="$ROOT/models/torch" PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src"
command -v flock >/dev/null
command -v timeout >/dev/null
OUTPUT="$ROOT/checkpoints/foundir_rain_onpolicy_continue_v1/$RUN_ID"
mkdir -p "$OUTPUT/attempts" logs
exec 9>/tmp/zhenhaowang_existing_flux_replay.lock
flock -n 9 || { echo 'ERROR: replay already active on this node'; exit 3; }
exec 8>"$OUTPUT/node${NODE_RANK}.lock"
flock -n 8 || { echo 'ERROR: same node rank already active'; exit 3; }
LOG="$ROOT/logs/foundir_rain_onpolicy_continue_v1_${RUN_ID}_node${NODE_RANK}.log"
exec > >(tee -a "$LOG") 2>&1
START=$SECONDS
WALL_CAP=10800
CLUSTER=("$PYTHON" -u smoke/train_foundir_rain_onpolicy_continue_v1.py)
COMMON=(--run-id "$RUN_ID" --launch-id "$LAUNCH_ID" --node-rank "$NODE_RANK")
step() {
  local remaining=$((WALL_CAP-SECONDS+START))
  ((remaining > 0)) || return 124
  timeout --signal=TERM --kill-after=30s "${remaining}s" "$@" &
  local task_pid=$!
  while kill -0 "$task_pid" 2>/dev/null; do
    if compgen -G "$OUTPUT/launches/$LAUNCH_ID/failed_node*.json" >/dev/null; then
      echo 'ERROR: another node failed; stopping only this launcher child'
      kill -TERM "$task_pid" 2>/dev/null || true
      wait "$task_pid" || true
      return 1
    fi
    sleep 2
  done
  wait "$task_pid"
}
# Duplicate attempts must not poison the live launch.
step "${CLUSTER[@]}" publish --event boot "${COMMON[@]}"
finish() {
  status=$?
  elapsed=$((SECONDS-START))
  trap - EXIT
  if ((status != 0)); then
    timeout --kill-after=5s 15s "${CLUSTER[@]}" publish --event failed "${COMMON[@]}" || true
  fi
  printf 'RAIN_ONPOLICY_CONTINUE_NODE_END node=%s status=%s elapsed_seconds=%s allocated_gpu_seconds=%s launch=%s\n' "$NODE_RANK" "$status" "$elapsed" "$((2*elapsed))" "$LAUNCH_ID"
  (set -o noclobber; printf '{"node":%s,"status":%s,"seconds":%s,"allocated_gpu_seconds":%s,"gpus":2}\n' "$NODE_RANK" "$status" "$elapsed" "$((2*elapsed))" > "$OUTPUT/attempts/${LAUNCH_ID}_node${NODE_RANK}.json")
  exit "$status"
}
trap finish EXIT
echo "RAIN_ONPOLICY_CONTINUE_START run=$RUN_ID nodes=4 local_gpus=2 total_gpus=8 node=$NODE_RANK launch=$LAUNCH_ID parent_step=600 new_updates=200 fixed_call=25 global_batch=32 teacher_inference=false"
echo 'ESTIMATE 75_to_90_minutes_unmeasured; cap=3h_per_node,total<=24_GPU_hours_plus_cleanup; update ETA from first5 steps and checkpoint cost'
SCRIPT=smoke/train_foundir_rain_onpolicy_continue_v1.py
step "${CLUSTER[@]}" wait --event boot "${COMMON[@]}"
step "$PYTHON" -u "$SCRIPT" hardware "${COMMON[@]}"
if [[ "$NODE_RANK" == 0 ]]; then
  step "$PYTHON" -u "$SCRIPT" prepare "${COMMON[@]}"
  step "${CLUSTER[@]}" publish --event prepared "${COMMON[@]}"
fi
step "${CLUSTER[@]}" wait --event prepared "${COMMON[@]}"
step "$PYTHON" -u "$SCRIPT" runtime "${COMMON[@]}"
step "${CLUSTER[@]}" publish --event ready "${COMMON[@]}"
step "${CLUSTER[@]}" wait --event ready "${COMMON[@]}"
# Shared node lock refuses concurrent jobs. Restore full step600 state; keep objective unchanged.
step "$PYTHON" -u -m torch.distributed.run --nnodes=4 --nproc_per_node=2 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_conf=timeout=3600 --max_restarts=0 \
  "$SCRIPT" run "${COMMON[@]}"
step "${CLUSTER[@]}" publish --event done "${COMMON[@]}"
step "${CLUSTER[@]}" wait --event done "${COMMON[@]}"
if [[ "$NODE_RANK" == 0 ]]; then
  step "$PYTHON" -u "$SCRIPT" summarize "${COMMON[@]}"
  step "${CLUSTER[@]}" publish --event verified "${COMMON[@]}"
fi
step "${CLUSTER[@]}" wait --event verified "${COMMON[@]}"
echo 'RAIN_ONPOLICY_CONTINUE_GPU_COMPLETE new_updates=200 full_CPU_audit=false quality_success_not_implied'
# No automatic extension, MoE, energy or follow-on.
