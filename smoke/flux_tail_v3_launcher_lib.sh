#!/usr/bin/env bash
# Shared argument validation. Sourcing this file never starts a process.
configure_tail_v3() {
  ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
  PYTHON=${FLUX_TAIL_V3_PYTHON:-}
  if [[ "$PYTHON" != /* || ! -f "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo 'ERROR: set FLUX_TAIL_V3_PYTHON to the absolute qualified interpreter' >&2
    return 2
  fi
  MODE=${1:-qualification}
  GPU_COUNT=${3:-6}
  case "$GPU_COUNT" in
    6) EXPECTED_NODES=3; LOCAL_GPUS=2; DEFAULT_VISIBLE=0,1; LAYOUT=3x2 ;;
    4|8) EXPECTED_NODES=1; LOCAL_GPUS=$GPU_COUNT; LAYOUT=1x${GPU_COUNT}
      DEFAULT_VISIBLE=0,1,2,3
      [[ "$GPU_COUNT" != 8 ]] || DEFAULT_VISIBLE=0,1,2,3,4,5,6,7 ;;
    *) echo 'ERROR: supported layouts: 3x2=6, 1x4, 1x8' >&2; return 2 ;;
  esac
  NNODES=${NNODES:-$EXPECTED_NODES}
  [[ "$NNODES" == "$EXPECTED_NODES" ]] || { echo 'ERROR: NNODES/layout mismatch' >&2; return 2; }
  RUN_ID=${2:-${MODE}_${LAYOUT}_v1}
  QUALIFICATION_RUN_ID=${4:-qualification_${LAYOUT}_v1}
  [[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ && "$QUALIFICATION_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || return 2
  case "$MODE" in
    qualification) TRAIN_MODE=qualification; WALL_CAP=$((43200/GPU_COUNT)); EXTRA_ARGS=() ;;
    trainfit) TRAIN_MODE=trainfit; WALL_CAP=$((288000/GPU_COUNT)); EXTRA_ARGS=() ;;
    connectivity) TRAIN_MODE=qualification; WALL_CAP=300; EXTRA_ARGS=(--connectivity-only) ;;
    *) echo 'ERROR: mode must be qualification, connectivity or trainfit' >&2; return 2 ;;
  esac
  if [[ "$NNODES" == 1 ]]; then
    NODE_RANK=${NODE_RANK:-0}; MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  else
    [[ -n "${NODE_RANK:-}" && -n "${MASTER_ADDR:-}" ]] || {
      echo 'ERROR: set NODE_RANK=0/1/2 and the same reachable MASTER_ADDR on all three nodes' >&2; return 2;
    }
    [[ "$MASTER_ADDR" != localhost && "$MASTER_ADDR" != 127.* && "$MASTER_ADDR" != ::1 ]] || return 2
  fi
  [[ "$NODE_RANK" =~ ^[0-2]$ ]] && ((NODE_RANK<NNODES)) || return 2
  MASTER_PORT=${MASTER_PORT:-29683}
  [[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && ((10#$MASTER_PORT>=1024 && 10#$MASTER_PORT<=65535)) || return 2
  VISIBLE=${CUDA_VISIBLE_DEVICES:-$DEFAULT_VISIBLE}
  local devices=() value seen=,
  IFS=, read -r -a devices <<< "$VISIBLE"
  [[ ${#devices[@]} == "$LOCAL_GPUS" ]] || { echo 'ERROR: visible device count/layout mismatch' >&2; return 2; }
  for value in "${devices[@]}"; do
    [[ "$value" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ && "$seen" != *",$value,"* ]] || return 2
    seen+="$value,"
  done
  TRANSPORT=("NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}")
  [[ -z "${NCCL_SOCKET_IFNAME:-}" ]] || TRANSPORT+=("NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
  [[ -z "${GLOO_SOCKET_IFNAME:-}" ]] || TRANSPORT+=("GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME")
  LOG="$ROOT/logs/foundir_flux_tail_relative_energy_v3_${RUN_ID}_node${NODE_RANK}.log"
}

tail_v3_torchrun_command() {
  TORCHRUN=("$PYTHON" -u -m torch.distributed.run
    --nnodes="$NNODES" --nproc_per_node="$LOCAL_GPUS" --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    --rdzv_conf=timeout=180 --max-restarts=0
    "$ROOT/smoke/run_foundir_flux_tail_relative_energy_v3.py" pipeline
    --mode "$TRAIN_MODE" --run-id "$RUN_ID" --gpus "$GPU_COUNT"
    --qualification-run-id "$QUALIFICATION_RUN_ID" "${EXTRA_ARGS[@]}")
}
