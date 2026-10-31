#!/usr/bin/env bash
# Shared validation only; sourcing this file does not launch a job.
configure_native_energy() {
  ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
  RUN_ID=${1:-qualification_3x2_v1}; GPUS=${2:-6}; RESUME=${3:-fresh}
  [[ "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'ERROR: invalid run ID'; return 2; }
  [[ "$RESUME" == fresh || "$RESUME" == resume ]] || return 2
  case "$GPUS" in
    6) EXPECTED_NODES=3; LOCAL_GPUS=2; DEFAULT_VISIBLE=0,1 ;;
    2|4|8)
      EXPECTED_NODES=1; LOCAL_GPUS=$GPUS
      case "$GPUS" in
        2) DEFAULT_VISIBLE=0,1 ;; 4) DEFAULT_VISIBLE=0,1,2,3 ;; 8) DEFAULT_VISIBLE=0,1,2,3,4,5,6,7 ;;
      esac ;;
    *) echo 'ERROR: use total GPUs=6 (3x2 default), or explicit local 2/4/8'; return 2 ;;
  esac
  if [[ "$EXPECTED_NODES" == 3 ]]; then
    NNODES=${NNODES:-3}
    [[ "$NNODES" == 3 && "${NODE_RANK:-}" =~ ^[0-2]$ && -n "${MASTER_ADDR:-}" ]] || {
      echo 'ERROR: set NODE_RANK=0/1/2 and the same reachable node0 MASTER_ADDR on all 3 nodes'; return 2;
    }
    [[ "$MASTER_ADDR" =~ ^[A-Za-z0-9_.:-]+$ && "$MASTER_ADDR" != localhost && "$MASTER_ADDR" != 127.* && "$MASTER_ADDR" != ::1 && "$MASTER_ADDR" != 0.0.0.0 ]] || return 2
  else
    NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1
  fi
  MASTER_PORT=${MASTER_PORT:-29731}
  [[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] && ((10#$MASTER_PORT>=1024 && 10#$MASTER_PORT<=65535)) || return 2
  VISIBLE=${CUDA_VISIBLE_DEVICES:-$DEFAULT_VISIBLE}
  local devices=() item seen=,
  IFS=, read -r -a devices <<< "$VISIBLE"
  [[ ${#devices[@]} == "$LOCAL_GPUS" && "$VISIBLE" != *, ]] || {
    echo "ERROR: expose exactly $LOCAL_GPUS local GPUs, not $GPUS on each node"; return 2;
  }
  for item in "${devices[@]}"; do
    [[ "$item" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ && "$seen" != *",$item,"* ]] || return 2
    seen+="$item,"
  done
  PYTHON=${FLUX_NATIVE_PYTHON:-"$ROOT/.conda/gpu-train-stable/bin/python"}
  TRANSPORT=("NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}" "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}")
  [[ -z "${NCCL_SOCKET_IFNAME:-}" ]] || TRANSPORT+=("NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
  [[ -z "${GLOO_SOCKET_IFNAME:-}" ]] || TRANSPORT+=("GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME")
  TORCHRUN=("$PYTHON" -u -m torch.distributed.run --nnodes="$NNODES" --nproc_per_node="$LOCAL_GPUS"
    --max_restarts=0)
  if [[ "$NNODES" == 1 ]]; then TORCHRUN+=(--standalone)
  else TORCHRUN+=(--node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --rdzv_conf=timeout=900); fi
  TORCHRUN+=("$ROOT/smoke/train_foundir_flux_native_energy_v1.py" --run-id "$RUN_ID" --world-size "$GPUS")
  [[ "$RESUME" != resume ]] || TORCHRUN+=(--resume)
  return 0
}
