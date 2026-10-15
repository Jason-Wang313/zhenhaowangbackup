#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
NODE_RANK="${NODE_RANK:?set NODE_RANK to 0, 1, or 2}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to the reachable private IP of node 0}"
MASTER_PORT="${MASTER_PORT:-29621}"
LOG="$ROOT/logs/multinode_nccl_smoke_node${NODE_RANK}.log"

[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" || "$NODE_RANK" == "2" ]] || {
  echo "ERROR: NODE_RANK must be 0, 1, or 2" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly GPUs 0-7" >&2
  exit 1
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 1; }
mkdir -p "$ROOT/logs"
cd "$ROOT"

export MASTER_ADDR MASTER_PORT
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,COLL}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

exec > >(tee -a "$LOG") 2>&1
echo "MULTINODE_NCCL_SMOKE_START node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE 3_nodes 8_GPUs_per_node 24_total TCP_socket_test no_model no_dataset no_training"
"$PYTHON" -m torch.distributed.run \
  --nnodes=3 \
  --nproc_per_node=8 \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  smoke/multinode_nccl_smoke.py
echo "MULTINODE_NCCL_SMOKE_COMPLETE node_rank=$NODE_RANK utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
