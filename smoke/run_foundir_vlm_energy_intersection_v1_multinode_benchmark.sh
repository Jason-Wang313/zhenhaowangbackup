#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
NNODES="${NNODES:?set NNODES to 3, 4, or 5}"
NODE_RANK="${NODE_RANK:?set NODE_RANK to 0..NNODES-1}"
MASTER_ADDR="${MASTER_ADDR:?set MASTER_ADDR to node 0 private IP}"
MASTER_PORT="${MASTER_PORT:-29631}"
RUN_ID="${VLM_ENERGY_MULTINODE_BENCHMARK_RUN_ID:-benchmark_${NNODES}node_v1}"
LOG="$ROOT/logs/foundir_vlm_energy_multinode_benchmark_${RUN_ID}_node${NODE_RANK}.log"

[[ "$NNODES" =~ ^(3|4|5)$ ]] || { echo "ERROR: NNODES must be 3, 4, or 5" >&2; exit 2; }
(( NODE_RANK >= 0 && NODE_RANK < NNODES )) || { echo "ERROR: invalid NODE_RANK" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly GPUs 0-7" >&2; exit 2;
}

cd "$ROOT"
mkdir -p "$ROOT/logs"
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export VLM_ENERGY_MULTINODE_BENCHMARK_RUN_ID="$RUN_ID"

exec > >(tee -a "$LOG") 2>&1
echo "VLM_ENERGY_MULTINODE_BENCHMARK_START nnodes=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT utc=$(date -u +%FT%TZ)"
"$PYTHON" -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  smoke/benchmark_foundir_vlm_energy_intersection_v1_multinode.py
echo "VLM_ENERGY_MULTINODE_BENCHMARK_END status=0 utc=$(date -u +%FT%TZ)"
