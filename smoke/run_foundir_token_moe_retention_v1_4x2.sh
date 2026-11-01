#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EXPECTED_ROOT=/inspire/hdd/project/multimodal-discrete-diffusion/liuxiaohong-25080/zhenhaowang
if [[ "$ROOT" != "$EXPECTED_ROOT" ]]; then
  echo "ERROR: use the migrated GPU workspace: $EXPECTED_ROOT" >&2
  exit 2
fi
NODE_RANK="${1:?Pass node rank 0,1,2,3}"
MASTER_ADDR="${2:?Pass CURRENT node0 address}"
RUN_ID="${3:-qualification_4x2_v1}"
RESUME_RECEIPT="${4:-}"
MASTER_PORT="${MASTER_PORT:-29768}"
if [[ ! "$NODE_RANK" =~ ^[0-3]$ || ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ || ! "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
  echo 'ERROR: invalid rank/run ID/port' >&2
  exit 2
fi
PYTHON="${PROJECT_PYTHON:-/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python}"
if [[ ! -x "$PYTHON" ]]; then echo 'ERROR: set PROJECT_PYTHON to the verified environment'; exit 2; fi
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_NVLS_ENABLE=0
export PYTHONPATH="$ROOT:$ROOT/third_party/flux2/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs checkpoints/foundir_token_residual_moe_v1/node_locks
exec 9>"checkpoints/foundir_token_residual_moe_v1/node_locks/$(hostname).lock"
if ! flock -n 9; then echo 'ERROR: another token MoE job is active on this node'; exit 3; fi
LOG="logs/token_moe_retention_${RUN_ID}_node${NODE_RANK}_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG") 2>&1
START=$SECONDS
echo "TOKEN_MOE_RETENTION_START topology=4x2 world=8 node_rank=$NODE_RANK run=$RUN_ID parent=2000 new_phase_steps=25 fresh_Adam=true batch=40"
echo 'PRELAUNCH_ESTIMATE wall_hours=0.3_to_0.8 allocated_GPU_hours=2.4_to_6.4 hard_cap_hours=1.5 estimate_not_guarantee'
trap 'status=$?; echo "TOKEN_MOE_RETENTION_END status=$status seconds=$((SECONDS-START)) node_rank=$NODE_RANK"' EXIT
args=(--nnodes=4 --nproc_per_node=2 --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --max_restarts=0)
resume=()
# Fourth argument ONLY resumes this new batch40 phase; old batch32 is rejected.
if [[ -n "$RESUME_RECEIPT" ]]; then resume=(--resume "$RESUME_RECEIPT"); fi
for phase in features reference train; do
  REMAINING=$((5400-SECONDS+START))
  if (( REMAINING <= 0 )); then echo 'ERROR: qualification time budget exhausted'; exit 124; fi
  timeout --signal=TERM --kill-after=30s "${REMAINING}s" "$PYTHON" -u -m torch.distributed.run "${args[@]}" \
    smoke/train_foundir_token_moe_retention_v1.py --phase "$phase" --run-id "$RUN_ID" "${resume[@]}"
done
# No automatic formal/mixed training or work on the independent energy branch.
