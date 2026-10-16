#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
LOG="$ROOT/logs/foundir_proposal_energy_v2_oracle_feasibility_v1.log"

cd "$ROOT"
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "ERROR: expose exactly local GPUs 0-7" >&2
  exit 2
fi
export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_PROPOSAL_ENERGY_V2_ORACLE_START utc=$(date -u +%FT%TZ) allocated_gpus=8" | tee -a "$LOG"
echo "SCOPE fixed_256_development GT_oracle_only no_training no_protected_data" | tee -a "$LOG"
sha256sum \
  configs/foundir_proposal_relative_energy_noise_v2.json \
  smoke/evaluate_foundir_proposal_energy_v2_oracle_feasibility_ddp.py | tee -a "$LOG"
set +e
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  smoke/evaluate_foundir_proposal_energy_v2_oracle_feasibility_ddp.py 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
echo "FOUNDIR_PROPOSAL_ENERGY_V2_ORACLE_END status=$status utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$status"
