#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PYTHON=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
LOG="$ROOT/logs/foundir_complete_candidate_energy_risk_coverage_v1.log"
cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose GPUs 0-7" >&2; exit 2; }
export TORCH_HOME="$ROOT/models/torch" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "FOUNDIR_COMPLETE_ENERGY_RISK_COVERAGE_START utc=$(date -u +%FT%TZ) allocated_gpus=8" | tee -a "$LOG"
sha256sum configs/foundir_complete_candidate_energy_risk_coverage_v1.json smoke/evaluate_foundir_complete_candidate_energy_risk_coverage_v1_ddp.py | tee -a "$LOG"
set +e
"$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 smoke/evaluate_foundir_complete_candidate_energy_risk_coverage_v1_ddp.py 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}; set -e
echo "FOUNDIR_COMPLETE_ENERGY_RISK_COVERAGE_END status=$status utc=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$status"
