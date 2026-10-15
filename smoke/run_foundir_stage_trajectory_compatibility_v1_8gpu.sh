#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_stage_trajectory_compatibility_v1.json"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
TRAINER="$ROOT/smoke/train_foundir_stage_trajectory_compatibility_ddp.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_stage_trajectory_compatibility_endpoint_ddp.py"
DIAGNOSTIC="$ROOT/smoke/diagnose_foundir_stage_trajectory_nccl_ddp.py"
DIAGNOSTIC_RESULT="$ROOT/checkpoints/foundir_stage_trajectory_compatibility_v1/nccl_rehearsal.json"
LOG="$ROOT/logs/foundir_stage_trajectory_compatibility_v1.log"
END="$ROOT/logs/foundir_stage_trajectory_compatibility_v1.end"
LOCK="$ROOT/logs/foundir_stage_trajectory_compatibility_v1.lock"

cd "$ROOT"
mkdir -p logs
exec 9>"$LOCK"
flock -n 9 || { echo "stage-trajectory run already active" >&2; exit 1; }
test ! -e "$END" || { echo "terminal marker already exists" >&2; exit 1; }

exec > >(tee -a "$LOG") 2>&1
echo "STAGE_TRAJECTORY_RUN_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PYTHON" smoke/advisor_governance.py check
"$PYTHON" "$TRAINER" --config "$CONFIG" --check
"$PYTHON" -m py_compile "$DIAGNOSTIC"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "0,1,2,3,4,5,6,7" ]]; then
  echo "CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3,4,5,6,7" >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import torch
if torch.cuda.device_count() != 8:
    raise RuntimeError(f"expected 8 visible GPUs, found {torch.cuda.device_count()}")
names = [torch.cuda.get_device_name(i) for i in range(8)]
if not (all("H100" in name for name in names) or all("H200" in name for name in names)):
    raise RuntimeError(f"requires homogeneous H100 or H200 GPUs: {names}")
print({"gpu_names": names})
PY

export NCCL_P2P_DISABLE=0 NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -f "$DIAGNOSTIC_RESULT" ]]; then
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$DIAGNOSTIC"
fi
"$PYTHON" - <<'PY'
import json
from pathlib import Path
path = Path("checkpoints/foundir_stage_trajectory_compatibility_v1/nccl_rehearsal.json")
result = json.loads(path.read_text())
assert result["status"] == "stage_trajectory_nccl_ddp_rehearsal_pass"
assert result["optimizer_steps"] == 0 and result["model_weight_updates"] is False
print("STAGE_TRAJECTORY_NCCL_REHEARSAL_VERIFIED")
PY

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$TRAINER" --config "$CONFIG"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
path = Path("checkpoints/foundir_stage_trajectory_compatibility_v1/development_selection.json")
result = json.loads(path.read_text())
if result.get("endpoint_promotion_pass") is not True:
    raise RuntimeError("double-only gate failed; triple endpoint remains sealed")
print("STAGE_TRAJECTORY_DOUBLE_GATE_PASS")
PY
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$EVALUATOR" --config "$CONFIG"

echo "status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$END"
echo "STAGE_TRAJECTORY_RUN_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
