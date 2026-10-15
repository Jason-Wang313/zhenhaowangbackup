#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
DIAGNOSTIC="$ROOT/smoke/diagnose_foundir_lowlight_haze_nccl_ddp.py"
DIAGNOSTIC_RESULT="$ROOT/checkpoints/foundir_lowlight_haze_restormer_v1/nccl_recovery/result.json"
LAUNCHER="$ROOT/smoke/launch_foundir_lowlight_haze_restormer_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_lowlight_haze_nccl_recovery.log"
cd "$ROOT"
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly GPUs 0-7" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi required" >&2; exit 1; }
[[ ! -f "$DIAGNOSTIC_RESULT" ]] || { echo "ERROR: immutable NCCL recovery result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "FOUNDIR_LOWLIGHT_HAZE_NCCL_RECOVERY_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "BUDGET diagnostic_wall_max=0.05h diagnostic_GPU_hours_max=0.4; recovered_full_run_wall=1.7-2.7h recovered_full_run_GPU_hours=13.3-21.3"
"$PYTHON" -m py_compile "$DIAGNOSTIC"
bash -n "$LAUNCHER" "$ROOT/smoke/run_foundir_lowlight_haze_restormer_v1_8gpu.sh"
"$PYTHON" "$ROOT/smoke/advisor_governance.py" check
export NCCL_P2P_DISABLE=0 NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 "$DIAGNOSTIC"
"$PYTHON" -c 'import json; from pathlib import Path; p=Path("checkpoints/foundir_lowlight_haze_restormer_v1/nccl_recovery/result.json"); d=json.loads(p.read_text()); assert d["status"] == "pass" and d["optimizer_step_performed"] is False'
echo "FOUNDIR_LOWLIGHT_HAZE_NCCL_RECOVERY_COMPLETE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) restarting_same_frozen_run=true"
bash "$LAUNCHER"
