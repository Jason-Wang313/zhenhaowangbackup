#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
RUNNER="$ROOT/smoke/run_foundir_five_type_qwen_routed_integrated_v2_internal_8gpu.sh"
LOG="$ROOT/logs/foundir_five_type_qwen_routed_integrated_v2_internal.log"
PID_FILE="$ROOT/logs/foundir_five_type_qwen_routed_integrated_v2_internal.pid"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_five_type_qwen_routed_integrated_v2.json"
cd "$ROOT"
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || { echo "ERROR: only optional --check is supported" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || { echo "ERROR: expose exactly eight GPUs" >&2; exit 1; }
command -v setsid >/dev/null || { echo "ERROR: setsid is required" >&2; exit 1; }
"$PYTHON" -m py_compile \
  smoke/evaluate_foundir_five_type_qwen_routed_integrated_ddp.py \
  smoke/freeze_foundir_five_type_qwen_routed_development.py
bash -n "$RUNNER" "$0"
"$PYTHON" - "$CONFIG" <<'PY'
import pathlib, sys
from smoke.evaluate_foundir_five_type_qwen_routed_integrated_ddp import (
    CONFIDENCE_GUARDED_POLICY,
    validate_router_config,
    verify_development_freeze,
)
path = pathlib.Path(sys.argv[1])
router, parent, specialization, checkpoint = validate_router_config(path)
verify_development_freeze(path, router)
assert router["routing"]["policy"] == CONFIDENCE_GUARDED_POLICY
assert router["protected_foundir_test_accessed"] is False
print("FOUNDIR_QWEN_ROUTED_INTEGRATED_V2_INTERNAL_CONTRACT_CHECK_PASS")
PY
if [[ "${1:-}" == "--check" ]]; then echo "FOUNDIR_QWEN_ROUTED_INTEGRATED_V2_INTERNAL_LAUNCHER_CHECK_PASS"; exit 0; fi
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || { echo "ERROR: integrated v2 internal evaluation exited; inspect $LOG" >&2; tail -100 "$LOG" >&2 || true; exit 1; }
echo "FOUNDIR_QWEN_ROUTED_INTEGRATED_V2_INTERNAL_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Stop safely: kill -- -$pid"
