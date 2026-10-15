#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_composition_integrated_restormer_v2.json"
EVALUATOR="$ROOT/smoke/evaluate_foundir_composition_integrated_restormer_v2.py"
OUTPUT="$ROOT/checkpoints/foundir_composition_integrated_restormer_v2"
LOG="$ROOT/logs/foundir_composition_integrated_restormer_v2.log"
LOCK="$ROOT/logs/foundir_composition_integrated_restormer_v2.lock"
cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUTPUT"
exec > >(tee -a "$LOG") 2>&1
finish_run(){ status=$?; echo "FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; return "$status"; }
trap finish_run EXIT
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: integrated composition replay already running" >&2; exit 1; }
[[ ! -f "$OUTPUT/result.json" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
echo "FOUNDIR_COMPOSITION_INTEGRATED_RESTORMER_V2_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE DIAGNOSTIC_ONLY route_aware immutable_artifact_replay no_training no_fresh_inference no_official_test"
"$PYTHON" "$EVALUATOR" --config "$CONFIG" --preflight
"$PYTHON" "$EVALUATOR" --config "$CONFIG"
"$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("checkpoints/foundir_composition_integrated_restormer_v2/result.json")
d = json.loads(p.read_text())
if d.get("status") != "foundir_composition_integrated_restormer_v2_complete" or d.get("records") != 660:
    raise SystemExit("terminal integrated composition result is invalid")
print(json.dumps(d["overall_metrics"], indent=2, sort_keys=True))
PY
