#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
ATTEMPTS="$ROOT/logs/foundir_lowlight_gate2_gpu_v1b_attempts.tsv"
RESULT="$ROOT/reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json"
cd "$ROOT"

if [[ -f "$RESULT" ]]; then
  "$ROOT/.venv/flux2-smoke/bin/python" - <<'PY'
import json
from pathlib import Path
result = json.loads(Path("reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json").read_text())
assert result["status"] == "pass_gate2_complete"
print("FOUNDIR_LOWLIGHT_GATE2_COMPLETE", result["decision"]["provisional_conservative_base"])
PY
  exit 0
fi

last_log=""
if [[ -f "$ATTEMPTS" ]]; then
  while IFS=$'\t' read -r pid log; do
    last_log="$log"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "FOUNDIR_LOWLIGHT_GATE2_RUNNING pid=$pid log=$log"
      tail -n 5 "$log"
      exit 0
    fi
  done < "$ATTEMPTS"
fi

echo "FOUNDIR_LOWLIGHT_GATE2_NOT_RUNNING_OR_FAILED log=$last_log"
if [[ -n "$last_log" && -f "$last_log" ]]; then
  tail -n 30 "$last_log"
fi
exit 1
