#!/usr/bin/env bash
set -uo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
cd "$ROOT"

echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_RUN_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_GIT_HEAD $(git rev-parse HEAD)"
echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-unset}"

set +e
bash "$ROOT/smoke/run_foundir_lowlight_gate2_gpu_resume.sh"
status=$?
set -e
if [[ "$status" -eq 0 ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_TERMINAL_PASS"
else
  echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_TERMINAL_FAIL status=$status"
fi
echo "FOUNDIR_LOWLIGHT_GATE2_RESUME_RUN_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
