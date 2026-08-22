#!/usr/bin/env bash
set -uo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
cd "$ROOT"

echo "FOUNDIR_LOWLIGHT_GATE2_RUN_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_LOWLIGHT_GATE2_WORKSPACE $ROOT"
echo "FOUNDIR_LOWLIGHT_GATE2_GIT_HEAD $(git rev-parse HEAD)"
echo "FOUNDIR_LOWLIGHT_GATE2_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES:-unset}"
sha256sum \
  configs/foundir_lowlight_gate2_teacher_v1b.json \
  configs/foundir_lowlight_gate2_historical_flux_v1b.json \
  configs/foundir_lowlight_gate2_metric_blend_v1b.json

set +e
bash "$ROOT/smoke/run_foundir_lowlight_gate2_gpu_bundle.sh"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_TERMINAL_PASS"
else
  echo "FOUNDIR_LOWLIGHT_GATE2_TERMINAL_FAIL status=$status"
fi
echo "FOUNDIR_LOWLIGHT_GATE2_RUN_END utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
exit "$status"
