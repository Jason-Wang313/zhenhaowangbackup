#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
HISTORICAL_RESULT="$ROOT/reports/foundir_lowlight_gate2_historical_flux_result_v1b.json"
cd "$ROOT"

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo "ERROR: expose one or two GPUs to the Gate 2 resume" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
teacher_device="${devices[0]}"

export PYTHONPATH="$ROOT/.tools/foundir-runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/.tools/foundir-transformers-4.47.1-v2:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-huggingface-hub-0.27.1:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-tokenizers-0.21.0:$PYTHONPATH"

echo "FOUNDIR_LOWLIGHT_GATE2_SPECIALIST_RESUME_START device=$teacher_device"
CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.conda/gpu-train-stable/bin/python" -u \
  -m smoke.run_foundir_lowlight_gate2_teachers --stage specialist-resume

waited=0
while [[ ! -f "$HISTORICAL_RESULT" && "$waited" -lt 1800 ]]; do
  if (( waited % 60 == 0 )); then
    echo "FOUNDIR_LOWLIGHT_GATE2_WAITING_FOR_HISTORICAL seconds=$waited"
  fi
  sleep 10
  waited=$((waited + 10))
done
if [[ ! -f "$HISTORICAL_RESULT" ]]; then
  echo "ERROR: historical FLUX result did not complete within 1800 seconds" >&2
  exit 1
fi

echo "FOUNDIR_LOWLIGHT_GATE2_METRICS_RESUME_START device=$teacher_device waited_for_historical=$waited"
CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.venv/foundir-metrics/bin/python" -u \
  -m smoke.run_foundir_lowlight_gate2_metric_blend
echo "FOUNDIR_LOWLIGHT_GATE2_GPU_RESUME_PASS"
