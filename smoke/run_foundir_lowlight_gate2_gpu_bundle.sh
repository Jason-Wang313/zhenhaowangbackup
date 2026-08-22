#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
cd "$ROOT"

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo "ERROR: expose one or two GPUs to the Gate 2 bundle" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#devices[@]}" -eq 2 && "${devices[0]}" == "${devices[1]}" ]]; then
  echo "ERROR: the two GPU indices must be distinct" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/.tools/foundir-runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/.tools/foundir-transformers-4.47.1-v2:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-huggingface-hub-0.27.1:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-tokenizers-0.21.0:$PYTHONPATH"

teacher_device="${devices[0]}"
flux_device="${devices[1]:-${devices[0]}}"

if [[ "${#devices[@]}" -eq 2 ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_PARALLEL_PROPOSALS_START teacher_gpu=$teacher_device flux_gpu=$flux_device"
  set +e
  CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.conda/gpu-train-stable/bin/python" -u -m smoke.run_foundir_lowlight_gate2_teachers &
  teacher_pid=$!
  CUDA_VISIBLE_DEVICES="$flux_device" "$ROOT/.venv/flux2-smoke/bin/python" -u -m smoke.run_foundir_lowlight_gate2_historical_flux &
  flux_pid=$!
  wait "$teacher_pid"
  teacher_status=$?
  wait "$flux_pid"
  flux_status=$?
  set -e
  echo "FOUNDIR_LOWLIGHT_GATE2_PARALLEL_PROPOSALS_END teacher_status=$teacher_status flux_status=$flux_status"
  if [[ "$teacher_status" -ne 0 || "$flux_status" -ne 0 ]]; then
    exit 1
  fi
else
  CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.conda/gpu-train-stable/bin/python" -u -m smoke.run_foundir_lowlight_gate2_teachers
  CUDA_VISIBLE_DEVICES="$flux_device" "$ROOT/.venv/flux2-smoke/bin/python" -u -m smoke.run_foundir_lowlight_gate2_historical_flux
fi

CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.venv/foundir-metrics/bin/python" -u -m smoke.run_foundir_lowlight_gate2_metric_blend

echo "FOUNDIR_LOWLIGHT_GATE2_GPU_BUNDLE_PASS"
