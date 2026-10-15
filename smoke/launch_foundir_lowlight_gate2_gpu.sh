#!/usr/bin/env bash
set -euo pipefail

ROOT="/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang"
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
ATTEMPTS="$ROOT/logs/foundir_lowlight_gate2_gpu_v1b_attempts.tsv"
RESULT="$ROOT/reports/foundir_lowlight_gate2_proposal_lattice_result_v1b.json"
cd "$ROOT"

if [[ ! "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)?$ ]]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES to one or two GPU indices, for example 0,1" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#devices[@]}" -eq 2 && "${devices[0]}" == "${devices[1]}" ]]; then
  echo "ERROR: the two GPU indices must be distinct" >&2
  exit 2
fi

for executable in \
  "$ROOT/.conda/gpu-train-stable/bin/python" \
  "$ROOT/.venv/flux2-smoke/bin/python" \
  "$ROOT/.venv/foundir-metrics/bin/python"; do
  if [[ ! -x "$executable" ]]; then
    echo "ERROR: missing executable: $executable" >&2
    exit 2
  fi
done

if [[ -f "$RESULT" ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_ALREADY_COMPLETE"
  exit 0
fi

if [[ -f "$ATTEMPTS" ]]; then
  while IFS=$'\t' read -r existing_pid existing_log; do
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
      echo "FOUNDIR_LOWLIGHT_GATE2_ALREADY_RUNNING pid=$existing_pid log=$existing_log"
      exit 0
    fi
  done < "$ATTEMPTS"
fi

export PYTHONPATH="$ROOT/.tools/foundir-runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$ROOT/.tools/foundir-transformers-4.47.1-v2:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-huggingface-hub-0.27.1:$PYTHONPATH"
export PYTHONPATH="$ROOT/.tools/foundir-tokenizers-0.21.0:$PYTHONPATH"

mkdir -p "$ROOT/logs"
log="$(mktemp "$ROOT/logs/foundir_lowlight_gate2_gpu_v1b_XXXXXX.log")"
teacher_device="${devices[0]}"
flux_device="${devices[1]:-${devices[0]}}"

set +e
(
set -euo pipefail
echo "FOUNDIR_LOWLIGHT_GATE2_PREFLIGHT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) devices=$CUDA_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.conda/gpu-train-stable/bin/python" - <<'PY'
import importlib
import torch
import huggingface_hub
import tokenizers
import transformers
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) == (9, 0), (major, minor)
assert transformers.__version__ == "4.47.1", transformers.__version__
assert tokenizers.__version__ == "0.21.0", tokenizers.__version__
assert huggingface_hub.__version__ == "0.27.1", huggingface_hub.__version__
importlib.import_module("ema_pytorch")
importlib.import_module("causal_conv1d")
importlib.import_module("mamba_ssm")
print("GATE2_TEACHER_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY

CUDA_VISIBLE_DEVICES="$flux_device" "$ROOT/.venv/flux2-smoke/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GATE2_FLUX_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY

CUDA_VISIBLE_DEVICES="$teacher_device" "$ROOT/.venv/foundir-metrics/bin/python" - <<'PY'
import lpips
import pyiqa
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
print("GATE2_METRIC_GPU_PREFLIGHT_PASS", torch.cuda.get_device_name(0))
PY
echo "FOUNDIR_LOWLIGHT_GATE2_PREFLIGHT_PASS utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
) 2>&1 | tee -a "$log"
preflight_status=${PIPESTATUS[0]}
set -e
if [[ "$preflight_status" -ne 0 ]]; then
  echo "FOUNDIR_LOWLIGHT_GATE2_PREFLIGHT_FAILED status=$preflight_status log=$log" | tee -a "$log" >&2
  exit "$preflight_status"
fi

nohup bash "$ROOT/smoke/run_foundir_lowlight_gate2_gpu_logged.sh" >> "$log" 2>&1 &
pid=$!
printf '%s\t%s\n' "$pid" "$log" >> "$ATTEMPTS"
echo "FOUNDIR_LOWLIGHT_GATE2_DETACHED pid=$pid log=$log"
