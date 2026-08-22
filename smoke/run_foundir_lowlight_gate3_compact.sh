#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/foundir_lowlight_gate3_v1.json"
CONFIG_SHA256="818bd402ad5aec6b12e826cc30c99b49bead9c30a2a85a7ff63c4eb744434ca3"
TRAIN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
METRIC_PYTHON="$ROOT/.venv/foundir-metrics/bin/python"
TRAINER="$ROOT/smoke/train_foundir_lowlight_gate3_compact.py"
EVALUATOR="$ROOT/smoke/evaluate_foundir_lowlight_gate3_compact.py"
RESULT="$ROOT/reports/foundir_lowlight_gate3_compact_result_v1.json"
LOCK="$ROOT/logs/foundir_lowlight_gate3_compact_v1.lock"
DEVICE_SET="${CUDA_VISIBLE_DEVICES:-}"

REQUIRED_TRACKED_FILES=(
  configs/foundir_lowlight_gate3_v1.json
  lowlight_restoration/compact_residual.py
  smoke/train_foundir_lowlight_gate3_compact.py
  smoke/evaluate_foundir_lowlight_gate3_compact.py
  smoke/run_foundir_lowlight_gate3_compact.sh
  tests/test_foundir_lowlight_gate3_compact.py
)

static_check() {
  if [[ "$DEVICE_SET" != "2,3" ]]; then
    echo "ERROR: compact Gate 3 requires CUDA_VISIBLE_DEVICES exactly 2,3" >&2
    return 1
  fi
  if [[ ! -x "$TRAIN_PYTHON" || ! -x "$METRIC_PYTHON" ]]; then
    echo "ERROR: a frozen Gate 3 Python environment is unavailable" >&2
    return 1
  fi
  if [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA256" ]]; then
    echo "ERROR: frozen Gate 3 config hash changed" >&2
    return 1
  fi
  bash -n "$ROOT/smoke/run_foundir_lowlight_gate3_compact.sh"
  "$TRAIN_PYTHON" -m py_compile "$TRAINER" "$EVALUATOR" "$ROOT/lowlight_restoration/compact_residual.py"
  "$TRAIN_PYTHON" -m unittest tests.test_foundir_lowlight_gate3_compact
  "$TRAIN_PYTHON" "$TRAINER" --config "$CONFIG" --contract-only
  for relative_path in "${REQUIRED_TRACKED_FILES[@]}"; do
    git -C "$ROOT" ls-files --error-unmatch "$relative_path" >/dev/null 2>&1 || {
      echo "ERROR: required Gate 3 file is not tracked: $relative_path" >&2
      return 1
    }
  done
  if ! git -C "$ROOT" diff --quiet HEAD -- "${REQUIRED_TRACKED_FILES[@]}"; then
    echo "ERROR: required Gate 3 files differ from the committed revision" >&2
    return 1
  fi
}

if [[ "${1:-}" == "--check" ]]; then
  if [[ "$#" -ne 1 ]]; then
    echo "ERROR: --check takes no additional arguments" >&2
    exit 1
  fi
  static_check
  echo "FOUNDIR_LOWLIGHT_GATE3_COMPACT_RUNNER_CHECK_PASS devices=$DEVICE_SET"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "ERROR: unexpected compact Gate 3 runner argument: $1" >&2
  exit 1
fi

static_check
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
if ! flock -n 9; then
  echo "ERROR: compact Gate 3 is already active" >&2
  exit 1
fi

mapfile -t gpu_inventory < <(
  nvidia-smi --id="$DEVICE_SET" --query-gpu=index,name,memory.total --format=csv,noheader,nounits
)
if [[ "${#gpu_inventory[@]}" -ne 2 ]]; then
  echo "ERROR: compact Gate 3 expected two selected GPUs" >&2
  exit 1
fi
for row in "${gpu_inventory[@]}"; do
  if [[ ! "$row" =~ ^[[:space:]]*[0-9]+,[[:space:]]*NVIDIA[[:space:]]+H(100|200)[^,]*,[[:space:]]*([0-9]+) ]] || [[ "${BASH_REMATCH[2]}" -lt 76800 ]]; then
    echo "ERROR: compact Gate 3 requires H100/H200 GPUs with at least 76800 MiB: $row" >&2
    exit 1
  fi
done

if [[ -f "$RESULT" ]]; then
  "$TRAIN_PYTHON" - "$RESULT" "$CONFIG_SHA256" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "pass_gate3_compact_evaluation_complete" or payload.get("config_sha256") != sys.argv[2]:
    raise SystemExit("ERROR: existing Gate 3 result is inconsistent")
print("FOUNDIR_LOWLIGHT_GATE3_COMPACT_ALREADY_COMPLETE")
PY
  exit 0
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_LOWLIGHT_GATE3_COMPACT_BEGIN utc=$started_utc git_revision=$(git -C "$ROOT" rev-parse HEAD) devices=$DEVICE_SET"
(
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES="$DEVICE_SET" "$TRAIN_PYTHON" -m torch.distributed.run \
    --standalone --nproc-per-node=2 --max-restarts=0 \
    "$TRAINER" --config "$CONFIG"
)
CUDA_VISIBLE_DEVICES=2 "$METRIC_PYTHON" "$EVALUATOR" --config "$CONFIG"
ended_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_LOWLIGHT_GATE3_COMPACT_END utc=$ended_utc status=0"
