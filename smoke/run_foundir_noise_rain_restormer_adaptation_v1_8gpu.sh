#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
TRAINER="$ROOT/smoke/train_foundir_task_native_restormer_adaptation_ddp.py"
NOISE_CONFIG="$ROOT/configs/foundir_noise_restormer_adaptation_v1.json"
RAIN_CONFIG="$ROOT/configs/foundir_rain_restormer_adaptation_v1.json"
LOG="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1.log"
NOISE_LOG="$ROOT/logs/foundir_noise_restormer_adaptation_v1.log"
RAIN_LOG="$ROOT/logs/foundir_rain_restormer_adaptation_v1.log"
GPU_LOG="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1_gpu.csv"
LOCK="$ROOT/logs/foundir_noise_rain_restormer_adaptation_v1.lock"

cd "$ROOT"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

noise_pid=
rain_pid=
telemetry_pid=
stop_child_group() {
  local child_pid="$1"
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -- -"$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
finish_run() {
  local status=$?
  trap - EXIT TERM INT HUP
  stop_child_group "$noise_pid"
  stop_child_group "$rain_pid"
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
}
on_signal() {
  exit 143
}
trap finish_run EXIT
trap on_signal TERM INT HUP

exec 9>>"$LOCK"
flock -n 9 || {
  echo "ERROR: another Noise/Rain Restormer adaptation holds $LOCK" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: expose exactly CUDA devices 0,1,2,3,4,5,6,7" >&2
  exit 1
}

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DESYNC_DEBUG=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_WORKSPACE $ROOT"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_GIT $(git rev-parse HEAD)"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_DEVICES $CUDA_VISIBLE_DEVICES"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_TRAINER_SHA256 $(sha256sum "$TRAINER" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RESTORMER_ADAPTATION_V1_CONFIG_SHA256 $(sha256sum "$NOISE_CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V1_CONFIG_SHA256 $(sha256sum "$RAIN_CONFIG" | cut -d' ' -f1)"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_SCOPE isolated_development_capacity_only"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_INTERNAL sealed_internal_not_accessed"
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_PROTECTED_OFFICIAL_TEST not_accessed"

{
  echo "RUN_START,$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 5
} >>"$GPU_LOG" 2>&1 &
telemetry_pid=$!

{
  echo "FOUNDIR_NOISE_RESTORMER_ADAPTATION_V1_JOB_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) devices=0,1,2,3"
  echo "config_sha256=$(sha256sum "$NOISE_CONFIG" | cut -d' ' -f1)"
} >>"$NOISE_LOG"
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 "$TRAINER" --config "$NOISE_CONFIG" \
  >>"$NOISE_LOG" 2>&1 &
noise_pid=$!

{
  echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V1_JOB_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) devices=4,5,6,7"
  echo "config_sha256=$(sha256sum "$RAIN_CONFIG" | cut -d' ' -f1)"
} >>"$RAIN_LOG"
setsid env CUDA_VISIBLE_DEVICES=4,5,6,7 "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=4 "$TRAINER" --config "$RAIN_CONFIG" \
  >>"$RAIN_LOG" 2>&1 &
rain_pid=$!

echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_CHILDREN noise_process_group=$noise_pid rain_process_group=$rain_pid"

set +e
completed_pid=
wait -n -p completed_pid "$noise_pid" "$rain_pid"
first_status=$?
set -e

if [[ "$completed_pid" == "$noise_pid" ]]; then
  noise_status=$first_status
  echo "FOUNDIR_NOISE_RESTORMER_ADAPTATION_V1_JOB_END status=$noise_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$NOISE_LOG"
  if [[ "$noise_status" -ne 0 ]]; then
    stop_child_group "$rain_pid"
    rain_pid=
    exit "$noise_status"
  fi
  set +e
  wait "$rain_pid"
  rain_status=$?
  set -e
  rain_pid=
  echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V1_JOB_END status=$rain_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RAIN_LOG"
  [[ "$rain_status" -eq 0 ]] || exit "$rain_status"
else
  rain_status=$first_status
  echo "FOUNDIR_RAIN_RESTORMER_ADAPTATION_V1_JOB_END status=$rain_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RAIN_LOG"
  if [[ "$rain_status" -ne 0 ]]; then
    stop_child_group "$noise_pid"
    noise_pid=
    exit "$rain_status"
  fi
  set +e
  wait "$noise_pid"
  noise_status=$?
  set -e
  noise_pid=
  echo "FOUNDIR_NOISE_RESTORMER_ADAPTATION_V1_JOB_END status=$noise_status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$NOISE_LOG"
  [[ "$noise_status" -eq 0 ]] || exit "$noise_status"
fi

noise_pid=
rain_pid=
"$PYTHON" - "$ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for task in ("noise", "rain"):
    path = root / "checkpoints/foundir_five_type_flux2_dit_v3" / f"{task}_restormer_adaptation_v1" / "development_selection.json"
    result = json.loads(path.read_text())
    if (
        result.get("status") != "foundir_task_native_adaptation_development_selection"
        or result.get("task") != task
        or result.get("sealed_internal_test_accessed") is not False
        or result.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid completed {task} adaptation selection")
    print(
        "FOUNDIR_TASK_NATIVE_ADAPTATION_SELECTION "
        + json.dumps(
            {
                "task": task,
                "selected_step": result["selected_step"],
                "metrics": result["metrics"],
                "source_group_bootstrap": result["source_group_bootstrap"],
                "strong_gate_pass": result["strong_gate_pass"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
PY
echo "FOUNDIR_NOISE_RAIN_RESTORMER_ADAPTATION_V1_COMPLETE"
