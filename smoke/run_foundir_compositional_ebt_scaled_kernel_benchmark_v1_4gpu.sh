#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/zhenhaowang
PY=/inspire/hdd3/project/agent-driven-world-model/liuxiaohong-25080/.envs/gpu-train-stable-hdd3/bin/python
RUN_ID=${EBT_SCALED_BENCHMARK_RUN_ID:-h200_benchmark_v1}
OUT_DIR="$ROOT/checkpoints/foundir_compositional_ebt_scaled_kernel_benchmark_v1/$RUN_ID"
MAIN_LOG="$ROOT/logs/foundir_compositional_ebt_scaled_kernel_benchmark_v1_${RUN_ID}.log"
CONFIG=configs/foundir_compositional_ebt_scaled_kernel_benchmark_v1.json
SCRIPT=smoke/benchmark_foundir_compositional_ebt_scaled_kernel_v1.py

if [[ -z "$RUN_ID" || "$RUN_ID" == *[!a-zA-Z0-9_.-]* ]]; then
  echo "invalid EBT_SCALED_BENCHMARK_RUN_ID: $RUN_ID" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "missing project interpreter: $PY" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES must name exactly four distinct GPUs" >&2
  exit 2
fi
if [[ $(printf '%s\n' "${GPU_IDS[@]}" | sort -u | wc -l) -ne 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES contains duplicate GPU IDs" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" || -e "$MAIN_LOG" ]]; then
  echo "refusing duplicate benchmark run id: $RUN_ID" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$ROOT/logs"
cd "$ROOT"
exec > >(tee "$MAIN_LOG") 2>&1
echo "FOUNDIR_EBT_SCALED_KERNEL_BENCHMARK_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "workspace=$ROOT"
echo "python=$PY"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PRELAUNCH_ESTIMATE wall_time_hours=0.08_to_0.75 gpu_hours=0.32_to_3.00 allocated_gpus=4"
sha256sum "$CONFIG" "$SCRIPT"

"$PY" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise RuntimeError("expected exactly four visible CUDA devices")
names = [torch.cuda.get_device_name(i) for i in range(4)]
families = ["H200" if "H200" in name else "H100" if "H100" in name else "unsupported" for name in names]
if len(set(families)) != 1 or families[0] == "unsupported":
    raise RuntimeError(f"requires four homogeneous NVIDIA H100/H200 GPUs, got {names}")
print({"gpus": names, "memory_bytes": [torch.cuda.get_device_properties(i).total_memory for i in range(4)]})
PY

profiles=(
  xxs__official_s1_one_step
  xxs__project_three_step
  smallish__official_s1_one_step
  smallish__project_three_step
)
pids=()
set +e
for index in 0 1 2 3; do
  profile=${profiles[$index]}
  profile_log="$ROOT/logs/foundir_compositional_ebt_scaled_kernel_benchmark_v1_${RUN_ID}_${profile}.log"
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$index]}" "$PY" -u "$SCRIPT" \
    --profile "$profile" --output "$OUT_DIR/$profile.json" >"$profile_log" 2>&1 &
  pids+=("$!")
  echo "profile_started profile=$profile physical_gpu=${GPU_IDS[$index]} pid=${pids[$index]} log=$profile_log"
done
status=0
for index in 0 1 2 3; do
  wait "${pids[$index]}"
  worker_status=$?
  echo "profile_finished profile=${profiles[$index]} status=$worker_status"
  tail -n 12 "$ROOT/logs/foundir_compositional_ebt_scaled_kernel_benchmark_v1_${RUN_ID}_${profiles[$index]}.log" || true
  if [[ $worker_status -ne 0 && $status -eq 0 ]]; then status=$worker_status; fi
done
set -e
if [[ $status -ne 0 ]]; then
  echo "FOUNDIR_EBT_SCALED_KERNEL_BENCHMARK_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit "$status"
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$PY" -u "$SCRIPT" --finalize --run-id "$RUN_ID"
echo "FOUNDIR_EBT_SCALED_KERNEL_BENCHMARK_END status=0 utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) result=$OUT_DIR/result.json"
