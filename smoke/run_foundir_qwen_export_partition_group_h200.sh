#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
/usr/bin/python3 "$ROOT/smoke/advisor_governance.py" check
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v3.json"
MANIFEST="$ROOT/manifests/foundir_five_type_pairs_v1.csv"
GROUP="${1:-}"
PARTITIONS=8
WORKERS_PER_INSTANCE=4
LOCK="$ROOT/logs/foundir_qwen_export_partition_group_${GROUP}.lock"

cd "$ROOT"
[[ "$GROUP" == "0" || "$GROUP" == "1" ]] || {
  echo "ERROR: partition group must be exactly 0 or 1" >&2
  exit 1
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2
  exit 1
}
[[ -x "$PYTHON" ]] || { echo "ERROR: Qwen Python is missing" >&2; exit 1; }
mkdir -p "$ROOT/logs"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: partition group $GROUP is already active" >&2; exit 1; }

mapfile -t inventory < <(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
[[ "${#inventory[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPUs are required" >&2; exit 1; }
for item in "${inventory[@]}"; do
  [[ "$item" == NVIDIA\ H200,* ]] || {
    echo "ERROR: four concurrent workers per instance require homogeneous H200 GPUs: $item" >&2
    exit 1
  }
  memory_mib=${item##*, }
  [[ "$memory_mib" =~ ^[0-9]+$ && "$memory_mib" -ge 140000 ]] || {
    echo "ERROR: H200 memory inventory is below 140000 MiB: $item" >&2
    exit 1
  }
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=()
logs=()
terminate_children() {
  for child in "${pids[@]:-}"; do
    kill -TERM "$child" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 143
}
trap terminate_children INT TERM

for slot in 0 1 2 3; do
  partition=$((GROUP * WORKERS_PER_INSTANCE + slot))
  log="$ROOT/logs/foundir_qwen_export_partition_${partition}_of_${PARTITIONS}.log"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    smoke/export_foundir_five_type_physical_v8_conditions_ddp.py \
    --config "$CONFIG" --manifest "$MANIFEST" \
    --partitions "$PARTITIONS" --partition-index "$partition" \
    >>"$log" 2>&1 &
  pids+=("$!")
  logs+=("$log")
  echo "FOUNDIR_QWEN_PARTITION_STARTED group=$GROUP partition=$partition pid=$! log=$log"
  if [[ "$slot" -lt 3 ]]; then sleep 10; fi
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "ERROR: partition worker failed: ${logs[$index]}" >&2
    status=1
  fi
done
[[ "$status" -eq 0 ]] || exit "$status"
for log in "${logs[@]}"; do
  rg -q 'FOUNDIR_PHYSICAL_V8_PARTITION_PASS' "$log" || {
    echo "ERROR: partition log has no verified pass marker: $log" >&2
    exit 1
  }
done
echo "FOUNDIR_QWEN_PARTITION_GROUP_PASS group=$GROUP partitions=$((GROUP * 4))-$((GROUP * 4 + 3))"
