#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
CONFIG="$ROOT/configs/foundir_five_type_flux2_dit_v3.json"
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
LOCK="$ROOT/logs/foundir_flux2_speed_benchmark.lock"
LOG="$ROOT/logs/foundir_flux2_speed_benchmark.log"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/speed_benchmarks"
exec > >(tee -a "$LOG") 2>&1

telemetry_pid=
finish_run() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then
    kill "$telemetry_pid" 2>/dev/null || true
    wait "$telemetry_pid" 2>/dev/null || true
  fi
  echo "FOUNDIR_FLUX_SPEED_BENCHMARK_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish_run EXIT

exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: another FLUX speed benchmark holds $LOCK" >&2; exit 1; }
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" >&2; exit 1;
}
[[ -x "$PYTHON" ]] || { echo "ERROR: workspace FLUX Python is missing" >&2; exit 1; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ ${#gpu_names[@]} -eq 8 ]] || { echo "ERROR: benchmark requires exactly eight GPUs" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: GPUs are not homogeneous" >&2; exit 1; }
done
case "$first_name" in
  *H200*) family=H200; baseline_micro=2; baseline_accum=2; candidate_micro=4; candidate_accum=1 ;;
  *H100*) family=H100; baseline_micro=1; baseline_accum=4; candidate_micro=2; candidate_accum=2 ;;
  *) echo "ERROR: only homogeneous H100/H200 is supported: $first_name" >&2; exit 1 ;;
esac

run_id=$(date -u +%Y%m%dT%H%M%SZ)_${family}
output="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/speed_benchmarks/$run_id"
mkdir "$output"
gpu_log="$ROOT/logs/foundir_flux2_speed_benchmark_${run_id}_gpu.csv"

export TORCH_HOME="$ROOT/models/torch"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DESYNC_DEBUG=1

echo "FOUNDIR_FLUX_SPEED_BENCHMARK_START run_id=$run_id family=$family utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "FOUNDIR_FLUX_SPEED_BENCHMARK_SCOPE training_only no_checkpoint no_official_test"
"$PYTHON" smoke/preflight_foundir_flux2_training_geometry.py --config "$CONFIG"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv -l 2 >"$gpu_log" 2>&1 &
telemetry_pid=$!

run_case() {
  name=$1
  micro=$2
  accumulation=$3
  workers=$4
  echo "FOUNDIR_FLUX_SPEED_CASE_START name=$name micro=$micro accumulation=$accumulation workers=$workers"
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    smoke/train_foundir_five_type_flux2_full_ddp.py \
    --config "$CONFIG" \
    --benchmark-name "$name" \
    --benchmark-output "$output/$name.json" \
    --benchmark-steps 20 \
    --benchmark-warmup-steps 5 \
    --micro-batch-size-per-rank "$micro" \
    --gradient-accumulation-steps "$accumulation" \
    --data-loader-workers "$workers"
  echo "FOUNDIR_FLUX_SPEED_CASE_END name=$name status=0"
}

run_case baseline_${baseline_micro}x${baseline_accum}_w4 "$baseline_micro" "$baseline_accum" 4
run_case candidate_${candidate_micro}x${candidate_accum}_w4 "$candidate_micro" "$candidate_accum" 4
run_case candidate_${candidate_micro}x${candidate_accum}_w8 "$candidate_micro" "$candidate_accum" 8

selection="$ROOT/checkpoints/foundir_five_type_flux2_dit_v3/speed_benchmark_selection_${family}.json"
[[ ! -e "$selection" ]] || {
  echo "ERROR: refusing to overwrite prior $family speed selection: $selection" >&2; exit 1;
}
"$PYTHON" smoke/select_foundir_flux2_speed_benchmark.py \
  --result "$output/baseline_${baseline_micro}x${baseline_accum}_w4.json" \
  --result "$output/candidate_${candidate_micro}x${candidate_accum}_w4.json" \
  --result "$output/candidate_${candidate_micro}x${candidate_accum}_w8.json" \
  --output "$selection"
echo "FOUNDIR_FLUX_SPEED_BENCHMARK_PASS selection=$selection gpu_log=$gpu_log"
