#!/usr/bin/env bash
set -euo pipefail
ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
cd "$ROOT"
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
CONFIG=configs/foundir_blur_only_zimage_endpoint_distill_v1.json
LOG=logs/foundir_blur_only_zimage_endpoint_distill_v1.log
GPU_LOG=logs/foundir_blur_only_zimage_endpoint_distill_v1_gpu.csv
/usr/bin/python3 smoke/advisor_governance.py check
mkdir -p logs
exec 9>>logs/foundir_blur_only_zimage_endpoint_distill_v1.lock
flock -n 9 || { echo 'ERROR: endpoint job already running'; exit 1; }
exec > >(tee -a "$LOG") 2>&1
telemetry_pid=
finish() {
  status=$?
  if [[ -n "$telemetry_pid" ]]; then kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; fi
  if [[ "$status" -ne 0 && ! -f checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/result.json ]]; then
    /usr/bin/python3 smoke/advisor_governance.py runtime-failure --gate zimage_endpoint_student_runtime_failure --evidence "$CONFIG" --config "$CONFIG" --model 'Z-Image Base 6B endpoint student' --status "$status" --log "$LOG" --diagnostic-only || true
  fi
  echo "ENDPOINT_END status=$status utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap finish EXIT
[[ "${CUDA_VISIBLE_DEVICES:-}" == '0,1,2,3,4,5,6,7' ]]
export PYTHONPATH="$ROOT/.deps/zimage:$ROOT/third_party/diffsynth-studio:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=True
export TORCH_HOME="$ROOT/models/torch" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "ENDPOINT_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) workspace=$ROOT config=$CONFIG git=$(git rev-parse HEAD) devices=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "$GPU_LOG" 2>&1 &
telemetry_pid=$!
"$PYTHON" smoke/preflight_foundir_blur_only_zimage_vae_ceiling_solver_sweep.py --config configs/foundir_blur_only_zimage_vae_ceiling_v1.json --require-cuda
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 smoke/train_foundir_blur_only_zimage_endpoint_distill_ddp.py --config "$CONFIG"
/usr/bin/python3 smoke/advisor_governance.py emit --gate zimage_endpoint_student_v1 --evidence checkpoints/foundir_blur_only_zimage_endpoint_distill_v1/result.json --config "$CONFIG" --model 'Z-Image Base 6B GT-anchored single-step student FP32 LoRA frozen VAE' --split fixed_16_training_pairs_plus_16_source_disjoint_visual_development_pairs --outcome auto --diagnostic-only --root-cause-or-status 'Bounded endpoint student finished; inspect fixed visuals and both metrics before scaling. Teacher acceptance filter uses only training pairs.' --advisor-decision-question 'Continue within existing user authorization; no new advisor round requested.'
