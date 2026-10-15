#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_official_sow600_qwen_v1.json"
LOG="$ROOT/logs/foundir_official_sow600_qwen_route_v1.log"
LOCK="$ROOT/logs/foundir_official_sow600_qwen_route_v1.lock"
DECISIONS="$ROOT/checkpoints/foundir_official_sow600_qwen_v1/router_decisions.csv"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2
  exit 1
}
[[ -x "$PYTHON" ]] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/checkpoints/foundir_official_sow600_qwen_v1"
exec 9>>"$LOCK"
flock -n 9 || { echo "ERROR: official SOW600 Qwen route stage is already running" >&2; exit 1; }
[[ ! -f "$DECISIONS" ]] || { echo "ERROR: frozen official router decisions already exist" >&2; exit 1; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" -eq 8 ]] || { echo "ERROR: exactly eight visible GPUs are required" >&2; exit 1; }
first_name=${gpu_names[0]}
for name in "${gpu_names[@]}"; do
  [[ "$name" == "$first_name" ]] || { echo "ERROR: mixed GPU models are forbidden" >&2; exit 1; }
  [[ "$name" == *H100* || "$name" == *H200* ]] || {
    echo "ERROR: official Qwen route stage requires homogeneous H100/H200 GPUs: $name" >&2
    exit 1
  }
done

exec > >(tee -a "$LOG") 2>&1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
echo "FOUNDIR_OFFICIAL_SOW600_QWEN_ROUTE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SCOPE official_FoundIR_five_SOW_categories_600_pairs real_Qwen no_category_oracle no_restoration_yet"
"$PYTHON" smoke/export_foundir_official_sow600_qwen_features_ddp.py --config "$CONFIG" --check-only
"$PYTHON" smoke/route_foundir_official_sow600_qwen.py --config "$CONFIG" --check-only
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
  smoke/export_foundir_official_sow600_qwen_features_ddp.py --config "$CONFIG"
"$PYTHON" smoke/route_foundir_official_sow600_qwen.py --config "$CONFIG"
[[ -f "$DECISIONS" ]] || { echo "ERROR: official router decisions missing" >&2; exit 1; }
echo "FOUNDIR_OFFICIAL_SOW600_QWEN_ROUTE_COMPLETE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
