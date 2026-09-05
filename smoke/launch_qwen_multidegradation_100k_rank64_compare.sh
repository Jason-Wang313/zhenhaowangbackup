#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
RUNNER="$ROOT/smoke/run_qwen_multidegradation_100k_rank64_method.sh"

cd "$ROOT" || exit 1
mkdir -p "$ROOT/logs"

visible=${CUDA_VISIBLE_DEVICES:-}
version=${QWEN_100K_VERSION:-v1}
case "$version" in
  v1)
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_compare_launcher_v1.log"
    joint_config=configs/qwen_multidegradation_100k_rank64_joint_v1.json
    per_type_config=configs/qwen_multidegradation_100k_rank64_per_type_v1.json
    joint_result=checkpoints/qwen_multidegradation_100k_rank64_joint_v1/merger_trainable/validation_result.json
    per_type_result=checkpoints/qwen_multidegradation_100k_rank64_per_type_v1/merger_trainable/validation_result.json
    comparison_output=reports/qwen_multidegradation_100k_rank64_comparison_v1.json
    ;;
  v2)
    LOG="$ROOT/logs/qwen_multidegradation_100k_rank64_compare_launcher_v2.log"
    joint_config=configs/qwen_multidegradation_100k_rank64_joint_v2.json
    per_type_config=configs/qwen_multidegradation_100k_rank64_per_type_v2.json
    joint_result=checkpoints/qwen_multidegradation_100k_rank64_joint_v2/merger_trainable/validation_result.json
    per_type_result=checkpoints/qwen_multidegradation_100k_rank64_per_type_v2/merger_trainable/validation_result.json
    comparison_output=reports/qwen_multidegradation_100k_rank64_comparison_v2.json
    ;;
  *) echo "ERROR: QWEN_100K_VERSION must be v1 or v2" >&2; exit 1 ;;
esac
exec > >(tee -a "$LOG") 2>&1
IFS=, read -r -a devices <<<"$visible"
count=${#devices[@]}
schedule=${QWEN_COMPARE_SCHEDULING:-sequential}
case "$schedule" in
  sequential|concurrent) ;;
  *) echo "ERROR: QWEN_COMPARE_SCHEDULING must be sequential or concurrent" >&2; exit 1 ;;
esac
case "$count" in
  2)
    if [[ "$schedule" == concurrent ]]; then
      joint_visible=${devices[0]}
      per_type_visible=${devices[1]}
    else
      joint_visible=$visible
      per_type_visible=$visible
    fi
    ;;
  8)
    if [[ "$schedule" == concurrent ]]; then
      joint_visible=$(IFS=,; echo "${devices[*]:0:4}")
      per_type_visible=$(IFS=,; echo "${devices[*]:4:4}")
    else
      joint_visible=$visible
      per_type_visible=$visible
    fi
    ;;
  *)
    echo "ERROR: comparison launcher requires exactly 2 or 8 visible GPUs" >&2
    exit 1
    ;;
esac
if [[ $(printf '%s\n' "${devices[@]}" | sort -u | wc -l) -ne $count ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES contains duplicates" >&2
  exit 1
fi

echo "QWEN_100K_RANK64_COMPARE_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "QWEN_100K_RANK64_COMPARE_VERSION $version"
echo "QWEN_100K_RANK64_COMPARE_SCHEDULING $schedule"
echo "QWEN_100K_RANK64_COMPARE_ASSIGN joint=$joint_visible per_type=$per_type_visible"
echo "QWEN_100K_RANK64_COMPARE_POLICY validation_only_no_protected_test"

# One full hash audit covers the shared 120k image corpus; the second preflight
# still checks its distinct prompt/assistant-token construction.
"$PYTHON" smoke/qwen_multidegradation_lora_preflight.py \
  --config "$joint_config" --full-hashes
"$PYTHON" smoke/qwen_multidegradation_lora_preflight.py \
  --config "$per_type_config"

if [[ "$schedule" == concurrent ]]; then
  env CUDA_VISIBLE_DEVICES="$joint_visible" QWEN_100K_VERSION="$version" QWEN_100K_METHOD=joint bash "$RUNNER" &
  joint_pid=$!
  env CUDA_VISIBLE_DEVICES="$per_type_visible" QWEN_100K_VERSION="$version" QWEN_100K_METHOD=per_type bash "$RUNNER" &
  per_type_pid=$!
  set +e
  wait "$joint_pid"
  joint_status=$?
  wait "$per_type_pid"
  per_type_status=$?
  set -e
else
  set +e
  env CUDA_VISIBLE_DEVICES="$joint_visible" QWEN_100K_VERSION="$version" QWEN_100K_METHOD=joint bash "$RUNNER"
  joint_status=$?
  env CUDA_VISIBLE_DEVICES="$per_type_visible" QWEN_100K_VERSION="$version" QWEN_100K_METHOD=per_type bash "$RUNNER"
  per_type_status=$?
  set -e
fi

echo "QWEN_100K_RANK64_COMPARE_METHOD_STATUS joint=$joint_status per_type=$per_type_status"
if [[ -f "$joint_result" && -f "$per_type_result" ]]; then
  "$PYTHON" smoke/compare_qwen_multidegradation_100k_rank64.py \
    --joint-result "$joint_result" \
    --per-type-result "$per_type_result" \
    --output "$comparison_output"
  comparison_status=$?
else
  echo "QWEN_100K_RANK64_COMPARISON_INCOMPLETE" >&2
  exit 1
fi
echo "QWEN_100K_RANK64_PROTECTED_TEST_NOT_ACCESSED"
if [[ $comparison_status -ne 0 ]]; then
  exit 1
fi
if [[ $joint_status -ne 0 || $per_type_status -ne 0 ]]; then
  exit 2
fi
echo "QWEN_100K_RANK64_COMPARE_PASS"
