#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
ARCHIVES="$ROOT/data/foundir/_download_stage/five_type_archives_v1"
DOWNLOAD_MARKER="$ROOT/data/foundir/_download_stage/state_v1/download.complete"
MANIFEST="$ROOT/manifests/foundir_five_type_training_archives_baidupan_v1.tsv"
SEVEN="$ROOT/.tools/7zip-23.01-ubuntu/extracted/usr/lib/7zip/7z"
SEVEN_DEB="$ROOT/.tools/7zip-23.01-ubuntu/package/7zip_23.01+dfsg-11_amd64.deb"
SEVEN_DEB_SHA256=0f79450d81e64326a862a8645ae1613890e486115101e51e41ca07d929388d87
TRAIN_ROOT="$ROOT/data/foundir/MillionIRData/Train"
ATTEMPTS="$ROOT/data/foundir/_extract_stage/five_type_v1/attempts"
STATE="$ROOT/data/foundir/_extract_stage/five_type_v1/state"
LOCK="$STATE/install.lock"
COMPLETE="$STATE/install.complete"
PREFLIGHT="$STATE/preflight_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
EXTRACT_WORKERS=${FOUNDIR_EXTRACT_WORKERS:-2}

categories=(
  01Blur 02Blur_Noise 05Noise 08Haze 09Lowlight_Haze 10Rain
  13Rain_Haze 14Lowlight 15Lowlight_Blur 16Lowlight_Noise
  18Lowlight_Blur_Noise
)
declare -A expected_pairs=(
  [01Blur]=109480
  [02Blur_Noise]=29950
  [05Noise]=58015
  [08Haze]=79800
  [09Lowlight_Haze]=79800
  [10Rain]=39900
  [13Rain_Haze]=79750
  [14Lowlight]=39962
  [15Lowlight_Blur]=85893
  [16Lowlight_Noise]=52995
  [18Lowlight_Blur_Noise]=29950
)

mkdir -p "$ATTEMPTS" "$STATE"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "FOUNDIR_FIVE_TYPE_INSTALL_ALREADY_RUNNING"
  exit 75
fi

if [[ -f "$COMPLETE" ]]; then
  if grep -qx 'FOUNDIR_FIVE_TYPE_INSTALL_COMPLETE' "$COMPLETE" && [[ -d "$TRAIN_ROOT" ]]; then
    echo "FOUNDIR_FIVE_TYPE_INSTALL_REUSE marker=$COMPLETE training_root=$TRAIN_ROOT"
    exit 0
  fi
  echo "ERROR: existing FoundIR install marker is inconsistent: $COMPLETE" >&2
  exit 1
fi

on_exit() {
  local status=$?
  echo "FOUNDIR_FIVE_TYPE_INSTALL_EXIT utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) status=$status"
}
trap on_exit EXIT

if [[ ! -f "$DOWNLOAD_MARKER" ]] || ! grep -qx 'FOUNDIR_FIVE_TYPE_DOWNLOAD_COMPLETE' "$DOWNLOAD_MARKER"; then
  echo "ERROR: verified download completion marker is absent: $DOWNLOAD_MARKER" >&2
  exit 1
fi
if [[ ! -x "$SEVEN" || ! -f "$SEVEN_DEB" ]]; then
  echo "ERROR: workspace-local 7-Zip package is missing" >&2
  exit 1
fi
if [[ "$(sha256sum "$SEVEN_DEB" | awk '{print $1}')" != "$SEVEN_DEB_SHA256" ]]; then
  echo "ERROR: workspace-local 7-Zip package hash changed" >&2
  exit 1
fi
seven_banner=$("$SEVEN" 2>&1 || true)
if ! grep -q '7-Zip 23.01' <<<"$seven_banner"; then
  echo "ERROR: unexpected workspace-local 7-Zip binary" >&2
  exit 1
fi
if [[ ! "$EXTRACT_WORKERS" =~ ^[1-9][0-9]*$ ]] || [[ "$EXTRACT_WORKERS" -gt 4 ]]; then
  echo "ERROR: FOUNDIR_EXTRACT_WORKERS must be between 1 and 4" >&2
  exit 1
fi

declare -A archive_sizes=()
while IFS=$'\t' read -r filename size_bytes provider_md5 category; do
  [[ "$filename" == filename ]] && continue
  if [[ "$filename" == "$category.zip" || "$filename" == "$category.z"* ]]; then
    archive_sizes["$filename"]=$size_bytes
  fi
done <"$MANIFEST"

for filename in "${!archive_sizes[@]}"; do
  archive="$ARCHIVES/$filename"
  if [[ ! -f "$archive" ]] || [[ "$(stat -c '%s' "$archive")" -ne "${archive_sizes[$filename]}" ]]; then
    echo "ERROR: archive is missing or has the wrong size: $archive" >&2
    exit 1
  fi
  if [[ -e "$archive.BaiduPCS-Go-downloading" ]]; then
    echo "ERROR: archive still has a download sidecar: $archive" >&2
    exit 1
  fi
done

install_category() {
  local category=$1
  local archive="$ARCHIVES/$category.zip"
  local target="$TRAIN_ROOT/$category"
  local marker="$STATE/$category.complete"
  local attempt="$ATTEMPTS/${category}_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
  local -a candidates=()
  local candidate
  local lq_count
  local gt_count

  if [[ -f "$marker" ]]; then
    if [[ ! -d "$target/LQ" || ! -d "$target/GT" ]]; then
      echo "ERROR: category marker exists but installed data is incomplete: $category" >&2
      return 1
    fi
    echo "FOUNDIR_FIVE_TYPE_INSTALL_REUSE category=$category marker=$marker"
    return 0
  fi
  if [[ -e "$target" ]]; then
    echo "ERROR: refusing to overwrite an unmarked training category: $target" >&2
    return 1
  fi

  mkdir -p "$attempt"
  echo "FOUNDIR_FIVE_TYPE_EXTRACT_START category=$category archive=$archive attempt=$attempt utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$SEVEN" x -y -bb1 -mmt=16 "-o$attempt" "$archive"

  while IFS= read -r -d '' candidate; do
    if [[ -d "$candidate/LQ" && -d "$candidate/GT" ]]; then
      candidates+=("$candidate")
    fi
  done < <(find "$attempt" -type d -name "$category" -print0)
  if [[ ${#candidates[@]} -ne 1 ]]; then
    echo "ERROR: expected exactly one $category directory containing LQ and GT, found ${#candidates[@]} in $attempt" >&2
    return 1
  fi

  candidate=${candidates[0]}
  lq_count=$(find "$candidate/LQ" -type f | wc -l)
  gt_count=$(find "$candidate/GT" -type f | wc -l)
  if [[ "$lq_count" -ne "${expected_pairs[$category]}" || "$gt_count" -ne "${expected_pairs[$category]}" ]]; then
    echo "ERROR: extracted pair count mismatch: category=$category expected=${expected_pairs[$category]} LQ=$lq_count GT=$gt_count" >&2
    return 1
  fi

  mkdir -p "$TRAIN_ROOT"
  mv -- "$candidate" "$target"
  printf 'FOUNDIR_FIVE_TYPE_CATEGORY_INSTALLED\ncategory=%s\narchive=%s\nlq_files=%s\ngt_files=%s\nutc=%s\nofficial_test_accessed=false\n' \
    "$category" "$archive" "$lq_count" "$gt_count" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$marker"
  echo "FOUNDIR_FIVE_TYPE_EXTRACT_PASS category=$category LQ=$lq_count GT=$gt_count marker=$marker"
}

install_worker() {
  local worker=$1
  local i
  for ((i=worker; i<${#categories[@]}; i+=EXTRACT_WORKERS)); do
    install_category "${categories[$i]}"
  done
}

echo "FOUNDIR_FIVE_TYPE_INSTALL_START utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) extract_workers=$EXTRACT_WORKERS official_test_accessed=false"
declare -a worker_pids=()
for ((worker=0; worker<EXTRACT_WORKERS; worker++)); do
  install_worker "$worker" &
  worker_pids+=("$!")
done

install_status=0
for worker in "${!worker_pids[@]}"; do
  if ! wait "${worker_pids[$worker]}"; then
    echo "ERROR: extraction worker failed: worker=$worker pid=${worker_pids[$worker]}" >&2
    install_status=1
  fi
done
if [[ $install_status -ne 0 ]]; then
  exit 1
fi

QWEN_PYTHON="$ROOT/.conda/gpu-train-stable/bin/python"
if [[ ! -x "$QWEN_PYTHON" ]]; then
  echo "ERROR: workspace Qwen Python is unavailable for FoundIR preflight" >&2
  exit 1
fi
"$QWEN_PYTHON" "$ROOT/smoke/foundir_five_type_preflight.py" \
  --config "$ROOT/configs/foundir_five_type_flux2_dit_v1.json" \
  --require-data --skip-large-hashes | tee "$PREFLIGHT"
if ! grep -q 'FOUNDIR_FIVE_TYPE_PREFLIGHT' "$PREFLIGHT"; then
  echo "ERROR: FoundIR five-type preflight did not publish a pass record" >&2
  exit 1
fi

printf 'FOUNDIR_FIVE_TYPE_INSTALL_COMPLETE\nutc=%s\ncategories=%s\ntraining_root=%s\npreflight_log=%s\nofficial_test_accessed=false\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#categories[@]}" "$TRAIN_ROOT" "$PREFLIGHT" >"$COMPLETE"
echo "FOUNDIR_FIVE_TYPE_INSTALL_PASS categories=${#categories[@]} training_root=$TRAIN_ROOT marker=$COMPLETE"
