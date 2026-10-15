#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$ROOT/third_party/fftformer-upstream"
REVISION="01306b5d4072146a011c8c6e9d4927cf4607f1d5"

if [[ ! -d "$TARGET/.git" ]]; then
  mkdir -p "$ROOT/third_party"
  git clone https://github.com/kkkls/FFTformer.git "$TARGET"
fi

actual_revision="$(git -C "$TARGET" rev-parse HEAD)"
if [[ "$actual_revision" != "$REVISION" ]]; then
  git -C "$TARGET" fetch origin "$REVISION"
  git -C "$TARGET" checkout --detach "$REVISION"
fi

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$actual" != "$expected" ]]; then
    echo "Pinned FFTformer artifact changed: $path" >&2
    exit 1
  fi
}

check_sha "dc0e1775eed3c3358f637084e9e345ec0246b66084487dc4220aad0a5cb66ddf" \
  "$TARGET/basicsr/models/archs/fftformer_arch.py"
check_sha "336717b7c4e97b66af112cc7878106cb185b698bd872c2a8247911507540b2a3" \
  "$TARGET/pretrain_model/fftformer_GoPro.pth"
check_sha "ebbb97354efc45d568b5bd7100a3e70447256b7bbbe0c6d340cd382103751e30" \
  "$TARGET/license"

echo "FFTformer upstream acquisition verified at $REVISION"
