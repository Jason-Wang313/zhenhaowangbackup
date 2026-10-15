#!/usr/bin/env bash
set -euo pipefail

ROOT=/inspire/hdd/global_user/liuxiaohong-25080/zhenhaowang
PYTHON="$ROOT/.venv/flux2-smoke/bin/python"
CONFIG="$ROOT/configs/foundir_haze_flux2_source_diverse_pilot_v1.json"
RUNNER="$ROOT/smoke/run_foundir_haze_flux2_source_diverse_pilot_v1_8gpu.sh"
LOG="$ROOT/logs/foundir_haze_flux2_source_diverse_pilot_v1.log"
PID_FILE="$ROOT/logs/foundir_haze_flux2_source_diverse_pilot_v1.pid"
RESULT="$ROOT/checkpoints/foundir_haze_flux2_source_diverse_pilot_v1/result.json"
PREFLIGHT="$ROOT/reports/foundir_haze_flux2_source_diverse_pilot_preflight_v1.json"

cd "$ROOT"
/usr/bin/python3 smoke/advisor_governance.py check
[[ $# -le 1 && ( $# -eq 0 || "$1" == "--check" ) ]] || {
  echo "ERROR: only optional --check is supported" >&2; exit 1;
}
[[ "${CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7" ]] || {
  echo "ERROR: export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" >&2; exit 1;
}
for command in setsid flock sha256sum; do command -v "$command" >/dev/null || { echo "ERROR: $command is required" >&2; exit 1; }; done
for path in "$PYTHON" "$CONFIG" "$RUNNER"; do [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 1; }; done

"$PYTHON" -m py_compile \
  smoke/freeze_foundir_haze_flux2_source_diverse_pilot.py \
  smoke/train_foundir_blur_only_flux2_overfit_ddp.py \
  tests/test_foundir_haze_flux2_source_diverse_pilot.py
bash -n "$RUNNER" "$0"
"$PYTHON" -m pytest -q tests/test_foundir_haze_flux2_source_diverse_pilot.py
"$PYTHON" smoke/freeze_foundir_haze_flux2_source_diverse_pilot.py --config "$CONFIG"
"$PYTHON" - "$PREFLIGHT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path.cwd()
receipt_path=pathlib.Path(sys.argv[1])
p=json.loads(receipt_path.read_text())
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16*1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
assert p["status"] == "FOUNDIR_SINGLE_TASK_PREFLIGHT_PASS"
for path_key,hash_key in (
    ("config","config_sha256"),
    ("frozen_manifest","frozen_manifest_sha256"),
    ("manifest_metadata","manifest_metadata_sha256"),
    ("flow_weights","flow_weights_sha256"),
    ("autoencoder_weights","autoencoder_weights_sha256"),
):
    path=root/p[path_key]
    assert path.is_file() and sha(path) == p[hash_key], path
assert p["pair_files_hash_verified"] == 3416 and p["crops_decoded_verified"] == 3416
assert p["train_source_groups"] == 1580 and p["validation_source_groups"] == 128
assert p["train_validation_source_group_overlap"] == 0
assert p["protected_foundir_test_accessed"] is False and p["sealed_internal_accessed"] is False
print("FOUNDIR_HAZE_SOURCE_DIVERSE_PREFLIGHT_RECEIPT_PASS")
PY
if [[ "${1:-}" == "--check" ]]; then
  echo "FOUNDIR_HAZE_SOURCE_DIVERSE_LAUNCHER_CHECK_PASS"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is required" >&2; exit 1; }
[[ ! -f "$RESULT" ]] || { echo "ERROR: terminal result already exists" >&2; exit 1; }
mkdir -p "$ROOT/logs"
nohup setsid env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash "$RUNNER" >/dev/null 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
sleep 3
kill -0 "$pid" 2>/dev/null || {
  echo "ERROR: runner exited during startup; inspect $LOG" >&2
  tail -100 "$LOG" >&2 || true
  exit 1
}
echo "FOUNDIR_HAZE_SOURCE_DIVERSE_DETACHED process_group=$pid"
echo "Persistent log: $LOG"
echo "Monitor: tail -f $LOG"
echo "Stop safely: kill -- -$pid"
