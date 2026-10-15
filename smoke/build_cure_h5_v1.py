#!/usr/bin/env python3
"""Build three immutable HDF5 databases using CURE's pinned original builder."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

from estimate_cure_h5_v1 import estimate
from preflight_cure_two_track_v1 import BASELINE, H5, OFFICIAL, OUTPUT, RESTORATION, check_h5


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = OFFICIAL / "tools/prepare_h5.py"
BUILDS = (
    ("published_baseline", "baseline", BASELINE),
    ("strict_baseline", "baseline", RESTORATION),
    ("shared_cure", "cure", RESTORATION),
)


def build(python: Path, *, check_only: bool) -> dict:
    subprocess.run([str(python), str(ROOT / "smoke/advisor_governance.py"), "check"], check=True)
    storage = estimate(OUTPUT / "paired_train/main_data/clear")
    missing_bytes = sum(
        storage["breakdown_bytes"]["shared_cure_finetune" if label == "shared_cure" else label]
        for label, _, _ in BUILDS if not (H5 / f"{label}.h5").exists()
    )
    if storage["filesystem_free_bytes"] < missing_bytes * 1.25:
        raise RuntimeError("insufficient filesystem free space for remaining CURE HDF5")
    expected_rows = storage["patch_count"]
    H5.mkdir(parents=True, exist_ok=True)
    result = {}
    with (H5 / "build.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for label, mode, names in BUILDS:
            destination = H5 / f"{label}.h5"
            if destination.exists():
                result[label] = check_h5(destination, names, expected_rows, cure=mode == "cure")
                continue
            if check_only:
                raise FileNotFoundError(destination)
            temporary = H5 / f"{label}.partial.h5"
            if temporary.exists():
                raise RuntimeError(f"partial HDF5 requires manual audit: {temporary}")
            args = [
                str(python), str(SCRIPT), mode,
                "--data-root", str(OUTPUT / "paired_train"),
                "--output", str(temporary),
                "--patch-size", "256", "--stride", "200", "--seed", "124",
            ]
            if label == "strict_baseline":
                args += ["--types", *RESTORATION]
            print(f"CURE_H5_BUILD_START {label} rows={expected_rows}", flush=True)
            subprocess.run(args, cwd=OFFICIAL, check=True)
            result[label] = check_h5(temporary, names, expected_rows, cure=mode == "cure")
            os.replace(temporary, destination)
            result[label]["path"] = str(destination.relative_to(ROOT))
            print(f"CURE_H5_BUILD_END {label} bytes={destination.stat().st_size}", flush=True)
    return {"storage_estimate": storage, "databases": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(Path(sys.executable), check_only=args.check), indent=2))


if __name__ == "__main__":
    main()
