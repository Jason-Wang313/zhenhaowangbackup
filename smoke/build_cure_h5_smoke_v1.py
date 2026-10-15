#!/usr/bin/env python3
"""One-source original-code HDF5 construction smoke, without GPU or test data."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from prepare_cure_two_track_v1 import OUTPUT, RESTORATION, ROOT, VOCAB, link_exact


SMOKE = ROOT / "data/third_party/cure_h5_smoke_v1"
REPORT = ROOT / "reports/cure_two_track_h5_smoke_v1.json"
OFFICIAL = ROOT / "third_party/cure_official/tools/prepare_h5.py"


def main() -> None:
    paired = OUTPUT / "paired_train"
    sources = sorted((paired / "main_data/clear").glob("*.png"))
    if not sources:
        raise RuntimeError("paired training view is missing")
    source_name = sources[0].name
    stem = Path(source_name).stem
    view = SMOKE / "half_train"
    for category in (*VOCAB, "zero"):
        source = paired / "main_data" / category / source_name
        link_exact(source.resolve(), view / "main_data" / category / source_name)
    for category in RESTORATION:
        parts = category.split("_")
        factors = (parts[0], "zero") if len(parts) == 1 else (parts[0], parts[1])
        for suffix in (*factors, "half"):
            name = f"{stem}_{suffix}_.png"
            source = paired / "sub_data" / category / stem / name
            link_exact(source.resolve(), view / "sub_data" / category / stem / name)
    builds = {}
    for label, mode, classes in (
        ("published_baseline", "baseline", None),
        ("strict_baseline", "baseline", RESTORATION),
        ("shared_cure", "cure", None),
    ):
        target = SMOKE / f"{label}.h5"
        if target.exists():
            raise RuntimeError(f"refusing to overwrite HDF5 smoke artifact: {target}")
        args = [
            sys.executable, str(OFFICIAL), mode,
            "--data-root", str(view), "--output", str(target),
            "--patch-size", "256", "--stride", "200", "--seed", "124",
        ]
        if classes is not None:
            args += ["--types", *classes]
        started = time.perf_counter()
        subprocess.run(args, check=True, cwd=ROOT / "third_party/cure_official")
        builds[label] = {"bytes": target.stat().st_size, "wall_seconds": time.perf_counter() - started}
    result = {
        "schema_version": 1,
        "source_name": source_name,
        "official_commit": "a294116ad3d7dce58723a21bdd64f74b085e7b13",
        "builds": builds,
        "gpu_hours": 0,
        "official_foundir_test_accessed": False,
    }
    with REPORT.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
