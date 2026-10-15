#!/usr/bin/env python3
"""Fail-closed data, exposure, and update-budget audit for CURE reproduction."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import h5py

from estimate_cure_h5_v1 import estimate
from extract_ccdd11_v1 import ARCHIVES, DATA, sha256
from prepare_cure_two_track_v1 import COMMON, DATA_ROOT, OUTPUT, RESTORATION, ROOT, TRIPLES, VOCAB, prepare


OFFICIAL = ROOT / "third_party/cure_official"
H5 = OUTPUT / "h5"
EXPECTED_COMMIT = "a294116ad3d7dce58723a21bdd64f74b085e7b13"
BASELINE = tuple(name for name in VOCAB if name != "clear")


def check_h5(path: Path, names: tuple[str, ...], expected_rows: int, *, cure: bool) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        actual_names = tuple(json.loads(handle.attrs["degradation_names"]))
        if actual_names != names:
            raise RuntimeError(f"HDF5 class exposure/order mismatch: {path}: {actual_names}")
        if len(handle) != expected_rows:
            raise RuntimeError(f"HDF5 row count mismatch: {path}: {len(handle)} != {expected_rows}")
        for key in ("0", str(expected_rows - 1)):
            if key not in handle:
                raise RuntimeError(f"HDF5 missing boundary row {key}: {path}")
            row = handle[key]
            if cure:
                expected_keys = {"zero", "gt", "lr", "type1", "type2", "half"}
                if set(row.keys()) != expected_keys:
                    raise RuntimeError(f"CURE HDF5 row fields mismatch: {path} {key}")
                if row["lr"].shape[0] != len(names):
                    raise RuntimeError(f"CURE HDF5 class dimension mismatch: {path} {key}")
            elif row.shape[0] != len(names) + 1:
                raise RuntimeError(f"baseline HDF5 class dimension mismatch: {path} {key}")
    return {"path": str(path.relative_to(ROOT)), "rows": expected_rows, "classes": names}


def preflight() -> dict:
    commit = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"official CURE commit drift: {commit}")
    archives = {}
    for split, (size, digest) in ARCHIVES.items():
        path = DATA / f"{split}.zip"
        if not path.is_file() or path.stat().st_size != size or sha256(path) != digest:
            raise RuntimeError(f"archive size/SHA mismatch: {path}")
        archives[split] = digest
    view = prepare(DATA_ROOT, OUTPUT, check_only=True)
    if view["strict_triple_training_samples"] or view["strict_triple_development_samples"]:
        raise RuntimeError("strict track has triple exposure")
    for category in TRIPLES:
        if any((OUTPUT / "embedder_strict_train/main_data" / category).iterdir()):
            raise RuntimeError("strict embedder triple exposure")
    storage = estimate(OUTPUT / "paired_train/main_data/clear")
    rows = storage["patch_count"]
    h5 = {
        "published_baseline": check_h5(H5 / "published_baseline.h5", BASELINE, rows, cure=False),
        "strict_baseline": check_h5(H5 / "strict_baseline.h5", RESTORATION, rows, cure=False),
        "shared_cure": check_h5(H5 / "shared_cure.h5", RESTORATION, rows, cure=True),
    }
    # With eight ranks and matched per-rank microbatches of 32/16/3, the
    # effective global batches are 256/128/24. DistributedSampler padding is
    # identical between the two tracks because HDF5 row counts match.
    updates = {
        "embedder": math.ceil(view["embedder_presentations_per_epoch_each_track"] / 256) * 200,
        "baseline": math.ceil(rows / 128) * 300,
        "cure": math.ceil(rows / 24) * 300,
    }
    return {
        "official_commit": commit,
        "archive_sha256": archives,
        "source_split": view,
        "h5": h5,
        "storage": storage,
        "matched_optimizer_updates_each_track": updates,
        "official_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(preflight(), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"CURE_TWO_TRACK_PREFLIGHT_FAIL {error}", file=sys.stderr)
        raise SystemExit(2) from None
