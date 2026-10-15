#!/usr/bin/env python3
"""Hard-link immutable v2 artifacts so v3 computes only 20 new fallbacks."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v3.csv"
SOURCE_MANIFEST = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v2.csv"
SOURCE = ROOT / "checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v2"
OUTPUT = ROOT / "checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v3"


def link_once(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if source.stat().st_ino != target.stat().st_ino:
            raise RuntimeError(f"existing target is not the frozen source hard link: {target}")
        return
    os.link(source, target)


def main() -> None:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with SOURCE_MANIFEST.open(newline="") as handle:
        source_status = {row["record_id"]: row["routing_status"] for row in csv.DictReader(handle)}
    fallback_ids = {
        row["record_id"] for row in rows
        if row["routing_status"] == "fallback" and source_status[row["record_id"]] == "abstain"
    }
    if len(fallback_ids) != 20:
        raise RuntimeError("v3 incremental fallback cardinality changed")
    for source in (SOURCE / "stages").glob("*/*.png"):
        link_once(source, OUTPUT / source.relative_to(SOURCE))
    linked_sidecars = 0
    for source in (SOURCE / "per_record").glob("*.json"):
        if source.stem in fallback_ids:
            continue
        link_once(source, OUTPUT / source.relative_to(SOURCE))
        linked_sidecars += 1
    for source in (SOURCE / "fixed_visuals").glob("*.png"):
        if source.stem not in fallback_ids:
            link_once(source, OUTPUT / source.relative_to(SOURCE))
    payload = {
        "schema_version": 1,
        "status": "prepared",
        "source": str(SOURCE.relative_to(ROOT)),
        "fallback_records_pending": len(fallback_ids),
        "unchanged_metric_sidecars_hard_linked": linked_sidecars,
        "model_weight_updates": False,
        "official_or_sealed_test_accessed": False,
    }
    path = OUTPUT / "reuse_manifest.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("reuse manifest identity changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
