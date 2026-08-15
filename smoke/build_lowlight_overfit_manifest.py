#!/usr/bin/env python3
"""Build the deterministic, source-unique 16-pair low-light overfit manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUOTAS = {"L1": 3, "L2": 3, "L3": 3, "L4": 3, "L5": 4}


def stable_key(seed: int, pair_id: str) -> str:
    return hashlib.sha256(f"{seed}:{pair_id}".encode()).hexdigest()


def write_once_or_verify(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise SystemExit(f"REFUSING_TO_OVERWRITE_DIFFERING_FILE {path}")
        print(f"REUSED {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    print(f"WROTE {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "manifests/low_light_pilot_10000.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests/low_light_overfit16.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "manifests/low_light_overfit16.json",
    )
    parser.add_argument("--seed", type=int, default=2026081501)
    args = parser.parse_args()

    with args.source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit("SOURCE_MANIFEST_HAS_NO_HEADER")
        rows = list(reader)
        fieldnames = reader.fieldnames

    selected: list[dict[str, str]] = []
    used_sources: set[str] = set()
    for level, quota in QUOTAS.items():
        candidates = sorted(
            (
                row
                for row in rows
                if row["split"] == "train"
                and row["degradation_type"] == "low_light"
                and row["project_level"] == level
            ),
            key=lambda row: stable_key(args.seed, row["pair_id"]),
        )
        for row in candidates:
            if row["source_id"] in used_sources:
                continue
            selected.append(row)
            used_sources.add(row["source_id"])
            if sum(item["project_level"] == level for item in selected) == quota:
                break
        realized = sum(item["project_level"] == level for item in selected)
        if realized != quota:
            raise SystemExit(f"INSUFFICIENT_UNIQUE_SOURCES {level}: {realized}/{quota}")

    selected.sort(key=lambda row: stable_key(args.seed, row["pair_id"]))
    if len(selected) != 16 or len(used_sources) != 16:
        raise SystemExit("BAD_OVERFIT_SELECTION")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    csv_bytes = buffer.getvalue().encode()
    write_once_or_verify(args.output, csv_bytes)

    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    output_digest = hashlib.sha256(csv_bytes).hexdigest()
    metadata = {
        "schema_version": 1,
        "purpose": "FLUX.2 low-light restoration 16-pair overfit smoke",
        "source_manifest": str(args.source.resolve().relative_to(ROOT)),
        "source_manifest_sha256": source_digest,
        "output_manifest": str(args.output.resolve().relative_to(ROOT)),
        "output_manifest_sha256": output_digest,
        "seed": args.seed,
        "records": len(selected),
        "unique_sources": len(used_sources),
        "split": "train",
        "levels": dict(Counter(row["project_level"] for row in selected)),
        "pair_ids": [row["pair_id"] for row in selected],
    }
    write_once_or_verify(args.metadata, (json.dumps(metadata, indent=2) + "\n").encode())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
