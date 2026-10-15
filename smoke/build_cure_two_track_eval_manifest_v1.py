#!/usr/bin/env python3
"""Freeze exact matched CCDD-11 triple full/selective target mappings.

This builder inspects names and paths only. It does not read image pixels or
compute metrics before both trained tracks are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/third_party/ccdd11/extracted/half_test"
MANIFEST = ROOT / "manifests/cure_two_track_v1_eval.csv"
METADATA = ROOT / "manifests/cure_two_track_v1_eval.json"
TRIPLES = ("low_haze_rain", "low_haze_snow")
FIELDS = (
    "id", "kind", "source_prompt", "remove_prompt", "input_path",
    "target_path", "target_provenance",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build(test: Path) -> tuple[str, dict]:
    test = test.resolve()
    test.relative_to(ROOT)
    rows = []
    counts = {}
    for source_prompt in TRIPLES:
        source_dir = test / "main_data" / source_prompt
        names = sorted(path.name for path in source_dir.glob("*.png"))
        if not names:
            raise RuntimeError(f"no triple test images: {source_dir}")
        counts[source_prompt] = len(names)
        factors = source_prompt.split("_")
        for name in names:
            stem = Path(name).stem
            source = source_dir / name
            clean = test / "main_data/clear" / name
            if not clean.is_file():
                raise FileNotFoundError(clean)
            jobs = [("full", source_prompt, clean, "CCDD11_clean_same_source")]
            intermediate = test / "sub_data" / source_prompt / stem
            for removed in factors:
                retained = "_".join(factor for factor in factors if factor != removed)
                target = intermediate / "2deg" / f"{stem}_{retained}_.png"
                jobs.append(("selective", removed, target, "verified_matched_intermediate"))
            for retained in factors:
                removed = "_".join(factor for factor in factors if factor != retained)
                target = intermediate / "1deg" / f"{stem}_{retained}_.png"
                jobs.append(("selective", removed, target, "verified_matched_intermediate"))
            for kind, remove_prompt, target, provenance in jobs:
                if not target.is_file():
                    raise FileNotFoundError(target)
                rows.append({
                    "id": f"{stem}_{source_prompt}_{kind}_{remove_prompt}",
                    "kind": kind,
                    "source_prompt": source_prompt,
                    "remove_prompt": remove_prompt,
                    "input_path": rel(source),
                    "target_path": rel(target),
                    "target_provenance": provenance,
                })
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate evaluation IDs")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = stream.getvalue()
    metadata = {
        "schema_version": 1,
        "dataset_revision": "4784b88382f8616008ac9ae1e9be36c75f871f4e",
        "target_mapping": "same-triple-source CCDD-11 sub_data/2deg for one removal, sub_data/1deg for two removals",
        "source_counts": counts,
        "full_rows": sum(row["kind"] == "full" for row in rows),
        "selective_rows": sum(row["kind"] == "selective" for row in rows),
        "manifest_sha256": sha256_bytes(content.encode()),
        "official_foundir_test_accessed": False,
    }
    return content, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, default=TEST)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content, metadata = build(args.test_root)
    metadata_content = json.dumps(metadata, sort_keys=True, indent=2) + "\n"
    if args.check:
        if args.manifest.read_text() != content or args.metadata.read_text() != metadata_content:
            raise RuntimeError("evaluation manifest/metadata drift")
    else:
        for path, value in ((args.manifest, content), (args.metadata, metadata_content)):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text() != value:
                    raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
            else:
                path.write_text(value)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
