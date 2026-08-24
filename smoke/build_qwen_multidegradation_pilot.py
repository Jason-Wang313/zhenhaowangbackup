#!/usr/bin/env python3
"""Build and freeze the deterministic 12,000-record Qwen pilot manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    CONTRACT_SEED,
    build_pilot_records,
    encode_jsonl,
    load_source_inventory,
    validate_pilot_records,
    write_new_or_verify,
)


def main() -> None:
    source_manifest = ROOT / "manifests/low_light_source_pool_2000.json"
    output = ROOT / "manifests/qwen_multidegradation_pilot_v1.jsonl"
    metadata_path = ROOT / "manifests/qwen_multidegradation_pilot_v1.json"
    inventory = load_source_inventory(source_manifest)
    records = build_pilot_records(inventory)
    validation = validate_pilot_records(records)
    manifest_sha = write_new_or_verify(output, encode_jsonl(records))
    metadata = {
        "schema_version": 1,
        "status": "frozen_manifest_only_images_not_implied_generated",
        "contract": "configs/qwen_multidegradation_contract_v1.json",
        "source_manifest": source_manifest.relative_to(ROOT).as_posix(),
        "seed": CONTRACT_SEED,
        "manifest": output.relative_to(ROOT).as_posix(),
        "manifest_sha256": manifest_sha,
        "validation": validation,
    }
    encoded = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    metadata_sha = write_new_or_verify(metadata_path, encoded)
    print(
        "QWEN_MULTIDEGRADATION_MANIFEST_PASS "
        f"records={len(records)} manifest_sha256={manifest_sha} metadata_sha256={metadata_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
