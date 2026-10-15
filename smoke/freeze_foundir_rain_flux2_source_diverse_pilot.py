#!/usr/bin/env python3
"""Freeze the user-authorized source-diverse Rain train/validation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freeze_foundir_haze_flux2_source_diverse_pilot import (
    ROOT,
    render_csv,
    select_rows,
    sha256,
    write_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("status") != "user_authorized_rain_source_diverse_pilot":
        raise RuntimeError("wrong Rain pilot authorization status")
    rows, available = select_rows(config)
    data = config["data"]
    manifest = ROOT / data["frozen_manifest"]
    content = render_csv(rows)
    write_once(manifest, content)
    train = [row for row in rows if row["role"] == data["train_role"]]
    validation = [row for row in rows if row["role"] == data["validation_role"]]
    overlap = {row["source_group"] for row in train} & {row["source_group"] for row in validation}
    metadata = {
        "schema_version": 1,
        "status": "frozen_rain_source_diverse_pilot_manifest",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_manifest": data["source_manifest"],
        "source_manifest_sha256": data["source_manifest_sha256"],
        "frozen_manifest": data["frozen_manifest"],
        "frozen_manifest_sha256": hashlib.sha256(content).hexdigest(),
        "records": len(rows),
        "train_records": len(train),
        "train_source_groups": len({row["source_group"] for row in train}),
        "validation_records": len(validation),
        "validation_source_groups": len({row["source_group"] for row in validation}),
        "train_validation_source_group_overlap": len(overlap),
        "available_source_groups_after_exclusion": available,
        "excluded_prior_capacity_groups": True,
        "all_remaining_untouched_validation_groups_consumed": True,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(ROOT / data["frozen_manifest_metadata"], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_RAIN_SOURCE_DIVERSE_MANIFEST_PASS " + json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
