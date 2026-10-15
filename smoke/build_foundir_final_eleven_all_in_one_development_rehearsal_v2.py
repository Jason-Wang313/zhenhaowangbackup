#!/usr/bin/env python3
"""Freeze the one-rule dark-abstention fallback after rehearsal v1."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.final_eleven_route_router import (  # noqa: E402
    LOW_LIGHT_FALLBACK_MAX_EXPOSURE,
    safe_expert_plan,
)
from lowlight_restoration.noise_lowlight_physical_guard import physical_features  # noqa: E402

SOURCE = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v1.csv"
SOURCE_SHA256 = "358ad8cbd8c4183442a43c14be0322c54ce2856f8daeb9c164a9d10e67843790"
OUTPUT = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v2.csv"
MAXIMUM_SIDE = 128


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("v1 rehearsal manifest identity changed")
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    output_rows = []
    fallback_truths = Counter()
    for row in rows:
        exposure = ""
        if row["routing_status"] == "abstain":
            features = physical_features(ROOT / row["lq_path"], maximum_side=MAXIMUM_SIDE)
            exposure = f"{features['exposure']:.12g}"
            plan = safe_expert_plan(None, exposure=features["exposure"])
            if plan:
                row["final_route"] = "low_light_fallback"
                row["expert_order_json"] = json.dumps(list(plan), separators=(",", ":"))
                row["routing_status"] = "fallback"
                fallback_truths[row["router_truth"]] += 1
        row["fallback_exposure"] = exposure
        output_rows.append(row)
    if sum(fallback_truths.values()) != 67:
        raise RuntimeError(f"fallback cardinality changed: {fallback_truths}")
    non_lowlight = {key: value for key, value in fallback_truths.items() if "low_light" not in key}
    if non_lowlight:
        raise RuntimeError(f"fallback reached non-low-light development classes: {non_lowlight}")
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["fallback_exposure"])
        writer.writeheader()
        writer.writerows(output_rows)
    temporary.replace(OUTPUT)
    payload = {
        "schema_version": 1,
        "status": "frozen_post_rehearsal_low_light_fallback_v2",
        "source_manifest": str(SOURCE.relative_to(ROOT)),
        "source_manifest_sha256": SOURCE_SHA256,
        "manifest_sha256": sha256(OUTPUT),
        "records": len(output_rows),
        "fallback_records": sum(fallback_truths.values()),
        "fallback_truth_counts_development_audit_only": dict(sorted(fallback_truths.items())),
        "rule": {
            "when": "final_router_abstains",
            "maximum_exposure": LOW_LIGHT_FALLBACK_MAX_EXPOSURE,
            "maximum_side": MAXIMUM_SIDE,
            "expert_plan": ["low_light"],
            "semantic": "partial_restoration_without_claiming_singleton_classification",
        },
        "model_weight_updates": False,
        "official_or_sealed_test_accessed": False,
        "not_an_independent_lockbox": True,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
