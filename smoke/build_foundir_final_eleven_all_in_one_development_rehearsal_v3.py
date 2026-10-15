#!/usr/bin/env python3
"""Extend v2 fallback when the frozen five-class proposal says low-light."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v2.csv"
SOURCE_SHA256 = "46606f4b7a46d8fcd6745288d1be39c75ab34f23f574af55aa7a8c66f091f503"
PROPOSALS = ROOT / "checkpoints/foundir_qwen_mixed_consensus_development_v3/predictions.csv"
PROPOSALS_SHA256 = "7fbd2a92ce659ecb6222b3c8971f182aa9a03006d3f1489a98a89d89030957a0"
OUTPUT = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v3.csv"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(PROPOSALS) != PROPOSALS_SHA256:
        raise RuntimeError("frozen v2 manifest or proposal identity changed")
    with PROPOSALS.open(newline="") as handle:
        proposals = {row["record_id"]: row for row in csv.DictReader(handle)}
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    added = Counter()
    for row in rows:
        proposal = proposals[row["record_id"]]["five_proposal"]
        row["fallback_five_proposal"] = proposal
        if row["routing_status"] == "abstain" and proposal == "low_light":
            row["final_route"] = "low_light_fallback"
            row["expert_order_json"] = '["low_light"]'
            row["routing_status"] = "fallback"
            added[row["router_truth"]] += 1
    if sum(added.values()) != 20 or added["low_light"] != 7:
        raise RuntimeError(f"v3 fallback cardinality changed: {added}")
    non_lowlight = {key: value for key, value in added.items() if "low_light" not in key}
    if non_lowlight:
        raise RuntimeError(f"proposal fallback reached non-low-light classes: {non_lowlight}")
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["fallback_five_proposal"])
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(OUTPUT)
    payload = {
        "schema_version": 1,
        "status": "frozen_post_rehearsal_low_light_fallback_v3",
        "source_manifest_sha256": SOURCE_SHA256,
        "proposal_predictions_sha256": PROPOSALS_SHA256,
        "manifest_sha256": sha256(OUTPUT),
        "records": len(rows),
        "v2_fallback_records": 67,
        "new_proposal_fallback_records": sum(added.values()),
        "total_fallback_records": sum(row["routing_status"] == "fallback" for row in rows),
        "new_fallback_truth_counts_development_audit_only": dict(sorted(added.items())),
        "rule": "retain_v2; additionally, on abstention, five_proposal=low_light runs low_light as partial restoration",
        "model_weight_updates": False,
        "official_or_sealed_test_accessed": False,
        "not_an_independent_lockbox": True,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
