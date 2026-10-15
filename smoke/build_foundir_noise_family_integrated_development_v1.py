#!/usr/bin/env python3
"""Freeze accepted Noise-family development rows for end-to-end integration."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "checkpoints/foundir_noise_family_fail_closed_consensus_v1/development_decisions.csv"
DECISIONS_SHA256 = "ad7e9d487741de458018c016c6aa9582b5e883c5384bb3bef9254b32bd5c3182"
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
SOURCE_SHA256 = "e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7"
OUTPUT = ROOT / "manifests/foundir_noise_family_integrated_development_v1.csv"
SUMMARY = OUTPUT.with_suffix(".json")
EXPECTED = {"noise": 674, "noise+low_light": 1168}
CATEGORY = {"noise": "05Noise", "noise+low_light": "16Lowlight_Noise"}
PLAN = {"noise": ["noise"], "noise+low_light": ["noise", "low_light"]}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if sha256(DECISIONS) != DECISIONS_SHA256 or sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("frozen source identity changed")
    with DECISIONS.open(newline="") as handle:
        decisions = list(csv.DictReader(handle))
    accepted = {
        row["record_id"]: row
        for row in decisions
        if row["status"] == "correct" and row["final_route"] in EXPECTED
    }
    if Counter(row["final_route"] for row in accepted.values()) != Counter(EXPECTED):
        raise RuntimeError("accepted decision cardinality changed")

    with SOURCE.open(newline="") as handle:
        source_rows = {row["record_id"]: row for row in csv.DictReader(handle)}
    if set(accepted) - set(source_rows):
        raise RuntimeError("accepted records are missing from the source manifest")

    rows = []
    for record_id in sorted(accepted):
        decision = accepted[record_id]
        source = source_rows[record_id]
        route = decision["final_route"]
        if source["split"] != "validation" or source["category"] != CATEGORY[route]:
            raise RuntimeError(f"development row contract changed: {record_id}")
        rows.append({
            **source,
            "router_truth": decision["truth"],
            "final_route": route,
            "expert_order_json": json.dumps(PLAN[route], separators=(",", ":")),
            "routing_status": decision["status"],
            "routing_reason": decision["reason"],
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(OUTPUT)
    summary = {
        "schema_version": 1,
        "status": "frozen_noise_family_fail_closed_accepted_development_v1",
        "records": len(rows),
        "records_by_route": dict(sorted(Counter(row["final_route"] for row in rows).items())),
        "expert_applications": sum(len(json.loads(row["expert_order_json"])) for row in rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "split": "validation",
        "source_manifest_sha256": SOURCE_SHA256,
        "routing_decisions_sha256": DECISIONS_SHA256,
        "manifest_sha256": sha256(OUTPUT),
        "official_or_internal_test_accessed": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
