#!/usr/bin/env python3
"""Freeze a status-unfiltered, class-balanced 11-class development rehearsal."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manifests/foundir_five_type_pairs_v1.csv"
SOURCE_SHA256 = "e8c018879029c0bcc2da8d074cc4999541158ca97d07f48aadbcdddcdccc3ca7"
DECISIONS = ROOT / "checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv"
DECISIONS_SHA256 = "0342bf00bef98eccb9c3d3ad8e3c42c2368db2761f3c2ddd38f9b1342ad31225"
OUTPUT = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v1.csv"
SEED = 2026092001
PER_CATEGORY = 128
ROUTE_PLANS = {
    "blur": ["blur"], "noise": ["noise"], "haze": ["haze"], "rain": ["rain"],
    "low_light": ["low_light"], "blur+noise": ["blur_noise"],
    "haze+low_light": ["haze", "low_light"], "rain+haze": ["rain", "haze"],
    "blur+low_light": ["low_light", "blur"],
    "noise+low_light": ["noise", "low_light"],
    "blur+noise+low_light": ["noise", "low_light", "blur"],
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def key(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256 or sha256(DECISIONS) != DECISIONS_SHA256:
        raise RuntimeError("frozen source identity changed")
    with DECISIONS.open(newline="") as handle:
        decisions = {row["record_id"]: row for row in csv.DictReader(handle)}
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with SOURCE.open(newline="") as handle:
        fields = list(csv.DictReader(handle).fieldnames or [])
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "validation" or row["record_id"] not in decisions:
                continue
            if min(int(row["width"]), int(row["height"])) < 512:
                continue
            grouped[row["category"]][row["source_group"]].append(dict(row))
    selected = []
    for category in sorted(grouped):
        groups = sorted(grouped[category], key=lambda value: key(SEED, category, "group", value))
        ordered = []
        depth = 0
        while len(ordered) < PER_CATEGORY:
            added = 0
            for group in groups:
                records = sorted(grouped[category][group], key=lambda row: key(SEED, category, group, row["record_id"]))
                if depth < len(records):
                    ordered.append(records[depth]); added += 1
                    if len(ordered) == PER_CATEGORY:
                        break
            if not added:
                raise RuntimeError(f"insufficient eligible records for {category}")
            depth += 1
        for row in ordered:
            decision = decisions[row["record_id"]]
            route = decision["final_route"] or None
            order = ROUTE_PLANS.get(route, [])
            selected.append({
                **row,
                "router_truth": decision["truth"],
                "final_route": route or "",
                "expert_order_json": json.dumps(order, separators=(",", ":")),
                "routing_status": decision["status"],
            })
    selected.sort(key=lambda row: (row["category"], key(SEED, "output", row["record_id"])))
    if len(selected) != 11 * PER_CATEGORY:
        raise RuntimeError("rehearsal cardinality changed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["router_truth", "final_route", "expert_order_json", "routing_status"])
        writer.writeheader(); writer.writerows(selected)
    temporary.replace(OUTPUT)
    payload = {
        "schema_version": 1,
        "status": "frozen_final_eleven_all_in_one_development_rehearsal_v1",
        "records": len(selected),
        "records_by_category": dict(sorted(Counter(row["category"] for row in selected).items())),
        "routing_status_counts": dict(sorted(Counter(row["routing_status"] for row in selected).items())),
        "route_counts": dict(sorted(Counter(row["final_route"] or "abstain" for row in selected).items())),
        "expert_applications": sum(len(json.loads(row["expert_order_json"])) for row in selected),
        "source_groups": len({row["source_group"] for row in selected}),
        "selection": "category_balanced_source_group_round_robin_independent_of_router_status",
        "abstention_output": "identity_input",
        "manifest_sha256": sha256(OUTPUT),
        "source_manifest_sha256": SOURCE_SHA256,
        "routing_decisions_sha256": DECISIONS_SHA256,
        "development_pixels_accessed": True,
        "official_or_sealed_test_accessed": False,
        "not_an_independent_lockbox": True,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
