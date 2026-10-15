#!/usr/bin/env python3
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
    ROUTE_PLANS,
    expert_plan,
    final_eleven_route,
)

CONFIG = ROOT / "configs/foundir_final_eleven_route_cpu_gate_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def boolean(value: str) -> bool:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise ValueError(value)


def main() -> None:
    config = json.loads(CONFIG.read_text())
    if config["scope"]["official_or_sealed_test_allowed"] is not False:
        raise RuntimeError("protected test access is not forbidden")
    loaded = {}
    for name, spec in config["inputs"].items():
        path = ROOT / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise RuntimeError(f"input identity changed: {name}")
        data = rows(path)
        loaded[name] = {row["record_id"]: row for row in data}
        if len(loaded[name]) != len(data):
            raise RuntimeError(f"duplicate record ID: {name}")
    ids = set(loaded["compositional"])
    if len(ids) != 16461 or any(set(value) != ids for value in loaded.values()):
        raise RuntimeError("the four heads are not aligned on 16,461 records")

    counts: dict[str, Counter] = {route: Counter() for route in ROUTE_PLANS}
    decisions = []
    for record_id in sorted(ids):
        comp = loaded["compositional"][record_id]
        eleven = loaded["eleven_consensus"][record_id]
        dual = loaded["dual_nonlinear"][record_id]
        physical = loaded["physical_guard"][record_id]
        truths = {comp["truth"], eleven["truth"], dual["truth"], physical["truth"]}
        if len(truths) != 1:
            raise RuntimeError(f"truth mismatch: {record_id}")
        truth = truths.pop()
        route = final_eleven_route(
            compositional_route=comp["route"],
            compositional_accepted=boolean(comp["accepted"]),
            eleven_route=eleven["route"],
            eleven_accepted=boolean(eleven["accepted"]),
            dual_route=dual["route"],
            dual_accepted=boolean(dual["accepted"]),
            physical_route=physical["final_route"],
        )
        status = "abstain" if route is None else "correct" if route == truth else "wrong"
        counts[truth][status] += 1
        decisions.append({
            "record_id": record_id,
            "truth": truth,
            "final_route": route or "abstain",
            "expert_plan": "+".join(expert_plan(route)) or "none",
            "status": status,
        })

    metrics = {}
    for truth, count in counts.items():
        total = sum(count.values())
        accepted = count["correct"] + count["wrong"]
        metrics[truth] = {
            "records": total,
            "correct": count["correct"],
            "wrong": count["wrong"],
            "abstain": count["abstain"],
            "coverage": accepted / total,
            "wrong_rate": count["wrong"] / total,
        }
    overall = Counter(row["status"] for row in decisions)
    overall_coverage = (overall["correct"] + overall["wrong"]) / len(decisions)
    checks = {
        "zero_wrong_routes": overall["wrong"] == config["gates"]["required_wrong_routes"],
        "minimum_overall_coverage": overall_coverage >= config["gates"]["minimum_overall_coverage"],
        "minimum_per_class_coverage": min(value["coverage"] for value in metrics.values()) >= config["gates"]["minimum_per_class_coverage"],
    }
    output = ROOT / config["outputs"]["directory"]
    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / config["outputs"]["decisions"]
    with decision_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader(); writer.writerows(decisions)
    result = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "failed_before_gpu_integration",
        "gate_pass": all(checks.values()),
        "records": len(decisions),
        "overall": {
            "correct": overall["correct"], "wrong": overall["wrong"],
            "abstain": overall["abstain"], "coverage": overall_coverage,
            "wrong_rate": overall["wrong"] / len(decisions),
        },
        "per_class": metrics,
        "checks": checks,
        "gpu_count": 0,
        "gpu_integration_launched": False,
        "model_weight_updates": False,
        "official_or_sealed_test_accessed": False,
        "decisions_sha256": sha256(decision_path),
    }
    (output / config["outputs"]["result"]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
