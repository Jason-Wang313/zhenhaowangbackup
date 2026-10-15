#!/usr/bin/env python3
"""Reproduce the frozen Noise-family consensus gate on development data only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lowlight_restoration.noise_family_fail_closed_router import (  # noqa: E402
    NOISE_FAMILY_ROUTES,
    select_noise_family_route,
)


DEFAULT_CONFIG = ROOT / "configs/foundir_noise_family_fail_closed_consensus_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["scope"]["official_or_internal_test_allowed"] is not False:
        raise RuntimeError("configuration does not forbid official/internal test access")

    resolved: dict[str, Path] = {}
    verified_hashes: dict[str, str] = {}
    for name, artifact in config["frozen_inputs"].items():
        path = ROOT / artifact["path"]
        actual = sha256(path)
        if actual != artifact["sha256"]:
            raise RuntimeError(f"frozen artifact hash mismatch for {name}: {actual}")
        resolved[name] = path
        verified_hashes[name] = actual

    dual_rows = load_rows(resolved["dual_predictions"])
    physical_rows = load_rows(resolved["physical_predictions"])
    if len(dual_rows) != len(physical_rows):
        raise RuntimeError("prediction files have different record counts")
    physical_by_id = {row["record_id"]: row for row in physical_rows}
    if len(physical_by_id) != len(physical_rows):
        raise RuntimeError("physical predictions contain duplicate record IDs")
    if {row["record_id"] for row in dual_rows} != set(physical_by_id):
        raise RuntimeError("prediction files do not contain the same record IDs")

    counts = {truth: Counter() for truth in sorted(NOISE_FAMILY_ROUTES)}
    decisions: list[dict[str, object]] = []
    for dual in dual_rows:
        truth = dual["truth"]
        physical = physical_by_id[dual["record_id"]]
        if physical["truth"] != truth:
            raise RuntimeError(f"truth mismatch for {dual['record_id']}")
        if truth not in NOISE_FAMILY_ROUTES:
            continue
        decision = select_noise_family_route(
            dual_route=dual["route"],
            dual_accepted=parse_bool(dual["accepted"]),
            physical_route=physical["final_route"],
        )
        correct = decision.accepted and decision.route == truth
        wrong = decision.accepted and decision.route != truth
        status = "correct" if correct else "wrong" if wrong else "abstain"
        counts[truth][status] += 1
        decisions.append(
            {
                "record_id": dual["record_id"],
                "truth": truth,
                "dual_route": dual["route"],
                "dual_accepted": dual["accepted"].lower(),
                "physical_route": physical["final_route"],
                "final_route": decision.route or "abstain",
                "status": status,
                "reason": decision.reason,
            }
        )

    metrics: dict[str, dict[str, int | float]] = {}
    gate_passed = True
    for truth, counter in counts.items():
        total = sum(counter.values())
        accepted = counter["correct"] + counter["wrong"]
        coverage = accepted / total
        metrics[truth] = {
            "records": total,
            "correct": counter["correct"],
            "wrong": counter["wrong"],
            "abstain": counter["abstain"],
            "coverage": coverage,
            "accepted_accuracy": counter["correct"] / accepted if accepted else 0.0,
        }
        gate_passed &= counter["wrong"] <= config["gates"]["maximum_wrong_per_class"]
        gate_passed &= coverage >= config["gates"]["minimum_coverage_per_class"]

    output_dir = ROOT / Path(config["outputs"]["result"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = ROOT / config["outputs"]["decisions"]
    with decisions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)

    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if gate_passed else "fail",
        "gpu_count": 0,
        "official_or_internal_test_accessed": False,
        "all_development_records_aligned": len(dual_rows),
        "noise_family_records_evaluated": len(decisions),
        "decision_rule": config["decision_rule"],
        "gates": config["gates"],
        "metrics": metrics,
        "verified_artifact_sha256": verified_hashes,
        "decisions_sha256": sha256(decisions_path),
        "limitation": config["limitations"],
    }
    result_path = ROOT / config["outputs"]["result"]
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
