#!/usr/bin/env python3
"""Run the bounded training-only feasibility pilot for the v2 generator."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    assign_source_splits,
    load_source_inventory,
    sha256,
    write_new_or_verify,
)
from lowlight_restoration.multidegradation_visibility_v2 import (
    build_candidate_record,
    evaluate_candidate_visibility,
    load_visibility_anchors,
    ordered_source_pool,
    select_visible_candidate,
)
from lowlight_restoration.qwen_multidegradation import DEGRADATIONS


def _run_slot(payload: tuple[object, ...]) -> dict[str, object]:
    (
        split,
        combination_index,
        within_index,
        combo,
        sources,
        thresholds,
        policy,
        seed,
        data_root,
    ) = payload
    return select_visible_candidate(
        split=str(split),
        composition_size=3,
        combination_index=int(combination_index),
        within_combination_index=int(within_index),
        combo=combo,
        sources=sources,
        thresholds=thresholds,
        policy=policy,
        seed=int(seed),
        data_root=str(data_root),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/qwen_multidegradation_visibility_v2.json",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    config_path = args.config.resolve()
    config_path.relative_to(ROOT.resolve())
    config = json.loads(config_path.read_text())
    source_path = (ROOT / config["source_manifest"]).resolve()
    anchor_path = (ROOT / config["visibility_anchor"]).resolve()
    if sha256(source_path) != config["source_manifest_sha256"]:
        raise RuntimeError("v2 source manifest changed")
    if sha256(anchor_path) != config["visibility_anchor_sha256"]:
        raise RuntimeError("v2 visibility anchor changed")
    inventory = load_source_inventory(source_path)
    assignments = assign_source_splits(inventory, seed=int(config["seed"]))
    pilot = config["pilot"]
    split = str(pilot["split"])
    policy = config["visibility_policy"]
    thresholds = load_visibility_anchors(
        anchor_path,
        metric=str(policy["primary_metric"]),
        quantile=str(policy["anchor_quantile"]),
    )
    anchor_multiplier = float(policy.get("anchor_multiplier", 1.0))
    if not 0 < anchor_multiplier <= 1:
        raise RuntimeError("v2 anchor multiplier must be in (0, 1]")
    thresholds = {
        key: value * anchor_multiplier for key, value in thresholds.items()
    }
    tasks: list[tuple[object, ...]] = []
    combinations = list(itertools.combinations(DEGRADATIONS, 3))
    for combination_index, combo in enumerate(combinations):
        sources = ordered_source_pool(
            assignments[split],
            seed=int(config["seed"]),
            split=split,
            combo=combo,
        )
        for within_index in range(int(pilot["records_per_combination"])):
            tasks.append(
                (
                    split,
                    combination_index,
                    within_index,
                    combo,
                    sources,
                    thresholds,
                    policy,
                    config["seed"],
                    config["data_root"],
                )
            )
    if len(tasks) != int(pilot["expected_records"]):
        raise RuntimeError("v2 pilot task cardinality changed")
    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(_run_slot, tasks), start=1):
            results.append(result)
            if index % 5 == 0 or index == len(tasks):
                print(f"QWEN_VISIBILITY_V2_PILOT_PROGRESS {index}/{len(tasks)}", flush=True)
    attempts = np.asarray([int(result["attempts"]) for result in results], dtype=np.float64)
    mean_attempts = float(attempts.mean())
    p95_attempts = float(np.quantile(attempts, 0.95))
    checks = {
        "all_slots_accepted": len(results) == int(pilot["expected_records"])
        and all(bool(result["accepted"]) for result in results),
        "mean_attempts": mean_attempts <= float(pilot["maximum_mean_attempts"]),
        "p95_attempts": p95_attempts <= float(pilot["maximum_p95_attempts"]),
        "all_components_visible": all(
            bool(component["pass"])
            for result in results
            for component in result["visibility"]
        ) and all(bool(result["accepted"]) for result in results),
    }
    output = {
        "schema_version": 1,
        "status": "pass" if all(checks.values()) else "failed",
        "automatic_gate_pass": all(checks.values()),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "source_manifest_sha256": config["source_manifest_sha256"],
        "visibility_anchor_sha256": config["visibility_anchor_sha256"],
        "records": len(results),
        "accepted_records": sum(bool(result["accepted"]) for result in results),
        "components": sum(len(result["visibility"]) for result in results),
        "mean_attempts": mean_attempts,
        "p95_attempts": p95_attempts,
        "maximum_attempts": int(attempts.max()),
        "checks": checks,
        "results": results,
    }
    output_path = (ROOT / pilot["output"]).resolve()
    output_path.relative_to(ROOT.resolve())
    write_new_or_verify(
        output_path,
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(
        ("QWEN_VISIBILITY_V2_PILOT_PASS " if all(checks.values()) else "QWEN_VISIBILITY_V2_PILOT_FAILED ")
        + json.dumps(
            {
                "records": len(results),
                "mean_attempts": mean_attempts,
                "p95_attempts": p95_attempts,
                "maximum_attempts": int(attempts.max()),
                "checks": checks,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    raise SystemExit(0 if all(checks.values()) else 2)


if __name__ == "__main__":
    main()
