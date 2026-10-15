#!/usr/bin/env python3
"""Aggregate the frozen all-in-one development rehearsal with absolute SOW bounds."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/foundir_final_eleven_all_in_one_development_rehearsal_v2.csv"
OUTPUT = ROOT / "checkpoints/foundir_final_eleven_all_in_one_development_rehearsal_v2"
RESULT = OUTPUT / "result.json"
TERMINAL = OUTPUT / "terminal_result.json"
SOW = {
    "01Blur": (25.9, 0.18), "05Noise": (38.9, 0.10), "08Haze": (22.1, 0.19),
    "10Rain": (34.2, 0.08), "14Lowlight": (18.9, 0.28),
}


def digest(*parts: object) -> int:
    return int(hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def bootstrap(rows: list[dict], category: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_group"]].append(row)
    groups = sorted(grouped)
    values = np.asarray([[np.mean([r["restored"][metric] for r in grouped[g]]) for metric in ("psnr", "lpips")] for g in groups])
    rng = np.random.default_rng(digest(2026092004, category))
    draws = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {
        "source_groups": len(groups), "resamples": 5000, "confidence": 0.95,
        "group_balanced_mean": {"psnr": float(values[:, 0].mean()), "lpips": float(values[:, 1].mean())},
        "lower": {"psnr": float(np.quantile(draws[:, 0], .025)), "lpips": float(np.quantile(draws[:, 1], .025))},
        "upper": {"psnr": float(np.quantile(draws[:, 0], .975)), "lpips": float(np.quantile(draws[:, 1], .975))},
    }


def means(rows: list[dict], role: str = "restored") -> dict:
    return {metric: float(np.mean([row[role][metric] for row in rows])) for metric in ("psnr", "lpips")}


def main() -> None:
    if not RESULT.is_file():
        raise RuntimeError("GPU rehearsal result is missing")
    with MANIFEST.open(newline="") as handle:
        manifest = {row["record_id"]: row for row in csv.DictReader(handle)}
    sidecars = []
    for record_id, row in manifest.items():
        payload = json.loads((OUTPUT / "per_record" / f"{record_id}.json").read_text())
        payload["routing_status"] = row["routing_status"]
        sidecars.append(payload)
    by_category, by_status = defaultdict(list), defaultdict(list)
    for row in sidecars:
        by_category[row["category"]].append(row); by_status[row["routing_status"]].append(row)
    categories = {}
    sow_checks = {}
    for category in sorted(by_category):
        rows = by_category[category]
        restored, input_values, bounds = means(rows), means(rows, "input"), bootstrap(rows, category)
        item = {
            "records": len(rows), "input": input_values, "restored": restored,
            "delta": {"psnr_db": restored["psnr"] - input_values["psnr"], "lpips": restored["lpips"] - input_values["lpips"]},
            "routing_status_counts": {key: sum(row["routing_status"] == key for row in rows) for key in ("correct", "wrong", "abstain")},
            "absolute_source_group_bootstrap": bounds,
        }
        if category in SOW:
            psnr_min, lpips_max = SOW[category]
            checks = {
                "mean_psnr": restored["psnr"] >= psnr_min,
                "mean_lpips": restored["lpips"] <= lpips_max,
                "bootstrap_psnr_lower": bounds["lower"]["psnr"] >= psnr_min,
                "bootstrap_lpips_upper": bounds["upper"]["lpips"] <= lpips_max,
            }
            item["sow"] = {
                "psnr_minimum": psnr_min, "lpips_maximum": lpips_max,
                "mean_margin": {
                    "psnr_db": restored["psnr"] - psnr_min,
                    "psnr_percent": 100 * (restored["psnr"] - psnr_min) / psnr_min,
                    "lpips": lpips_max - restored["lpips"],
                    "lpips_percent": 100 * (lpips_max - restored["lpips"]) / lpips_max,
                },
                "checks": checks,
            }
            sow_checks[category] = checks
        categories[category] = item
    class_means = [categories[key]["restored"] for key in sorted(categories)]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "interpretation": "post_rehearsal_low_light_fallback_not_independent_lockbox",
        "records": len(sidecars),
        "overall_class_balanced": {metric: float(np.mean([value[metric] for value in class_means])) for metric in ("psnr", "lpips")},
        "routing_status_metrics": {key: {"records": len(rows), **means(rows)} for key, rows in sorted(by_status.items())},
        "categories": categories,
        "sow_mean_gate_pass": all(value["mean_psnr"] and value["mean_lpips"] for value in sow_checks.values()),
        "sow_robust_bootstrap_gate_pass": all(all(value.values()) for value in sow_checks.values()),
        "sow_checks": sow_checks,
        "pipeline_frozen": True,
        "model_weight_updates": False,
        "development_pixels_accessed": True,
        "official_or_sealed_test_accessed": False,
        "abstention_output": "identity_input_unless_exposure_le_0.15_then_low_light_partial_restoration",
        "fallback_rule": {
            "maximum_exposure": 0.15,
            "maximum_side": 128,
            "expert_plan": ["low_light"],
        },
    }
    TERMINAL.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("FINAL_ELEVEN_ALL_IN_ONE_DEVELOPMENT_REHEARSAL " + json.dumps({
        "records": payload["records"], "sow_mean_gate_pass": payload["sow_mean_gate_pass"],
        "sow_robust_bootstrap_gate_pass": payload["sow_robust_bootstrap_gate_pass"],
        "overall_class_balanced": payload["overall_class_balanced"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
