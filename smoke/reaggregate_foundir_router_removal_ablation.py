#!/usr/bin/env python3
"""CPU-only ablation of the frozen FoundIR router evidence modules.

No model is loaded and no image is accessed.  The script only recombines
already-frozen development predictions to measure the effect of removing the
eleven-class head and/or the physical Noise/Low-light guard.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints/foundir_qwen_mixed_consensus_development_v3/predictions.csv"
DUAL = ROOT / "checkpoints/foundir_qwen_lowlight_noise_dual_nonlinear_v2/development_predictions.csv"
PHYSICAL = ROOT / "checkpoints/foundir_qwen_noise_lowlight_physical_guard_v3/full_development_predictions.csv"
FINAL = ROOT / "checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv"
OUT_JSON = ROOT / "reports/foundir_router_removal_ablation_2026-09-28.json"
OUT_CSV = ROOT / "reports/foundir_router_removal_ablation_2026-09-28.csv"
OUT_MD = ROOT / "reports/foundir_router_removal_ablation_2026-09-28.md"

NOISE_FAMILY = {"noise", "noise+low_light"}
SINGLETONS = {"blur", "noise", "haze", "rain", "low_light"}
FIVE_THRESHOLD = 0.925


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def indexed(path: Path) -> dict[str, dict[str, str]]:
    rows = read_rows(path)
    result = {row["record_id"]: row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"duplicate record IDs in {path}")
    return result


def summarize(
    truth: dict[str, str], routes: dict[str, str | None]
) -> dict[str, Any]:
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "correct": 0, "wrong": 0, "abstain": 0}
    )
    correct = wrong = abstain = 0
    for record_id, expected in truth.items():
        route = routes[record_id]
        per_class[expected]["records"] += 1
        if route is None:
            abstain += 1
            per_class[expected]["abstain"] += 1
        elif route == expected:
            correct += 1
            per_class[expected]["correct"] += 1
        else:
            wrong += 1
            per_class[expected]["wrong"] += 1
    accepted = correct + wrong
    return {
        "records": len(truth),
        "accepted": accepted,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "coverage": accepted / len(truth),
        "accepted_error_rate": wrong / accepted if accepted else None,
        "per_class": dict(sorted(per_class.items())),
    }


def build() -> dict[str, Any]:
    base = indexed(BASE)
    dual = indexed(DUAL)
    physical = indexed(PHYSICAL)
    final = indexed(FINAL)
    ids = set(base)
    if set(dual) != ids or set(final) != ids:
        raise RuntimeError("full-development prediction files are not aligned")

    truth = {record_id: row["truth"] for record_id, row in base.items()}
    if any(dual[record_id]["truth"] != expected for record_id, expected in truth.items()):
        raise RuntimeError("truth labels changed between frozen artifacts")

    full: dict[str, str | None] = {}
    minus_eleven: dict[str, str | None] = {}
    minus_physical: dict[str, str | None] = {}
    minus_both: dict[str, str | None] = {}
    requested_simplified: dict[str, str | None] = {}

    for record_id in sorted(ids):
        base_row = base[record_id]
        dual_row = dual[record_id]
        dual_route = dual_row["route"] if truthy(dual_row["accepted"]) else None
        physical_row = physical[record_id]
        physical_route = (
            None if physical_row["final_route"] == "abstain" else physical_row["final_route"]
        )

        final_route = final[record_id]["final_route"]
        full[record_id] = None if final_route == "abstain" else final_route

        # Strict removal of the eleven-class vote.  The retained dual route is
        # the frozen Atomic + Noise/Low-light nonlinear prediction.  The
        # physical agreement requirement is unchanged for the Noise family.
        if dual_route in NOISE_FAMILY:
            minus_eleven[record_id] = dual_route if physical_route == dual_route else None
        else:
            minus_eleven[record_id] = dual_route

        # Strict removal of the physical vote.  Noise-family routes use the
        # frozen nonlinear decision directly; all other final decisions stay
        # exactly as in the deployed router.
        if dual_route in NOISE_FAMILY:
            minus_physical[record_id] = dual_route
        else:
            minus_physical[record_id] = full[record_id]

        # Remove both modules: accept the frozen Atomic + nonlinear decision.
        minus_both[record_id] = dual_route

        # User-requested candidate: retain five-class, Atomic, and nonlinear
        # heads only.  Singleton routes require five-class confirmation;
        # supported mixed routes use the Atomic/nonlinear composition result.
        candidate = dual_route
        if candidate in SINGLETONS:
            five_ok = (
                base_row["five_proposal"] == candidate
                and float(base_row["five_confidence"]) >= FIVE_THRESHOLD
            )
            candidate = candidate if five_ok else None
        requested_simplified[record_id] = candidate

    variants = {
        "full_current_router": full,
        "minus_eleven_class_head": minus_eleven,
        "minus_physical_guard": minus_physical,
        "minus_eleven_and_physical": minus_both,
        "candidate_five_atomic_nonlinear_only": requested_simplified,
    }
    return {
        "schema_version": 1,
        "status": "cpu_frozen_prediction_ablation_complete",
        "date": "2026-09-28",
        "scope": {
            "records": len(ids),
            "gpu_used": False,
            "model_training": False,
            "new_inference": False,
            "official_or_sealed_images_accessed": False,
            "five_class_threshold": FIVE_THRESHOLD,
        },
        "variant_definitions": {
            "full_current_router": "Frozen deployed final router.",
            "minus_eleven_class_head": "Drop eleven-class agreement; retain Atomic/nonlinear route and physical agreement for Noise family.",
            "minus_physical_guard": "Drop physical agreement for Noise family; retain deployed decisions elsewhere.",
            "minus_eleven_and_physical": "Accept the frozen Atomic + nonlinear route directly.",
            "candidate_five_atomic_nonlinear_only": "For singleton routes require five-class confirmation of the Atomic/nonlinear route; accept supported mixed Atomic/nonlinear routes directly.",
        },
        "variants": {name: summarize(truth, routes) for name, routes in variants.items()},
        "limitations": [
            "This reuses the inspected 16,461-record development set and is not an independent lockbox result.",
            "The requested simplified candidate changes the consensus wiring; it is not a pure one-component deletion.",
            "No downstream restoration inference was rerun, so this ablation measures routing behavior only.",
        ],
    }


def write_outputs(payload: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "records", "accepted", "correct", "wrong", "abstain", "coverage", "accepted_error_rate"],
        )
        writer.writeheader()
        for name, stats in payload["variants"].items():
            writer.writerow({"variant": name, **{key: stats[key] for key in writer.fieldnames if key != "variant"}})

    lines = [
        "# FoundIR router removal ablation",
        "",
        "CPU-only reaggregation of 16,461 frozen development predictions; no training, inference, GPU, or official/sealed-image access.",
        "",
        "| Variant | Coverage | Correct | Wrong | Abstain | Accepted error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in payload["variants"].items():
        lines.append(
            f"| {name} | {stats['coverage']:.4%} | {stats['correct']:,} | {stats['wrong']:,} | {stats['abstain']:,} | {stats['accepted_error_rate']:.4%} |"
        )
    lines.extend(["", "## Definitions", ""])
    for name, definition in payload["variant_definitions"].items():
        lines.append(f"- `{name}`: {definition}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in payload["limitations"])
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    result = build()
    write_outputs(result)
    print(json.dumps({name: {key: stats[key] for key in ("coverage", "correct", "wrong", "abstain", "accepted_error_rate")} for name, stats in result["variants"].items()}, indent=2))
