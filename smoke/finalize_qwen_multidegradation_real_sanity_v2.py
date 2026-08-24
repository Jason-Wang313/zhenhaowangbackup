#!/usr/bin/env python3
"""Freeze the visually screened 25-image real-sanity v2 pool before Qwen inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify

MANIFEST = ROOT / "manifests/qwen_multidegradation_real_sanity_v2.json"
CANDIDATES = ROOT / "manifests/qwen_multidegradation_real_sanity_v2_candidates.json"
BLUR_SELECTION = ROOT / "manifests/qwen_multidegradation_realblur_after_selection_v2b.json"
V1 = ROOT / "manifests/qwen_multidegradation_real_sanity_v1.json"

HAZE_IDS = (
    "rtts_tree_0020",
    "rtts_tree_0060",
    "rtts_tree_0120",
    "rtts_tree_0160",
    "rtts_tree_0380",
)
LOWLIGHT_IDS = ("dicm_06", "dicm_12", "dicm_18", "dicm_23", "dicm_29")


def selected(records: list[dict[str, object]], ids: tuple[str, ...]) -> list[dict[str, object]]:
    by_id = {str(record["candidate_id"]): record for record in records}
    if any(candidate_id not in by_id for candidate_id in ids):
        raise RuntimeError("a predeclared v2 candidate is missing")
    return [dict(by_id[candidate_id]) for candidate_id in ids]


def validate_records(records: list[dict[str, object]]) -> dict[str, object]:
    types = ("blur", "noise", "rain", "haze", "low_light")
    if len(records) != 25 or len({record["record_id"] for record in records}) != 25:
        raise RuntimeError("real-sanity v2 must contain 25 unique record IDs")
    counts = {name: sum(record["expected_primary_type"] == name for record in records) for name in types}
    if counts != {name: 5 for name in types}:
        raise RuntimeError(f"real-sanity v2 type counts changed: {counts}")
    hashes = set()
    for record in records:
        path = ROOT / str(record["path"])
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"real-sanity v2 file hash changed: {path}")
        if record["sha256"] in hashes:
            raise RuntimeError("real-sanity v2 contains an exact duplicate")
        hashes.add(record["sha256"])
        if record["severity_ground_truth"] is not None:
            raise RuntimeError("real-sanity v2 must not invent absolute severity truth")
    return {
        "records": 25,
        "type_counts": counts,
        "exact_unique": 25,
        "foundir_used": False,
        "qwen_inference_before_freeze": False,
        "severity_annotation": "none_manual_qualitative_only",
    }


def main() -> None:
    candidates_payload = json.loads(CANDIDATES.read_text())
    blur_payload = json.loads(BLUR_SELECTION.read_text())
    v1_payload = json.loads(V1.read_text())
    if candidates_payload.get("qwen_inference_performed") is not False:
        raise RuntimeError("candidate screen is not pre-inference")
    if blur_payload.get("status") != "preinference_selected_blurred_after_images":
        raise RuntimeError("corrected blur selection is not frozen pre-inference evidence")

    candidate_records = candidates_payload["records"]
    groups: dict[str, list[dict[str, object]]] = {
        "blur": [dict(record) for record in blur_payload["selected_records"]],
        "noise": [
            dict(record)
            for record in candidate_records
            if record["expected_primary_type"] == "noise"
        ],
        "haze": selected(candidate_records, HAZE_IDS),
        "low_light": selected(candidate_records, LOWLIGHT_IDS),
        "rain": [
            dict(record)
            for record in v1_payload["records"]
            if record["expected_primary_type"] == "rain"
        ],
    }
    records = []
    for degradation in ("blur", "noise", "rain", "haze", "low_light"):
        if len(groups[degradation]) != 5:
            raise RuntimeError(f"selected v2 {degradation} count changed")
        for index, source in enumerate(groups[degradation]):
            record = {
                **source,
                "record_id": f"real_v2_{degradation}_{index:02d}",
                "expected_primary_type": degradation,
                "severity_ground_truth": None,
                "annotation_kind": "visually_screened_dataset_category_qualitative_not_exhaustive",
            }
            record.pop("candidate_id", None)
            record.pop("row_index", None)
            record.pop("screening_scene_burst", None)
            records.append(record)
    validation = validate_records(records)
    payload = {
        "schema_version": 2,
        "status": "frozen_before_qwen_inference",
        "scientific_role": "manual_qualitative_real_domain_sanity_not_quantitative_severity_test",
        "foundir_used": False,
        "absolute_real_severity_truth": "unavailable_and_not_invented",
        "selection_policy": {
            "blur": "blurred_after_side_proven_by_transform_prompt_then_strongest_sharpness_reduction_per_distinct_scene_burst",
            "noise": "five_distinct_high_iso_official_SIDD_Small_real_noisy_sRGB_scenes",
            "rain": "five_preinference_visually_accepted_RealRain_1k_inputs_reused_from_v1",
            "haze": "five_preinference_visually_obvious_diverse_RTTs_haze_scenes",
            "low_light": "five_preinference_visually_obvious_distinct_DICM_underexposed_scenes",
        },
        "manual_rubric": {
            "acceptable": [
                "adjacent_L3_L4_ambiguity",
                "legitimate_additional_visible_degradations",
            ],
            "failure": [
                "obvious_primary_type_omission",
                "obvious_unrelated_type_false_positive",
                "severity_contradicts_visible_strength",
            ],
            "review_scope": "approximately_correct_type_and_strength_only_no_absolute_real_L2_L5_score",
        },
        "rejected_screening_evidence": {
            "manifest": V1.relative_to(ROOT).as_posix(),
            "sha256": sha256(V1),
            "status": "rejected_preinference_input_screening",
            "reasons": [
                "blur_used_sharp_before_side_and_near_duplicate_burst_frames",
                "noise_was_scanned_engineering_drawings_not_camera_noise",
                "several_haze_inputs_were_not_visibly_hazy",
                "several_low_light_inputs_were_not_clearly_underexposed",
            ],
        },
        "screening_evidence": {
            "candidate_manifest": CANDIDATES.relative_to(ROOT).as_posix(),
            "candidate_manifest_sha256": sha256(CANDIDATES),
            "corrected_blur_selection": BLUR_SELECTION.relative_to(ROOT).as_posix(),
            "corrected_blur_selection_sha256": sha256(BLUR_SELECTION),
        },
        "validation": validation,
        "records": records,
    }
    digest = write_new_or_verify(
        MANIFEST, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(f"QWEN_MULTIDEGRADATION_REAL_SANITY_V2_FREEZE_PASS records=25 sha256={digest}")


if __name__ == "__main__":
    main()
