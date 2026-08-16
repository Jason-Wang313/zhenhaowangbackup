from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from lowlight_restoration.data import PairRecord
from lowlight_restoration.degradation_assessment import (
    SUPPORTED_DEGRADATIONS,
    assessment_gate,
    canonical_condition_json,
    evaluate_severity,
    fit_scalar_centroids,
    parse_protocol,
    predict_scalar_centroid,
    require_task_ready,
    strict_condition_payload,
    validate_source_isolation,
)
from predict_qwen_lowlight_fewshot import select_unique_source_records


def protocol(degradation: str, mode: str = "visual_few_shot") -> dict[str, object]:
    return {
        "degradation": degradation,
        "calibration_mode": mode,
        "calibration_split": "train",
        "validation_split": "validation",
        "protected_splits": ["test", "foundir_test"],
        "baseline_name": f"{degradation}_simple_baseline",
        "gates": {
            "structured_output_rate": 1.0,
            "require_all_predicted_levels": True,
            "minimum_per_level_recall": 0.4,
            "minimum_accuracy": 0.6,
            "require_baseline_match": True,
        },
    }


@dataclass(frozen=True)
class Record:
    source_id: str
    degraded_sha256: str


class DegradationAssessmentTests(unittest.TestCase):
    @staticmethod
    def qwen_records() -> list[PairRecord]:
        return [
            PairRecord(
                pair_id=f"source_{source:02d}_L{level}",
                source_id=f"source_{source:02d}",
                clean_path=Path("clean.png"),
                degraded_path=Path("degraded.png"),
                project_level=f"L{level}",
                seed=source,
                split="validation",
                clean_sha256="a" * 64,
                degraded_sha256=f"{source * 5 + level:064x}",
            )
            for source in range(30)
            for level in range(1, 6)
        ]

    def test_every_task_forbids_uncalibrated_severity_modes(self) -> None:
        for degradation in SUPPORTED_DEGRADATIONS:
            with self.subTest(degradation=degradation):
                parsed = parse_protocol(protocol(degradation))
                self.assertEqual(parsed.degradation, degradation)
                for forbidden in ("zero_shot", "rubric_only", "none"):
                    with self.assertRaises(ValueError):
                        parse_protocol(protocol(degradation, forbidden))

    def test_other_four_tasks_remain_blocked_until_individually_calibrated(self) -> None:
        require_task_ready("low_light")
        for degradation in ("noise", "blur", "haze", "rain"):
            with self.subTest(degradation=degradation), self.assertRaises(RuntimeError):
                require_task_ready(degradation)

    def test_generic_strict_schema_covers_all_five_tasks(self) -> None:
        for degradation in SUPPORTED_DEGRADATIONS:
            raw = canonical_condition_json(degradation, 5)
            self.assertEqual(strict_condition_payload(raw, degradation), 5)
            with self.assertRaises(ValueError):
                strict_condition_payload(raw, "low_light" if degradation != "low_light" else "noise")

    def test_protocol_requires_disjoint_splits_baseline_and_nonzero_recall(self) -> None:
        same_split = protocol("low_light")
        same_split["validation_split"] = "train"
        with self.assertRaises(ValueError):
            parse_protocol(same_split)
        no_baseline = protocol("low_light")
        no_baseline["baseline_name"] = "none"
        with self.assertRaises(ValueError):
            parse_protocol(no_baseline)
        no_recall = protocol("low_light")
        no_recall["gates"]["minimum_per_level_recall"] = 0.0  # type: ignore[index]
        with self.assertRaises(ValueError):
            parse_protocol(no_recall)

    def test_source_and_hash_leakage_are_rejected(self) -> None:
        calibration = [Record("train_a", "a" * 64)]
        validation = [Record("validation_b", "b" * 64)]
        validate_source_isolation(calibration, validation)
        with self.assertRaises(ValueError):
            validate_source_isolation(calibration, [Record("train_a", "c" * 64)])
        with self.assertRaises(ValueError):
            validate_source_isolation(calibration, [Record("validation_c", "a" * 64)])

    def test_validation_selection_is_balanced_deterministic_and_source_unique(self) -> None:
        first = select_unique_source_records(
            self.qwen_records(), split="validation", sample_count=25, seed=31
        )
        second = select_unique_source_records(
            self.qwen_records(), split="validation", sample_count=25, seed=31
        )
        self.assertEqual([record.pair_id for record in first], [record.pair_id for record in second])
        self.assertEqual(len({record.source_id for record in first}), 25)
        self.assertEqual(
            {level: sum(record.level_index == level for record in first) for level in range(1, 6)},
            {level: 5 for level in range(1, 6)},
        )

    def test_scalar_baseline_is_train_fitted_and_deterministic(self) -> None:
        centroids = fit_scalar_centroids(
            [(level, level * 10.0 + offset) for level in range(1, 6) for offset in (-1.0, 1.0)]
        )
        self.assertEqual(centroids, {level: level * 10.0 for level in range(1, 6)})
        self.assertEqual(predict_scalar_centroid(48.0, centroids), 5)
        self.assertEqual(predict_scalar_centroid(15.0, centroids), 1)

    def test_gate_rejects_class_collapse_even_with_valid_json(self) -> None:
        collapsed = evaluate_severity(
            [
                {
                    "true_level": true_level,
                    "predicted_level": 4,
                    "valid_structured_output": True,
                }
                for true_level in range(1, 6)
                for _ in range(5)
            ]
        )
        baseline = evaluate_severity(
            [
                {
                    "true_level": level,
                    "predicted_level": level,
                    "valid_structured_output": True,
                }
                for level in range(1, 6)
            ]
        )
        gate = assessment_gate(collapsed, baseline, parse_protocol(protocol("low_light")))
        self.assertFalse(gate["automatic_gate_pass"])
        self.assertFalse(gate["all_predicted_levels_pass"])
        self.assertFalse(gate["minimum_per_level_recall_pass"])

    def test_invalid_output_is_finite_evidence_and_fails_the_gate(self) -> None:
        records = [
            {
                "true_level": level,
                "predicted_level": level if level != 5 else None,
                "valid_structured_output": level != 5,
            }
            for level in range(1, 6)
        ]
        metrics = evaluate_severity(records)
        baseline = evaluate_severity(
            [
                {
                    "true_level": level,
                    "predicted_level": level,
                    "valid_structured_output": True,
                }
                for level in range(1, 6)
            ]
        )
        self.assertEqual(metrics["invalid_outputs"], 1)
        self.assertEqual(metrics["severity_mean_absolute_error"], 4.0)
        gate = assessment_gate(metrics, baseline, parse_protocol(protocol("low_light")))
        self.assertFalse(gate["structured_output_pass"])
        self.assertFalse(gate["automatic_gate_pass"])

    def test_gate_requires_matching_the_task_baseline(self) -> None:
        perfect_records = [
            {
                "true_level": level,
                "predicted_level": level,
                "valid_structured_output": True,
            }
            for level in range(1, 6)
            for _ in range(5)
        ]
        metrics = evaluate_severity(perfect_records)
        gate = assessment_gate(metrics, metrics, parse_protocol(protocol("low_light")))
        self.assertTrue(gate["automatic_gate_pass"])
        self.assertTrue(gate["all_predicted_levels_pass"])


if __name__ == "__main__":
    unittest.main()
