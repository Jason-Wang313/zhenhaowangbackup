from __future__ import annotations

import unittest

from lowlight_restoration.qwen_real_validation import (
    augment_real_metrics,
    proxy_anchor_error_ev,
    proxy_level_from_luma_ratio,
    real_metric_gate_checks,
    select_balanced_proxy_candidates,
)


class QwenRealValidationTests(unittest.TestCase):
    def test_proxy_levels_use_nearest_log_ratio_anchor(self) -> None:
        self.assertEqual(proxy_level_from_luma_ratio(0.50), 2)
        self.assertEqual(proxy_level_from_luma_ratio(0.38), 3)
        self.assertEqual(proxy_level_from_luma_ratio(0.30), 4)
        self.assertEqual(proxy_level_from_luma_ratio(0.14), 5)
        self.assertAlmostEqual(proxy_anchor_error_ev(0.50, 2), 0.0)
        with self.assertRaises(ValueError):
            proxy_level_from_luma_ratio(0.0)

    def test_balanced_selection_is_deterministic_and_rejects_outliers(self) -> None:
        candidates = []
        for level, anchor in ((2, 0.50), (3, 0.38), (4, 0.30), (5, 0.14)):
            for index, scale in enumerate((1.0, 1.02, 0.98)):
                candidates.append(
                    {
                        "candidate_id": f"L{level}-{index}",
                        "luma_ratio": anchor * scale,
                    }
                )
        candidates.append({"candidate_id": "too-dark", "luma_ratio": 0.001})
        selected, evidence = select_balanced_proxy_candidates(
            candidates,
            maximum_per_level=2,
            minimum_per_level=2,
            maximum_anchor_error_ev=0.25,
            seed=7,
        )
        repeated, repeated_evidence = select_balanced_proxy_candidates(
            candidates,
            maximum_per_level=2,
            minimum_per_level=2,
            maximum_anchor_error_ev=0.25,
            seed=7,
        )
        self.assertEqual(selected, repeated)
        self.assertEqual(evidence, repeated_evidence)
        self.assertEqual(evidence["selected_per_level"], 2)
        self.assertEqual(evidence["selected_count"], 8)
        self.assertEqual(evidence["rejected_outside_anchor_tolerance"], 1)

    def test_real_ordinal_metrics_and_gate(self) -> None:
        rows = [
            {"true_level": 1, "predicted_level": 1},
            {"true_level": 2, "predicted_level": 3},
            {"true_level": 3, "predicted_level": 3},
            {"true_level": 4, "predicted_level": 5},
            {"true_level": 5, "predicted_level": None},
        ]
        base = {
            "valid_structured_output_rate": 0.8,
            "degradation_type_accuracy": 0.8,
            "severity_accuracy": 0.4,
            "severity_macro_f1": 0.4,
            "severity_mean_absolute_error": 1.0,
            "per_level": {
                str(level): {"accuracy": 0.5} for level in range(1, 6)
            },
        }
        metrics = augment_real_metrics(rows, base)
        self.assertEqual(metrics["within_one_level_accuracy"], 0.8)
        self.assertEqual(metrics["quadratic_weighted_kappa_valid_output_count"], 4)
        self.assertEqual(metrics["evaluated_severity_levels"], [1, 2, 3, 4, 5])
        self.assertGreater(metrics["severity_macro_f1"], 0.0)
        self.assertGreater(metrics["degradation_type_macro_f1"], 0.0)
        gate = {
            "required_levels": [1, 2, 3, 4, 5],
            "minimum_valid_structured_output_rate": 0.8,
            "minimum_degradation_type_accuracy": 0.8,
            "minimum_degradation_type_macro_f1": 0.75,
            "minimum_severity_accuracy": 0.4,
            "minimum_severity_macro_f1": 0.3,
            "maximum_severity_mean_absolute_error": 1.0,
            "minimum_within_one_level_accuracy": 0.8,
            "minimum_per_level_accuracy": 0.5,
            "minimum_quadratic_weighted_kappa": 0.5,
        }
        self.assertTrue(
            all(real_metric_gate_checks(metrics, gate, prefix="real").values())
        )

    def test_real_metrics_allow_a_stress_subset_with_absent_levels(self) -> None:
        rows = [
            {"true_level": 1, "predicted_level": 1},
            {"true_level": 4, "predicted_level": 5},
            {"true_level": 5, "predicted_level": 5},
        ]
        base = {
            "valid_structured_output_rate": 1.0,
            "degradation_type_accuracy": 1.0,
            "severity_accuracy": 2 / 3,
            "severity_mean_absolute_error": 1 / 3,
            "per_level": {
                "1": {"support": 1, "correct": 1, "accuracy": 1.0},
                "2": {"support": 0, "correct": 0, "accuracy": 0.0},
                "3": {"support": 0, "correct": 0, "accuracy": 0.0},
                "4": {"support": 1, "correct": 0, "accuracy": 0.0},
                "5": {"support": 1, "correct": 1, "accuracy": 1.0},
            },
        }
        metrics = augment_real_metrics(rows, base)
        self.assertEqual(metrics["evaluated_severity_levels"], [1, 4, 5])
        self.assertEqual(set(metrics["severity_f1"]), {"1", "4", "5"})


if __name__ == "__main__":
    unittest.main()
