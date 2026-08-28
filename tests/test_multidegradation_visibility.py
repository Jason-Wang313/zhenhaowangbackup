from __future__ import annotations

import unittest

import numpy as np

from lowlight_restoration.multidegradation_data import (
    ROOT,
    generate_record,
    read_jsonl,
)
from lowlight_restoration.multidegradation_visibility import (
    audit_record_components,
    binary_auc,
    image_delta_metrics,
)
from lowlight_restoration.multidegradation_visibility_v2 import physical_order


class MultiDegradationVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records = read_jsonl(ROOT / "manifests/qwen_multidegradation_pilot_v1.jsonl")
        cls.records = records
        cls.record = next(
            record
            for record in records
            if record["split"] == "validation" and record["composition_size"] == 3
        )

    def test_component_audit_supports_every_degraded_composition_size(self):
        for size in (1, 2, 3):
            record = next(
                item
                for item in self.records
                if item["split"] == "validation"
                and item["composition_size"] == size
                and "haze" not in item["degradations"]
            )
            rows, full_sha = audit_record_components(
                record,
                predicted_degradations=tuple(record["degradations"]),
                require_frozen_output=False,
            )
            self.assertEqual(len(rows), size)
            self.assertEqual(len(full_sha), 64)
            self.assertEqual(
                [row["degradation"] for row in rows],
                record["application_order"],
            )

    def test_ordered_subset_render_is_deterministic_and_distinct(self):
        order = tuple(self.record["application_order"])
        full, full_sha = generate_record(self.record)
        repeated, repeated_sha = generate_record(
            self.record, application_order=order
        )
        without_first, without_sha = generate_record(
            self.record, application_order=order[1:]
        )
        self.assertTrue(np.array_equal(full, repeated))
        self.assertEqual(full_sha, repeated_sha)
        self.assertNotEqual(full_sha, without_sha)
        self.assertFalse(np.array_equal(full, without_first))

    def test_reordered_or_duplicate_intervention_is_rejected(self):
        order = tuple(self.record["application_order"])
        with self.assertRaises(ValueError):
            generate_record(self.record, application_order=tuple(reversed(order)))
        with self.assertRaises(ValueError):
            generate_record(self.record, application_order=(order[0], order[0]))

    def test_image_delta_metrics_have_expected_limits(self):
        zeros = np.zeros((16, 16, 3), dtype=np.uint8)
        tens = np.full((16, 16, 3), 10, dtype=np.uint8)
        identical = image_delta_metrics(zeros, zeros)
        changed = image_delta_metrics(tens, zeros)
        self.assertEqual(identical["rgb_rmse_0_255"], 0.0)
        self.assertEqual(identical["ssim"], 1.0)
        self.assertAlmostEqual(changed["rgb_rmse_0_255"], 10.0)
        self.assertEqual(changed["changed_pixel_fraction_gt_2"], 1.0)

    def test_auc_is_tie_aware(self):
        self.assertEqual(binary_auc([0.0, 1.0], [False, True]), 1.0)
        self.assertEqual(binary_auc([1.0, 0.0], [False, True]), 0.0)
        self.assertEqual(binary_auc([1.0, 1.0], [False, True]), 0.5)

    def test_physical_order_is_a_complete_ordered_subset(self):
        policy = ("haze", "blur", "rain", "low_light", "noise")
        self.assertEqual(
            physical_order(("noise", "blur", "rain"), policy),
            ("blur", "rain", "noise"),
        )
        with self.assertRaises(ValueError):
            physical_order(("blur", "unknown"), policy)


if __name__ == "__main__":
    unittest.main()
