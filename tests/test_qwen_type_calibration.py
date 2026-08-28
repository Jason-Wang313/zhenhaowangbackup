from __future__ import annotations

import unittest

from lowlight_restoration.qwen_multidegradation import DEGRADATIONS
from lowlight_restoration.qwen_type_calibration import (
    build_training_source_partition,
    calibrated_level,
    fit_type_calibration,
    validate_training_source_partition,
)


class QwenTypeCalibrationTests(unittest.TestCase):
    def test_source_partition_is_disjoint_and_reproducible(self):
        records = []
        for dataset in ("a", "b"):
            for source_index in range(20):
                source_id = f"{dataset}_{source_index:02d}"
                for level in range(1, 6):
                    records.append(
                        {
                            "record_id": f"{source_id}_{level}",
                            "split": "train",
                            "source_id": source_id,
                            "source_dataset": dataset,
                            "composition_size": 0 if level == 1 else 1,
                            "levels": {} if level == 1 else {"blur": level},
                        }
                    )
        partition = build_training_source_partition(
            records,
            manifest_sha256="a" * 64,
            seed=7,
            calibration_fraction_numerator=1,
            calibration_fraction_denominator=2,
            training_image_batch_multiple=10,
            minimum_calibration_type_level_records=1,
        )
        calibration = set(partition["calibration_source_ids"])
        gradient = set(partition["gradient_source_ids"])
        self.assertFalse(calibration & gradient)
        self.assertEqual(len(calibration | gradient), 40)
        self.assertEqual(
            validate_training_source_partition(
                partition, records, manifest_sha256="a" * 64
            ),
            partition,
        )

    def test_affine_calibration_corrects_a_shared_l1_bias(self):
        rows = []
        for degradation in DEGRADATIONS:
            for level in range(1, 6):
                for _ in range(8):
                    scores = [-1.0] * 5
                    scores[level - 1] = 2.0
                    scores[0] += 2.5
                    rows.append(
                        {
                            "degradation": degradation,
                            "true_level": level,
                            "candidate_scores": scores,
                        }
                    )
        calibration = fit_type_calibration(rows, maximum_iterations=80)
        self.assertLess(calibration["calibrated_nll"], calibration["raw_nll"])
        level, adjusted = calibrated_level("blur", [-1.0, -1.0, 2.0, -1.0, -1.0], calibration)
        self.assertEqual(level, 3)
        self.assertEqual(len(adjusted), 5)

    def test_calibration_rejects_missing_type_level_support(self):
        with self.assertRaisesRegex(ValueError, "cover every type and level"):
            fit_type_calibration(
                [
                    {
                        "degradation": "blur",
                        "true_level": 1,
                        "candidate_scores": [1, 0, 0, 0, 0],
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
