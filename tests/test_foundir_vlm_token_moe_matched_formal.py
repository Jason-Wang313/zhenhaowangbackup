import unittest
from pathlib import Path

from smoke.compare_foundir_vlm_token_moe_matched_formal_v1 import paired_summary
from smoke.train_foundir_vlm_token_moe_matched_formal_v1_ddp import (
    LABELS,
    aggregate_rows,
    deterministic_evaluation_seed,
    load_formal_config,
)


class MatchedFormalProtocolTest(unittest.TestCase):
    def test_frozen_config_loads(self):
        config = load_formal_config(
            Path("configs/foundir_vlm_token_moe_matched_formal_v1.json")
        )
        self.assertEqual(config["training"]["optimizer_steps"], 1000)
        self.assertEqual(config["evaluation"]["holdout_steps"], [1000])
        self.assertTrue(config["training"]["qualification_checkpoint_forbidden"])

    def test_evaluation_seed_is_deterministic_and_record_specific(self):
        first = deterministic_evaluation_seed(17, "a")
        self.assertEqual(first, deterministic_evaluation_seed(17, "a"))
        self.assertNotEqual(first, deterministic_evaluation_seed(17, "b"))

    def test_aggregate_reports_every_degradation(self):
        rows = []
        for index, label in enumerate(LABELS):
            rows.append(
                {
                    "label": label,
                    "input_psnr": 20.0 + index,
                    "output_psnr": 21.0 + index,
                    "input_lpips": 0.2,
                    "output_lpips": 0.1,
                    "route_entropy": 0.5,
                    "expert_weights": [0.25, 0.75],
                    "vlm_adjustment_l1": 0.2,
                }
            )
        result = aggregate_rows(rows)
        self.assertEqual(set(result["by_degradation"]), set(LABELS))
        self.assertAlmostEqual(result["psnr_gain_over_input"], 1.0)
        self.assertAlmostEqual(result["lpips_improvement_over_input"], 0.1)

    def test_paired_summary_uses_vlm_minus_image_signs(self):
        image_rows = []
        vlm_rows = []
        for label_index, label in enumerate(LABELS):
            for item in range(2):
                identity = f"{label}_{item}"
                image_rows.append(
                    {
                        "record_id": identity,
                        "label": label,
                        "output_psnr": 30.0 + label_index,
                        "output_lpips": 0.10,
                    }
                )
                vlm_rows.append(
                    {
                        "record_id": identity,
                        "label": label,
                        "output_psnr": 30.2 + label_index,
                        "output_lpips": 0.09,
                    }
                )
        result = paired_summary(image_rows, vlm_rows)
        self.assertTrue(result["primary_hypothesis_pass"])
        self.assertAlmostEqual(result["overall"]["mean_psnr_delta_db_vlm_minus_image"], 0.2)
        self.assertAlmostEqual(result["overall"]["mean_lpips_delta_vlm_minus_image"], -0.01)


if __name__ == "__main__":
    unittest.main()
