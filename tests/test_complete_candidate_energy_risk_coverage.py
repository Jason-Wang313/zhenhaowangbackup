import unittest

import torch

from smoke.evaluate_foundir_complete_candidate_energy_risk_coverage_v1_ddp import (
    split_source_groups,
    threshold_metrics,
)


class CompleteCandidateEnergyRiskCoverageTest(unittest.TestCase):
    def test_source_split_is_disjoint_and_stable(self):
        rows = [{"record_id": str(i), "source_group": str(i % 6)} for i in range(30)]
        first = split_source_groups(rows, 17)
        second = split_source_groups(rows, 17)
        self.assertEqual(first, second)
        self.assertFalse({r["source_group"] for r in first[0]} & {r["source_group"] for r in first[1]})

    def test_threshold_can_recover_safe_fallback(self):
        record = {
            "margin": torch.tensor([[.5, .1]]),
            "best": torch.tensor([[0, 0]]),
            "local_mse": torch.tensor([[[.01, .01]], [[.02, .02]]]),
        }
        accepted = threshold_metrics([record], 0.0, 1)
        safe = threshold_metrics([record], float("inf"), 1)
        self.assertEqual(accepted["accepted_fraction"], 1.0)
        self.assertEqual(safe["accepted_fraction"], 0.0)
        self.assertGreater(accepted["psnr"], safe["psnr"])


if __name__ == "__main__":
    unittest.main()
