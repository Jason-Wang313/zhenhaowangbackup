from __future__ import annotations

import math
import unittest

import torch

from lowlight_restoration.qwen_constrained import (
    mean_shifted_target_token_nll,
    select_minimum_mean_nll,
)


class QwenConstrainedScoringTests(unittest.TestCase):
    def test_mean_nll_uses_only_shifted_unmasked_targets(self) -> None:
        logits = torch.zeros((2, 5, 3), dtype=torch.float32)
        labels = torch.tensor(
            [
                [-100, -100, 1, 2, -100],
                [-100, 0, 1, 2, 0],
            ]
        )
        logits[0, 1, 1] = 4.0
        logits[0, 2, 2] = 4.0
        logits[1, 0, 0] = 2.0
        logits[1, 1, 1] = 2.0
        logits[1, 2, 2] = 2.0
        logits[1, 3, 0] = 2.0
        means, counts = mean_shifted_target_token_nll(logits, labels)
        expected_first = -math.log(math.exp(4.0) / (math.exp(4.0) + 2.0))
        expected_second = -math.log(math.exp(2.0) / (math.exp(2.0) + 2.0))
        self.assertEqual(counts, [2, 4])
        self.assertAlmostEqual(means[0], expected_first, places=6)
        self.assertAlmostEqual(means[1], expected_second, places=6)

    def test_candidate_selection_uses_lower_level_tie_break(self) -> None:
        selected, score, margin = select_minimum_mean_nll(
            {"1": 0.8, "2": 0.2, "3": 0.2, "4": 0.9, "5": 1.0}
        )
        self.assertEqual(selected, 2)
        self.assertEqual(score, 0.2)
        self.assertEqual(margin, 0.0)

    def test_candidate_selection_rejects_missing_or_nonfinite_scores(self) -> None:
        with self.assertRaises(ValueError):
            select_minimum_mean_nll({1: 0.1})
        with self.assertRaises(ValueError):
            select_minimum_mean_nll(
                {1: 0.1, 2: 0.2, 3: float("nan"), 4: 0.4, 5: 0.5}
            )


if __name__ == "__main__":
    unittest.main()
