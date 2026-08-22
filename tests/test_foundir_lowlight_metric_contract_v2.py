from __future__ import annotations

import math
import unittest

import numpy as np

from smoke.validate_foundir_lowlight_metric_contract import literal_psnr
from smoke.validate_foundir_lowlight_metric_contract_v2 import (
    expected_pyiqa_psnr,
    torch_literal_psnr,
)


class FoundIRLowlightMetricContractV2Tests(unittest.TestCase):
    def test_independent_literal_psnr_agrees(self) -> None:
        import torch

        reference = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = reference.copy()
        prediction[0, 0, 0] = 1
        reference_tensor = torch.from_numpy(reference.transpose(2, 0, 1).copy())[None].double() / 255
        prediction_tensor = torch.from_numpy(prediction.transpose(2, 0, 1).copy())[None].double() / 255
        self.assertAlmostEqual(
            literal_psnr(reference, prediction),
            torch_literal_psnr(reference_tensor, prediction_tensor),
            places=12,
        )

    def test_pyiqa_epsilon_semantics_are_explicit(self) -> None:
        import torch

        reference = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
        prediction = reference.clone()
        prediction[0, 0, 0, 0] = 1.0 / 255.0
        expected = expected_pyiqa_psnr(reference, prediction, 1e-8)
        literal = 10 * math.log10(1.0 / float(((reference - prediction) ** 2).mean()))
        self.assertLess(expected, literal)
        self.assertAlmostEqual(expected, 79.487884, places=5)


if __name__ == "__main__":
    unittest.main()
