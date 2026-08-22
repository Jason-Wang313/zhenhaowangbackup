from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from lowlight_restoration.compact_residual import CompactResidualControl, GlobalToneHead, parameter_count
from smoke.evaluate_foundir_lowlight_gate3_compact import best_blend
from smoke.train_foundir_lowlight_gate3_compact import DEFAULT_CONFIG, read_config, read_manifest


class Gate3CompactTests(unittest.TestCase):
    def test_frozen_contract_and_development_roles(self) -> None:
        config, digest = read_config(DEFAULT_CONFIG)
        training, outer = read_manifest(config)
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(training), 429)
        self.assertEqual(len(outer), 245)
        self.assertEqual({row["family"] for row in training}, {"sice", "sdsd"})
        self.assertEqual({row["family"] for row in outer}, {"lolv2", "sice", "sdsd"})
        self.assertFalse(config["protected_foundir_pixels_allowed"])

    def test_zero_initialized_model_is_exact_identity(self) -> None:
        torch.manual_seed(3)
        model = CompactResidualControl(width=8, encoder_blocks=(1, 1), middle_blocks=1, decoder_blocks=(1, 1), global_width=4)
        image = torch.rand(2, 3, 32, 36)
        image[..., 0, 0] = 0
        with torch.no_grad():
            output = model(image, torch.rand(2, 3, 16, 16))
        torch.testing.assert_close(output, image, rtol=0, atol=0)

    def test_frozen_model_size_is_compact_and_bounded(self) -> None:
        config = json.loads(DEFAULT_CONFIG.read_text())
        model = config["compact_control"]
        network = CompactResidualControl(
            width=model["width"],
            encoder_blocks=model["encoder_blocks"],
            middle_blocks=model["middle_blocks"],
            decoder_blocks=model["decoder_blocks"],
            global_width=model["global_width"],
            condition_dim=model["condition_dim"],
            local_residual_bound=model["local_residual_bound"],
        )
        self.assertGreater(parameter_count(network), 1_500_000)
        self.assertLess(parameter_count(network), 2_000_000)

    def test_tone_transform_is_finite_and_bounded(self) -> None:
        image = torch.tensor([[[[0.0, 0.2], [0.8, 1.0]]] * 3])
        parameters = torch.tensor([[20.0, -20.0, 1.0, -20.0, 20.0, 0.0, 20.0, -20.0, 0.0]])
        output = GlobalToneHead.apply(image, parameters)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_target_only_blend_is_an_oracle_ceiling(self) -> None:
        target = np.full((4, 4, 3), 100, dtype=np.uint8)
        compact = np.full_like(target, 80)
        cascade = np.full_like(target, 120)
        blend, alpha = best_blend(compact, cascade, target, [0.0, 0.5, 1.0])
        self.assertEqual(alpha, 0.5)
        np.testing.assert_array_equal(blend, target)


if __name__ == "__main__":
    unittest.main()
