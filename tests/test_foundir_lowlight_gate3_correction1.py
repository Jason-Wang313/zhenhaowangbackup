from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from lowlight_restoration.structured_qwen_residual import (
    StructuredQwenToneResidual,
    fit_measured_posterior_ridge,
    predict_measured_posterior,
    structured_qwen,
    support_reference,
    support_shrink,
)


def condition(level: int, entropy: float = 0.0, margin: float = 1.0) -> np.ndarray:
    posterior = np.zeros(5, dtype=np.float32)
    posterior[level - 1] = 1.0
    return np.concatenate((posterior, np.asarray([level / 5.0, entropy, margin], dtype=np.float32)))


class Gate3Correction1Tests(unittest.TestCase):
    def test_structured_condition_is_strict(self) -> None:
        np.testing.assert_array_equal(structured_qwen(condition(3)), condition(3))
        invalid = condition(3)
        invalid[5] = 0.2
        with self.assertRaises(ValueError):
            structured_qwen(invalid)

    def test_tone_response_is_monotone_and_bounded(self) -> None:
        model = StructuredQwenToneResidual(
            maximum_gain_parameter_delta=0.65,
            maximum_gamma_parameter_delta=0.45,
            entropy_deficit_anchor=0.04,
            margin_anchor=0.05,
            posterior_center=0.5,
            maximum_knot_adjustment=0.05,
            initial_raw_slope=1.0,
        )
        structured = torch.from_numpy(np.stack([condition(level) for level in range(1, 6)]))
        output = model(structured, torch.full((5, 5), 0.2), torch.ones(5))
        self.assertTrue(torch.all(output[1:, :3] >= output[:-1, :3]))
        self.assertTrue(torch.all(output[1:, 3:6] <= output[:-1, 3:6]))
        self.assertLessEqual(float(output[:, :3].abs().max()), 0.65)
        self.assertLessEqual(float(output[:, 3:6].abs().max()), 0.45)
        torch.testing.assert_close(output[:, 6:], torch.zeros_like(output[:, 6:]))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 12)

    def test_uncertainty_and_support_only_shrink(self) -> None:
        model = StructuredQwenToneResidual(
            maximum_gain_parameter_delta=0.65,
            maximum_gamma_parameter_delta=0.45,
            entropy_deficit_anchor=0.04,
            margin_anchor=0.05,
            posterior_center=0.5,
            maximum_knot_adjustment=0.05,
            initial_raw_slope=1.0,
        )
        certain = torch.from_numpy(np.stack([condition(5, 0.90, 0.10), condition(5, 0.99, 0.005)]))
        output = model(certain, torch.full((2, 5), 0.2), torch.tensor([1.0, 0.2]))
        self.assertGreater(float(output[0, 0].abs()), float(output[1, 0].abs()))

    def test_ridge_and_support_are_training_only_deterministic(self) -> None:
        rng = np.random.default_rng(11)
        measured = rng.normal(size=(12, 4)).astype(np.float32)
        structured = np.stack([condition(index % 5 + 1, 0.95, 0.05) for index in range(12)])
        ridge = fit_measured_posterior_ridge(measured, structured[:, :5], penalty=10.0)
        predicted = predict_measured_posterior(measured, ridge)
        self.assertEqual(predicted.shape, structured[:, :5].shape)
        np.testing.assert_allclose(predicted.sum(axis=1), 1.0, atol=1e-6)
        reference = support_reference(structured, [f"g{index // 2}" for index in range(12)], 0.95)
        shrink = support_shrink(
            structured[:3], reference, query_group_ids=["g0", "g0", "g1"], exclude_same_group=True
        )
        self.assertTrue(np.all((0.0 < shrink) & (shrink <= 1.0)))

    def test_contract_freezes_first_correction_and_parent_outputs(self) -> None:
        root = Path(__file__).resolve().parents[1]
        path = root / "configs/foundir_lowlight_gate3_correction1_v1.json"
        if not path.exists():
            self.skipTest("correction-1 contract not frozen yet")
        config = json.loads(path.read_text())
        self.assertEqual(config["status"], "frozen_before_gate3_qwen_correction1")
        self.assertEqual(config["training"]["correction_index"], 1)
        self.assertFalse(config["protected_foundir_pixels_allowed"])
        self.assertIn("causal_prediction_manifest", config["parents"])
        self.assertTrue(config["adapter"]["frozen_measured_and_oracle_outputs_reused"])
        self.assertEqual(config["runtime"]["world_size"], 4)


if __name__ == "__main__":
    unittest.main()
