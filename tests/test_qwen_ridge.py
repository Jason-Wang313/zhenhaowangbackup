from __future__ import annotations

import unittest

import numpy as np

from lowlight_restoration.qwen_ridge import (
    decode_scores,
    predict,
    standardize_apply,
    standardize_fit,
)


class QwenRidgeTests(unittest.TestCase):
    def test_standardization_is_train_fitted_and_reusable(self) -> None:
        train = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
        standardized, mean, scale = standardize_fit(train)
        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            standardize_apply(np.array([[7.0, 22.0]]), mean, scale),
            np.array([[2.449489742783178, 2.449489742783178]]),
        )

    def test_score_decoding_rounds_and_clips_to_ordered_levels(self) -> None:
        scores = np.array([-3.0, 1.49, 2.51, 4.49, 9.0])
        np.testing.assert_array_equal(decode_scores(scores), np.array([1, 1, 3, 4, 5]))

    def test_ridge_prediction_uses_frozen_statistics_and_parameters(self) -> None:
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        scores, levels = predict(
            features,
            mean=np.array([1.0, 2.0]),
            scale=np.array([2.0, 2.0]),
            coefficient=np.array([1.0, 1.0]),
            intercept=1.0,
        )
        np.testing.assert_allclose(scores, np.array([1.0, 3.0]))
        np.testing.assert_array_equal(levels, np.array([1, 3]))

    def test_nonfinite_or_mismatched_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            decode_scores(np.array([np.nan]))
        with self.assertRaises(ValueError):
            standardize_apply(
                np.ones((2, 3)), np.ones(2), np.ones(2)
            )
        with self.assertRaises(ValueError):
            predict(
                np.ones((1, 2)),
                mean=np.zeros(2),
                scale=np.ones(2),
                coefficient=np.ones(3),
                intercept=0.0,
            )


if __name__ == "__main__":
    unittest.main()
