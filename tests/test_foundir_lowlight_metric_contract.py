from __future__ import annotations

import unittest

import numpy as np

from smoke.validate_foundir_lowlight_metric_contract import (
    literal_psnr,
    make_canaries,
    png_roundtrip,
)


class FoundIRLowlightMetricContractTests(unittest.TestCase):
    def test_canaries_are_deterministic_and_distinct(self) -> None:
        first = make_canaries(20260822, 64, 64)
        second = make_canaries(20260822, 64, 64)
        self.assertEqual(set(first), set(second))
        for name in first:
            self.assertTrue(np.array_equal(first[name][0], second[name][0]))
            self.assertTrue(np.array_equal(first[name][1], second[name][1]))
        self.assertTrue(np.array_equal(first["identity"][0], first["identity"][1]))
        for name in set(first) - {"identity"}:
            self.assertFalse(np.array_equal(first[name][0], first[name][1]))

    def test_literal_psnr_uses_saved_uint8_rgb_semantics(self) -> None:
        reference = np.zeros((64, 64, 3), dtype=np.uint8)
        prediction = reference.copy()
        self.assertEqual(literal_psnr(reference, prediction), 100.0)
        prediction[0, 0, 0] = 1
        expected_mse = 1.0 / (64 * 64 * 3 * 255 * 255)
        expected = 10.0 * np.log10(1.0 / expected_mse)
        self.assertAlmostEqual(literal_psnr(reference, prediction), expected, places=12)

    def test_png_roundtrip_is_exact(self) -> None:
        image = make_canaries(20260822, 64, 64)["clipping"][1]
        decoded, digest = png_roundtrip(image)
        self.assertTrue(np.array_equal(decoded, image))
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
