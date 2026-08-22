import unittest

import numpy as np

from smoke.run_foundir_lowlight_gate2_metric_blend import convex_blend, frequency_blend


class Gate2MetricBlendTests(unittest.TestCase):
    def test_convex_blend_selects_exact_target_endpoint(self):
        first = np.zeros((16, 16, 3), dtype=np.uint8)
        second = np.full_like(first, 200)
        output, parameters = convex_blend({"a": first, "b": second}, second, [0.0, 0.5, 1.0])
        self.assertTrue(np.array_equal(output, second))
        self.assertEqual(parameters["alpha"], 1.0)

    def test_frequency_blend_is_uint8_and_shape_preserving(self):
        rng = np.random.default_rng(7)
        proposals = {name: rng.integers(0, 256, (18, 19, 3), dtype=np.uint8) for name in ("a", "b")}
        output, parameters = frequency_blend(proposals, proposals["a"], [["a", "b"]], [1.0], [0.5, 1.0])
        self.assertEqual(output.shape, proposals["a"].shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(parameters["low"], "a")


if __name__ == "__main__": unittest.main()
