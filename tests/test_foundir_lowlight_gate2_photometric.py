import unittest

import numpy as np

from smoke.run_foundir_lowlight_gate2_photometric import (
    fit_exposure,
    fit_monotone_lut,
    isotonic_increasing,
    render_predicted_summary,
)


class Gate2PhotometricTests(unittest.TestCase):
    def test_isotonic_projection_is_monotone(self):
        values = np.array([0.0, 0.8, 0.2, 0.7, 1.0])
        result = isotonic_increasing(values, np.ones_like(values))
        self.assertTrue(np.all(np.diff(result) >= 0))
        self.assertTrue(np.all((result >= 0) & (result <= 1)))

    def test_exposure_recovers_unclipped_scalar(self):
        source = np.linspace(0.01, 0.4, 300, dtype=np.float32).reshape(10, 10, 3)
        target = source * 2.25
        self.assertAlmostEqual(fit_exposure(source, target, 0.25, 16.0), 2.25, places=5)

    def test_monotone_lut_does_not_invert_channel(self):
        source = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        source = np.repeat(source, 3, axis=2)
        target = np.rint(np.sqrt(source / 255.0) * 255).astype(np.uint8)
        prediction, _ = fit_monotone_lut(source, target)
        self.assertTrue(np.all(np.diff(prediction[..., 0].reshape(-1).astype(int)) >= 0))

    def test_input_predictor_render_is_uint8_and_shape_preserving(self):
        source = np.full((12, 13, 3), 20, dtype=np.uint8)
        predicted = np.array([0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.1, -0.1])
        output = render_predicted_summary(source, predicted, [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
        self.assertEqual(output.shape, source.shape)
        self.assertEqual(output.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
