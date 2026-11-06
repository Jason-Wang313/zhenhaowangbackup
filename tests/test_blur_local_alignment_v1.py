import unittest
import numpy as np
from lowlight_restoration.blur_local_alignment_v1 import warp, validate_pair


class BackwardFlowTests(unittest.TestCase):
    def test_zero_flow_is_exact_identity(self):
        image = np.random.default_rng(17).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        np.testing.assert_array_equal(warp(image, np.zeros((48, 64, 2), dtype=np.float32)), image)

    def test_positive_flow_samples_positive_source_coordinates(self):
        image = np.broadcast_to(np.arange(64, dtype=np.uint8)[None, :, None], (48, 64, 3)).copy()
        flow = np.zeros((48, 64, 2), dtype=np.float32)
        flow[..., 0] = 3
        actual = warp(image, flow)
        np.testing.assert_array_equal(actual[:, :-3], image[:, 3:])
        np.testing.assert_array_equal(actual[:, -3:], np.repeat(image[:, -1:], 3, axis=1))

    def test_nonfinite_flow_and_wrong_images_rejected(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            warp(image, np.full((48, 64, 2), np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            validate_pair(image, image.astype(np.float32))


if __name__ == '__main__':
    unittest.main()
