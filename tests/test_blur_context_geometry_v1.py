import unittest
import numpy as np
from smoke.blur_context_geometry_v1 import crop, pixel_statistics
from smoke import official_vlm_router_v4_contract as contract


class ContextGeometryTests(unittest.TestCase):
    def test_same_pixels_survive_official_stitch_and_arbitrary_training_crop(self):
        row = {'height': '1080', 'width': '1920', 'record_id': 'synthetic_training_frame'}
        image = np.random.default_rng(17).integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        predictions = {tile['key']: contract.tile_array(image, tile)[0] for tile in contract.tiles(row)}
        full = contract.stitch(row, predictions)
        np.testing.assert_array_equal(full, image)
        item = {'crop_xy': [985, 489]}
        np.testing.assert_array_equal(crop(full, item), image[489:1001, 985:1497])
        np.testing.assert_array_equal(crop(full, item, 64), image[553:937, 1049:1433])
        self.assertEqual(crop(full, item, 64).shape, (384, 384, 3))

    def test_outside_statistics_exclude_exact_trained_region(self):
        lq = np.zeros((768, 1024, 3), dtype=np.uint8)
        full = lq.copy()
        item = {'crop_xy': [27, 81]}
        full[81:593, 27:539] = 255
        standalone = np.full((512, 512, 3), 255, dtype=np.uint8)
        stats = pixel_statistics(lq, lq, full, standalone, item)
        self.assertEqual(stats['outside_trained_crop_MSE'], 0)
        self.assertEqual(stats['same_crop_output_difference_RMS_RGB01'], 0)
        full[0, 0] = 255
        stats = pixel_statistics(lq, lq, full, standalone, item)
        self.assertAlmostEqual(stats['outside_trained_crop_MSE'], 1 / (768 * 1024 - 512 * 512))

    def test_reject_invalid_fixed_crop_and_margin(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for item, margin in [({'crop_xy': [1409, 0]}, 0), ({'crop_xy': [0, 569]}, 0),
                             ({'crop_xy': [0, 0]}, 256), ({'crop_xy': [0, 0]}, -1)]:
            with self.assertRaises(ValueError):
                crop(image, item, margin)


if __name__ == '__main__':
    unittest.main()
