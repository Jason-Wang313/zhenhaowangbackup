from __future__ import annotations

import unittest

import numpy as np
import torch

from lowlight_restoration.representation_ceiling import (
    centered_crop,
    high_frequency_diagnostic,
    reconstruct_tiled,
    seam_error_ratio,
    tile_origins,
)
from smoke.evaluate_lowlight_vae_ceiling_ddp import AutoencoderTileProcessor


class RepresentationCeilingTests(unittest.TestCase):
    def test_tile_origins_cover_irregular_dimension(self) -> None:
        self.assertEqual(tile_origins(17, 8, 2), (0, 6, 9))
        self.assertEqual(tile_origins(7, 8, 2), (0,))
        with self.assertRaisesRegex(ValueError, "geometry"):
            tile_origins(10, 8, 8)

    def test_identity_tile_processor_is_exact_on_irregular_image(self) -> None:
        generator = np.random.default_rng(3)
        image = generator.random((17, 23, 3), dtype=np.float32)
        output, metadata = reconstruct_tiled(
            image,
            lambda tile: tile.copy(),
            tile_size=8,
            overlap=2,
        )
        np.testing.assert_allclose(output, image, atol=1e-6)
        self.assertEqual(metadata["tile_count"], 12)
        self.assertAlmostEqual(seam_error_ratio(image, output, metadata), 1.0)

    def test_frequency_diagnostic_detects_blur(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.float32)
        image[::2, ::2] = 1.0
        blurred = np.full_like(image, image.mean())
        diagnostic = high_frequency_diagnostic(image, blurred)
        self.assertLess(diagnostic["high_frequency_energy_ratio"], 0.1)
        self.assertGreater(diagnostic["high_frequency_error_rms"], 0.1)

    def test_center_crop_preserves_requested_center(self) -> None:
        image = np.arange(9 * 11 * 3, dtype=np.float32).reshape(9, 11, 3)
        crop = centered_crop(image, 5)
        np.testing.assert_array_equal(crop, image[2:7, 3:8])

    def test_autoencoder_processor_preserves_shape_after_alignment_padding(self) -> None:
        class IdentityAutoencoder:
            @staticmethod
            def encode(tensor: torch.Tensor) -> torch.Tensor:
                return tensor

            @staticmethod
            def decode(tensor: torch.Tensor) -> torch.Tensor:
                return tensor

        generator = np.random.default_rng(9)
        image = generator.random((17, 23, 3), dtype=np.float32)
        processor = AutoencoderTileProcessor(
            IdentityAutoencoder(), torch.device("cpu"), alignment_multiple=16
        )
        output = processor(image)
        self.assertEqual(output.shape, image.shape)
        np.testing.assert_allclose(output, image, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
