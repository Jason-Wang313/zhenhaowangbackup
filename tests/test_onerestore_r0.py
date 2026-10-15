from __future__ import annotations

import unittest

import torch

from lowlight_restoration.onerestore_r0 import (
    CATEGORY_NAMES,
    CATEGORY_TO_INDEX,
    OneRestoreR0,
    category_name,
    make_descriptor_table,
    ms_ssim,
    parameter_count,
)


class OneRestoreR0Tests(unittest.TestCase):
    def test_category_contract_has_clean_five_singletons_ten_doubles(self):
        self.assertEqual(len(CATEGORY_NAMES), 16)
        self.assertEqual(category_name([]), "clean")
        self.assertEqual(category_name(["noise", "blur"]), "blur+noise")
        self.assertEqual(category_name(["low_light", "blur", "noise"]), "blur+noise+low_light")

    def test_descriptor_table_is_deterministic_and_normalized(self):
        left = make_descriptor_table(16, 324, 7)
        right = make_descriptor_table(16, 324, 7)
        self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.allclose(left.norm(dim=1), torch.ones(16)))

    def test_forward_shapes_range_and_triple_descriptor(self):
        model = OneRestoreR0(channel=8)
        image = torch.rand(2, 3, 32, 32)
        indices = torch.tensor([CATEGORY_TO_INDEX["blur+noise"], CATEGORY_TO_INDEX["clean"]])
        output = model(
            image,
            indices,
            descriptor_enabled=True,
            degradations=[["blur", "noise"], ["blur", "noise", "low_light"]],
        )
        self.assertEqual(output.shape, image.shape)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_descriptor_off_retains_graph_but_zeroes_cross_contribution(self):
        model = OneRestoreR0(channel=8)
        image = torch.rand(1, 3, 32, 32)
        first = torch.tensor([CATEGORY_TO_INDEX["blur"]])
        second = torch.tensor([CATEGORY_TO_INDEX["rain+haze"]])
        left = model(image, first, descriptor_enabled=False)
        right = model(image, second, descriptor_enabled=False)
        self.assertTrue(torch.equal(left, right))

    def test_ms_ssim_identity(self):
        image = torch.rand(1, 3, 192, 192)
        self.assertAlmostEqual(float(ms_ssim(image, image)), 1.0, places=5)

    def test_full_parameter_count_is_one_restore_scale(self):
        count = parameter_count(OneRestoreR0())
        self.assertGreater(count, 5_000_000)
        self.assertLess(count, 7_000_000)


if __name__ == "__main__":
    unittest.main()
