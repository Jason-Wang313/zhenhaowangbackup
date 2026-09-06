import unittest
from collections import Counter
from pathlib import Path

from lowlight_restoration.multidegradation_data import (
    BLUR_FAMILIES_PHYSICAL_V8,
    BLUR_PARAMETERS,
    BLUR_PARAMETERS_EXTREME_V6,
    BLUR_PARAMETERS_MILD_V7,
    NOISE_FAMILIES_PHYSICAL_V8,
    NOISE_SIGMA_RANGES_MILD_V7,
    ROOT,
    SPLIT_SOURCE_COUNTS,
    assign_source_splits,
    build_pilot_records,
    generate_record,
    load_source_inventory,
    validate_pilot_records,
    _apply_blur,
    _apply_lowlight,
    _apply_noise,
    _apply_non_target_nuisance,
    _operator_parameters,
)

import numpy as np
import cv2


class MultiDegradationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = load_source_inventory(
            ROOT / "manifests/low_light_source_pool_2000.json"
        )
        cls.records = build_pilot_records(cls.inventory)

    def test_exact_source_disjoint_stratified_split(self):
        assignments = assign_source_splits(self.inventory)
        roles = {}
        for split, records in assignments.items():
            counts = Counter(record["dataset"] for record in records)
            for dataset, expected in SPLIT_SOURCE_COUNTS.items():
                self.assertEqual(counts[dataset], expected[split])
            for record in records:
                self.assertNotIn(record["source_id"], roles)
                roles[record["source_id"]] = split
        self.assertEqual(len(roles), 2000)

    def test_full_manifest_contract_and_repeatability(self):
        summary = validate_pilot_records(self.records)
        self.assertEqual(summary["records"], 12000)
        self.assertTrue(summary["source_disjoint"])
        rebuilt = build_pilot_records(self.inventory)
        self.assertEqual(self.records, rebuilt)

    def test_all_types_and_compositions_exist_in_each_split(self):
        for split in ("train", "validation", "test"):
            subset = [record for record in self.records if record["split"] == split]
            self.assertEqual({record["composition_size"] for record in subset}, {0, 1, 2, 3})
            active = {name for record in subset for name in record["degradations"]}
            self.assertEqual(active, {"blur", "noise", "rain", "haze", "low_light"})

    def test_haze_never_uses_invented_depth(self):
        haze = [record for record in self.records if "haze" in record["degradations"]]
        self.assertTrue(haze)
        for record in haze:
            self.assertEqual(record["source_dataset"], "mio100")
            parameters = record["operators"]["haze"]
            self.assertTrue((ROOT / parameters["depth_path"]).is_file())
            self.assertEqual(len(parameters["depth_sha256"]), 64)

    def test_haze_accepts_only_explicit_hash_verified_aligned_depth(self):
        template = next(record for record in self.inventory if record["dataset"] == "mio100")
        depth_path = "data/agenticir/haze/001/predict_depth.mat"
        from lowlight_restoration.multidegradation_data import sha256

        explicit = {
            **template,
            "dataset": "lsdir",
            "depth_path": depth_path,
            "depth_sha256": sha256(ROOT / depth_path),
        }
        parameters = _operator_parameters("haze", 2, seed=7, source=explicit)
        self.assertEqual(parameters["depth_path"], depth_path)
        broken = {**explicit, "depth_sha256": "0" * 64}
        with self.assertRaisesRegex(RuntimeError, "depth SHA-256 changed"):
            _operator_parameters("haze", 2, seed=7, source=broken)
        missing = {key: value for key, value in explicit.items() if key != "depth_sha256"}
        with self.assertRaisesRegex(ValueError, "missing its frozen SHA-256"):
            _operator_parameters("haze", 2, seed=7, source=missing)

    def test_human_approved_blur_profile_is_locked_without_changing_legacy(self):
        self.assertEqual(
            BLUR_PARAMETERS,
            {2: (10, 12.0), 3: (20, 18.0), 4: (30, 30.0), 5: (30, 48.0)},
        )
        self.assertEqual(
            BLUR_PARAMETERS_EXTREME_V6,
            {2: (22, 18.0), 3: (30, 24.0), 4: (90, 72.0), 5: (200, 160.0)},
        )
        source = self.inventory[0]
        parameters = _operator_parameters(
            "blur",
            5,
            seed=7,
            source=source,
            blur_parameters=BLUR_PARAMETERS_EXTREME_V6,
        )
        self.assertEqual((parameters["radius"], parameters["sigma"]), (200, 160.0))

    def test_advisor_directed_v7_covers_subtle_blur_and_noise(self):
        self.assertEqual(
            BLUR_PARAMETERS_MILD_V7,
            {2: (1, 1.0), 3: (2, 1.6), 4: (4, 3.2), 5: (8, 6.4)},
        )
        self.assertEqual(
            NOISE_SIGMA_RANGES_MILD_V7,
            {2: (1.0, 3.0), 3: (3.0, 6.0), 4: (6.0, 12.0), 5: (12.0, 25.0)},
        )
        source = self.inventory[0]
        blur = _operator_parameters(
            "blur", 2, seed=7, source=source,
            blur_parameters=BLUR_PARAMETERS_MILD_V7,
        )
        noise = _operator_parameters(
            "noise", 2, seed=7, source=source,
            noise_sigma_ranges=NOISE_SIGMA_RANGES_MILD_V7,
        )
        self.assertEqual((blur["radius"], blur["sigma"]), (1, 1.0))
        self.assertGreaterEqual(noise["sigma_0_255"], 1.0)
        self.assertLessEqual(noise["sigma_0_255"], 3.0)

    def test_clean_crop_is_literal_and_one_composite_is_deterministic(self):
        clean = next(record for record in self.records if record["composition_size"] == 0)
        clean_image, clean_hash = generate_record(clean)
        self.assertEqual(clean_image.shape, (448, 448, 3))
        self.assertEqual(clean_hash, generate_record(clean)[1])
        composite = next(
            record
            for record in self.records
            if record["composition_size"] == 3 and "haze" in record["degradations"]
        )
        image, digest = generate_record(composite)
        self.assertEqual(image.shape, (448, 448, 3))
        self.assertEqual(digest, generate_record(composite)[1])

    def test_uniform_value_lowlight_limit_is_finite_and_ordered(self):
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        result = _apply_lowlight(
            image,
            {"dst_max": 120, "target_luma_ratio": 0.30},
        )
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isfinite(result).all())
        self.assertLess(float(result.mean()), float(image.mean()))
        self.assertAlmostEqual(float(result.mean()) / float(image.mean()), 0.30, places=2)

    def test_v8_physical_blur_families_are_deterministic_and_ordered(self):
        rng = np.random.default_rng(4)
        image = np.rint(rng.uniform(20, 235, (128, 128, 3))).astype(np.uint8)
        for family, levels in BLUR_FAMILIES_PHYSICAL_V8.items():
            outputs = [
                _apply_blur(
                    image,
                    {**levels[level], "family": family, "seed": 7123},
                )
                for level in range(2, 6)
            ]
            self.assertTrue(
                np.array_equal(
                    outputs[0],
                    _apply_blur(
                        image,
                        {**levels[2], "family": family, "seed": 7123},
                    ),
                )
            )
            high_frequency = [
                float(
                    cv2.Laplacian(
                        cv2.cvtColor(output, cv2.COLOR_BGR2GRAY),
                        cv2.CV_32F,
                    ).var()
                )
                for output in outputs
            ]
            self.assertTrue(
                all(a > b for a, b in zip(high_frequency, high_frequency[1:])),
                (family, high_frequency),
            )

    def test_v8_camera_noise_families_are_deterministic_and_ordered(self):
        rng = np.random.default_rng(5)
        image = np.rint(rng.uniform(20, 235, (128, 128, 3))).astype(np.uint8)
        for family, levels in NOISE_FAMILIES_PHYSICAL_V8.items():
            outputs = [
                _apply_noise(
                    image,
                    {**levels[level], "family": family, "seed": 8123},
                )
                for level in range(2, 6)
            ]
            self.assertTrue(
                np.array_equal(
                    outputs[0],
                    _apply_noise(
                        image,
                        {**levels[2], "family": family, "seed": 8123},
                    ),
                )
            )
            residual_rms = [
                float(np.sqrt(np.mean((output.astype(np.float32) - image) ** 2)))
                for output in outputs
            ]
            self.assertTrue(
                all(a < b for a, b in zip(residual_rms, residual_rms[1:])),
                (family, residual_rms),
            )

    def test_v8_non_target_nuisances_are_bounded_and_deterministic(self):
        image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        for parameters in (
            {"family": "jpeg", "quality": 81, "seed": 3},
            {"family": "resample", "scale": 0.83, "seed": 4},
        ):
            first = _apply_non_target_nuisance(image, parameters)
            second = _apply_non_target_nuisance(image, parameters)
            self.assertEqual(first.shape, image.shape)
            self.assertEqual(first.dtype, np.uint8)
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, image))


if __name__ == "__main__":
    unittest.main()
    NOISE_FAMILIES_PHYSICAL_V8,
