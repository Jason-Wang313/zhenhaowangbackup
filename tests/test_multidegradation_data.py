import unittest
from collections import Counter
from pathlib import Path

from lowlight_restoration.multidegradation_data import (
    ROOT,
    SPLIT_SOURCE_COUNTS,
    assign_source_splits,
    build_pilot_records,
    generate_record,
    load_source_inventory,
    validate_pilot_records,
    _apply_lowlight,
    _operator_parameters,
)

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
