from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from smoke.evaluate_foundir_five_type_qwen_native_patches_ddp import (
    aggregate,
    aggregated_predictions,
    crop_boxes,
)
from smoke.train_qwen_multidegradation_lora_ddp import open_images


ROOT = Path(__file__).resolve().parents[1]


class FoundIRNativePatchDiagnosticTests(unittest.TestCase):
    def test_crop_boxes_preserve_native_448_pixels(self) -> None:
        boxes = crop_boxes(1920, 1080, 448)
        self.assertEqual(boxes["top_left"], (0, 0, 448, 448))
        self.assertEqual(boxes["bottom_right"], (1472, 632, 1920, 1080))
        self.assertEqual(boxes["center"], (736, 316, 1184, 764))
        for box in boxes.values():
            self.assertEqual(box[2] - box[0], 448)
            self.assertEqual(box[3] - box[1], 448)

    def test_open_images_applies_checked_crop_box(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "logs") as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (32, 24), color=(1, 2, 3)).save(path)
            record = {
                "record_id": "r1",
                "output_path": path.relative_to(ROOT).as_posix(),
                "crop_box": [4, 3, 20, 19],
            }
            self.assertEqual(open_images([record])[0].size, (16, 16))
            record["crop_box"] = [4, 3, 40, 19]
            with self.assertRaisesRegex(ValueError, "out-of-bounds"):
                open_images([record])

    def test_aggregation_rules_are_explicit(self) -> None:
        values = [1, 2, 3, 4, 5]
        self.assertEqual(aggregate(values, "maximum"), 5)
        self.assertEqual(aggregate(values, "second_highest"), 4)
        self.assertEqual(aggregate(values, "median"), 3)

    def test_center_and_consensus_predictions_remain_distinct(self) -> None:
        views = ["top_left", "top_right", "center", "bottom_left", "bottom_right"]
        rows = []
        for index, view in enumerate(views):
            rows.append(
                {
                    "parent_record_id": "r1",
                    "view": view,
                    "valid": True,
                    "levels": {
                        "blur": 5 if index == 0 else 1,
                        "noise": 2,
                        "rain": 1,
                        "haze": 1,
                        "low_light": 1,
                    },
                }
            )
        payload = {"view_predictions": rows}
        records = [{"record_id": "r1", "category": "02Blur_Noise", "composition_size": "2"}]
        center = aggregated_predictions(payload, records, method="center")[0]
        maximum = aggregated_predictions(payload, records, method="maximum")[0]
        vote_two = aggregated_predictions(payload, records, method="second_highest")[0]
        self.assertEqual(center["levels"]["blur"], 1)
        self.assertEqual(maximum["levels"]["blur"], 5)
        self.assertEqual(vote_two["levels"]["blur"], 1)

    def test_contract_is_diagnostic_only_and_protects_official_test(self) -> None:
        config = json.loads(
            (ROOT / "configs/foundir_five_type_native_patch_diagnostic_v1.json").read_text()
        )
        self.assertFalse(config["restoration_training_authorized_by_this_diagnostic"])
        self.assertFalse(config["protected_foundir_test_accessed"])
        self.assertEqual(config["crop_size"], 448)
        self.assertEqual(len(config["views"]), 5)

    def test_launcher_has_no_redundant_dataset_or_runtime_preflight(self) -> None:
        launcher = (
            ROOT / "smoke/launch_foundir_five_type_native_patch_diagnostic_8gpu.sh"
        ).read_text()
        runner = (
            ROOT / "smoke/run_foundir_five_type_native_patch_diagnostic_8gpu.sh"
        ).read_text()
        self.assertIn('"${1:-}" == "--check"', launcher)
        self.assertNotIn("--require-data", launcher + runner)
        self.assertNotIn("runtime_preflight", launcher + runner)
        self.assertLess(runner.index("candidate in joint per_type"), runner.index("--finalize"))
        self.assertIn("PROTECTED_OFFICIAL_TEST not_accessed", runner)


if __name__ == "__main__":
    unittest.main()
