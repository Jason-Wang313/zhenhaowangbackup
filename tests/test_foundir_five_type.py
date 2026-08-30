from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from lowlight_restoration.foundir_five_type import (
    canonical_restoration_text,
    hamming64,
    load_five_type_metadata,
    perceptual_components,
    stable_group_split,
)
from smoke.build_foundir_five_type_manifest import ROOT as WORKSPACE_ROOT, build
from smoke.evaluate_foundir_five_type_vae_ceiling_ddp import select_balanced_targets
from smoke.export_foundir_five_type_qwen_conditions import (
    condition_row,
    select_balanced_pilot,
)
from smoke.export_foundir_five_type_qwen_conditions_sharded_ddp import (
    append_rows,
    load_shard,
    merge_shards,
)


class FoundIRFiveTypeTests(unittest.TestCase):
    def test_filters_jpeg_and_noncanonical_weather_variants(self) -> None:
        rows = "\n".join(
            [
                "01Blur/GT/1.png, 01Blur/LQ/1.png",
                "02Blur_Noise/GT/2.png, 02Blur_Noise/LQ/2.png",
                "06JPEG/GT/3.png, 06JPEG/LQ/3.jpg",
                "11Raindrop/GT/4.png, 11Raindrop/LQ/4.png",
                "18Lowlight_Blur_Noise/GT/5.png, 18Lowlight_Blur_Noise/LQ/5.png",
            ]
        )
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory:
            path = Path(directory) / "meta.txt"
            path.write_text(rows + "\n")
            records, counts = load_five_type_metadata(path)
        self.assertEqual([record.category for record in records], ["01Blur", "02Blur_Noise", "18Lowlight_Blur_Noise"])
        self.assertEqual(counts["06JPEG"], 1)
        self.assertEqual(counts["11Raindrop"], 1)

    def test_canonical_text_has_fixed_order_and_absent_word(self) -> None:
        text = canonical_restoration_text(
            {"blur": 3, "noise": 1, "rain": 2, "haze": 1, "low_light": 5}
        )
        self.assertIn("Blur: L3; Noise: absent; Rain: L2; Haze: absent; Low-light: L5", text)

    def test_group_split_is_deterministic_and_group_preserving(self) -> None:
        first = stable_group_split("same-gt", seed=7)
        self.assertEqual(first, stable_group_split("same-gt", seed=7))
        self.assertIn(first, {"train", "validation", "internal_test"})

    def test_invalid_metadata_layout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meta.txt"
            path.write_text("01Blur/GT/1.png 01Blur/LQ/1.png\n")
            with self.assertRaises(ValueError):
                load_five_type_metadata(path)

    def test_perceptual_components_find_all_close_pairs(self) -> None:
        hashes = [0, 1, (1 << 4) | 1, (1 << 20) | (1 << 21) | (1 << 22) | (1 << 23), 2**64 - 1]
        components = perceptual_components(hashes, maximum_distance=4)
        self.assertEqual(components[0], components[1])
        self.assertEqual(components[0], components[2])
        self.assertEqual(components[0], components[3])
        self.assertNotEqual(components[0], components[4])
        self.assertEqual(hamming64(0, hashes[3]), 4)

    def test_manifest_builder_groups_duplicate_gt_and_publishes_once(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory:
            root = Path(directory)
            training = root / "Train"
            rows = [
                ("01Blur", "a.png", (20, 30, 40), (18, 28, 38)),
                ("05Noise", "b.png", (20, 30, 40), (25, 35, 45)),
                ("10Rain", "c.png", (160, 120, 80), (150, 110, 70)),
            ]
            metadata_rows: list[str] = []
            for category, name, gt_color, lq_color in rows:
                gt_path = training / category / "GT" / name
                lq_path = training / category / "LQ" / name
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                lq_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), gt_color).save(gt_path)
                Image.new("RGB", (64, 64), lq_color).save(lq_path)
                metadata_rows.append(f"{category}/GT/{name}, {category}/LQ/{name}")
            metadata = root / "meta.txt"
            metadata.write_text("\n".join(metadata_rows) + "\n")
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "data": {
                            "official_metadata": str(metadata.relative_to(WORKSPACE_ROOT)),
                            "training_root": str(training.relative_to(WORKSPACE_ROOT)),
                            "expected_five_type_records": 3,
                            "split_seed": 17,
                        }
                    }
                )
            )
            output_csv = root / "nested" / "pairs.csv"
            output_json = root / "nested" / "pairs.json"
            summary = build(config, output_csv, output_json, workers=2)
            with output_csv.open(newline="") as handle:
                built = list(csv.DictReader(handle))
            self.assertEqual(summary["records"], 3)
            self.assertTrue(output_json.is_file())
            self.assertEqual(built[0]["source_group"], built[1]["source_group"])
            self.assertEqual(built[0]["split"], built[1]["split"])
            # Identical immutable output is reusable; differing output is not.
            self.assertEqual(build(config, output_csv, output_json, workers=1), summary)
            output_csv.write_text("different\n")
            with self.assertRaises(RuntimeError):
                build(config, output_csv, output_json, workers=1)

    def test_balanced_pilot_and_vae_selection_cover_categories(self) -> None:
        categories = [
            "01Blur", "02Blur_Noise", "05Noise", "08Haze", "09Lowlight_Haze",
            "10Rain", "13Rain_Haze", "14Lowlight", "15Lowlight_Blur",
            "16Lowlight_Noise", "18Lowlight_Blur_Noise",
        ]
        rows = [
            {
                "record_id": f"r-{category}-{index}",
                "split": "train",
                "source_group": f"g-{category}-{index}",
                "category": category,
            }
            for category in categories
            for index in range(3)
        ]
        pilot = select_balanced_pilot(rows, count=22, seed=9)
        self.assertEqual(len(pilot), 22)
        self.assertEqual(set(row["category"] for row in pilot), set(categories))
        vae = select_balanced_targets(rows, per_category=2, seed=9)
        self.assertEqual(len(vae), 22)
        self.assertEqual(len({row["source_group"] for row in vae}), 22)

    def test_condition_row_preserves_qwen_and_constructs_type_oracle(self) -> None:
        record = {
            "record_id": "r1",
            "split": "train",
            "source_group": "g1",
            "category": "02Blur_Noise",
            "composition_size": "2",
            "lq_path": "lq.png",
            "gt_path": "gt.png",
            "lq_sha256": "a" * 64,
            "gt_sha256": "b" * 64,
        }
        row = condition_row(
            record,
            '{"blur":1,"noise":4,"rain":3,"haze":1,"low_light":1}',
        )
        predicted = json.loads(row["qwen_levels_json"])
        oracle = json.loads(row["type_oracle_levels_json"])
        self.assertEqual(predicted["rain"], 3)
        self.assertEqual(oracle, {"blur": 2, "noise": 4, "rain": 1, "haze": 1, "low_light": 1})

    def test_sharded_condition_export_resumes_and_merges_in_manifest_order(self) -> None:
        raw = '{"blur":2,"noise":1,"rain":1,"haze":1,"low_light":1}'
        records = []
        for index in range(4):
            records.append(
                {
                    "record_id": f"r{index}",
                    "split": "train" if index < 2 else "validation",
                    "source_group": f"g{index}",
                    "category": "01Blur",
                    "composition_size": "1",
                    "lq_path": f"lq{index}.png",
                    "gt_path": f"gt{index}.png",
                    "lq_sha256": f"{index + 1:064x}",
                    "gt_sha256": f"{index + 11:064x}",
                }
            )
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory:
            root = Path(directory)
            shard_dir = root / "shards"
            for rank in range(2):
                assigned = records[rank::2]
                append_rows(
                    shard_dir / f"rank_{rank:02d}.jsonl",
                    [condition_row(record, raw) for record in assigned],
                )
                self.assertEqual(len(load_shard(shard_dir / f"rank_{rank:02d}.jsonl", assigned)), 2)
            manifest = root / "manifest.csv"
            manifest.write_text("placeholder\n")
            qwen_config = root / "qwen.json"
            qwen_config.write_text("{}\n")
            adapter = root / "adapter.safetensors"
            adapter.write_bytes(b"adapter")
            output_csv = root / "conditions.csv"
            output_json = root / "conditions.json"
            payload = merge_shards(
                manifest_rows=records,
                shard_dir=shard_dir,
                world=2,
                output_csv=output_csv,
                output_json=output_json,
                config_path=qwen_config,
                qwen_config_path=qwen_config,
                manifest_path=manifest,
                adapter=adapter,
            )
            with output_csv.open(newline="") as handle:
                merged = list(csv.DictReader(handle))
            self.assertEqual([row["record_id"] for row in merged], ["r0", "r1", "r2", "r3"])
            self.assertEqual(payload["records"], 4)


if __name__ == "__main__":
    unittest.main()
