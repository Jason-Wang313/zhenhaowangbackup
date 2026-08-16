from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_ordinal import (
    OrdinalSeverityHead,
    decode_ordinal_logits,
    monotonic_penalty,
    ordinal_targets,
    select_complete_source_sequences,
)
from train_qwen_lowlight_ordinal_head import (
    feature_metadata,
    load_cached_feature,
    load_config,
    save_feature_new,
)

ROOT = Path(__file__).resolve().parents[1]


class QwenOrdinalTests(unittest.TestCase):
    @staticmethod
    def records() -> list[PairRecord]:
        return [
            PairRecord(
                pair_id=f"source_{source:02d}_L{level}",
                source_id=f"source_{source:02d}",
                clean_path=Path("clean.png"),
                degraded_path=Path("degraded.png"),
                project_level=f"L{level}",
                seed=source,
                split="train",
                clean_sha256="a" * 64,
                degraded_sha256=f"{source * 5 + level:064x}",
            )
            for source in range(20)
            for level in range(1, 6)
        ]

    def test_ordinal_targets_and_decoding_cover_all_levels(self) -> None:
        levels = torch.arange(1, 6, dtype=torch.int64)
        targets = ordinal_targets(levels)
        self.assertEqual(
            targets.tolist(),
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
        )
        logits = (targets * 2.0 - 1.0) * 5.0
        torch.testing.assert_close(decode_ordinal_logits(logits), levels)

    def test_monotonic_penalty_detects_boundary_inversion(self) -> None:
        ordered = torch.tensor([[3.0, 2.0, 1.0, -1.0]])
        inverted = torch.tensor([[1.0, 3.0, 0.0, -1.0]])
        self.assertEqual(float(monotonic_penalty(ordered)), 0.0)
        self.assertGreater(float(monotonic_penalty(inverted)), 0.0)

    def test_head_is_bounded_and_shape_checked(self) -> None:
        head = OrdinalSeverityHead(7168)
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 28676)
        self.assertEqual(head(torch.zeros(2, 7168)).shape, (2, 4))
        with self.assertRaises(ValueError):
            head(torch.zeros(2, 10))

    def test_complete_source_selection_is_deterministic_and_balanced(self) -> None:
        first = select_complete_source_sequences(
            self.records(), split="train", source_count=10, seed=41
        )
        second = select_complete_source_sequences(
            self.records(), split="train", source_count=10, seed=41
        )
        self.assertEqual(
            [[record.pair_id for record in sequence] for sequence in first],
            [[record.pair_id for record in sequence] for sequence in second],
        )
        self.assertEqual(len({sequence[0].source_id for sequence in first}), 10)
        self.assertTrue(
            all([record.level_index for record in sequence] == [1, 2, 3, 4, 5] for sequence in first)
        )

    def test_feature_cache_is_fail_closed(self) -> None:
        record = self.records()[0]
        metadata = feature_metadata(
            record,
            config_sha="b" * 64,
            model_revision="c" * 40,
            feature_dim=8,
            schema_version=1,
            extractor_sha256="d" * 64,
        )
        feature = torch.arange(8, dtype=torch.float32)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "feature.safetensors"
            save_feature_new(path, feature, metadata)
            torch.testing.assert_close(load_cached_feature(path, metadata, 8), feature)
            save_feature_new(path, feature, metadata)
            changed = dict(metadata)
            changed["model_revision"] = "e" * 40
            with self.assertRaises(RuntimeError):
                load_cached_feature(path, changed, 8)

    def test_frozen_ordinal_config_loads_and_binds_references(self) -> None:
        config, digest, _ = load_config(ROOT / "configs/qwen_lowlight_ordinal_head.json")
        self.assertEqual(len(digest), 64)
        self.assertEqual(config["features"]["feature_dim"], 7168)  # type: ignore[index]
        self.assertEqual(config["head"]["expected_trainable_parameters"], 28676)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
