from __future__ import annotations

import unittest
from pathlib import Path

from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_distributed import (
    complete_source_groups,
    cosine_learning_rate,
    merge_indexed_outputs,
    partition_source_groups,
    validate_full_batch_contract,
)


def record(source: str, level: int) -> PairRecord:
    digest = str(level) * 64
    return PairRecord(
        pair_id=f"{source}_L{level}",
        source_id=source,
        clean_path=Path(f"{source}_clean.png"),
        degraded_path=Path(f"{source}_L{level}.png"),
        project_level=f"L{level}",
        seed=level,
        split="train",
        clean_sha256=digest,
        degraded_sha256=digest,
    )


class QwenDistributedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            record(source, level)
            for source in ("a", "b", "c", "d")
            for level in range(1, 6)
        ]

    def test_complete_groups_keep_levels_together(self) -> None:
        groups = complete_source_groups(self.records)
        self.assertEqual(len(groups), 4)
        self.assertEqual(
            [[item.level_index for item in group] for group in groups],
            [[1, 2, 3, 4, 5]] * 4,
        )
        self.assertEqual(
            [[group[0].source_id for group in partition_source_groups(
                self.records, world_size=2, rank=rank
            )] for rank in range(2)],
            [["a", "c"], ["b", "d"]],
        )

    def test_incomplete_or_imbalanced_groups_fail(self) -> None:
        with self.assertRaises(ValueError):
            complete_source_groups(self.records[:-1])
        with self.assertRaises(ValueError):
            partition_source_groups(self.records, world_size=3, rank=0)

    def test_full_batch_contract_preserves_all_samples(self) -> None:
        validate_full_batch_contract(
            sample_count=40,
            source_count=8,
            world_size=8,
            micro_batch_size=5,
            gradient_accumulation_steps=1,
            global_batch_size=40,
        )
        validate_full_batch_contract(
            sample_count=40,
            source_count=8,
            world_size=1,
            micro_batch_size=5,
            gradient_accumulation_steps=8,
            global_batch_size=40,
        )
        with self.assertRaises(ValueError):
            validate_full_batch_contract(
                sample_count=40,
                source_count=8,
                world_size=8,
                micro_batch_size=5,
                gradient_accumulation_steps=2,
                global_batch_size=80,
            )

    def test_cosine_schedule_hits_declared_boundaries(self) -> None:
        self.assertAlmostEqual(
            cosine_learning_rate(
                step=1,
                total_steps=100,
                warmup_steps=5,
                base_learning_rate=2e-5,
                minimum_learning_rate=2e-6,
            ),
            4e-6,
        )
        self.assertAlmostEqual(
            cosine_learning_rate(
                step=5,
                total_steps=100,
                warmup_steps=5,
                base_learning_rate=2e-5,
                minimum_learning_rate=2e-6,
            ),
            2e-5,
        )
        self.assertAlmostEqual(
            cosine_learning_rate(
                step=100,
                total_steps=100,
                warmup_steps=5,
                base_learning_rate=2e-5,
                minimum_learning_rate=2e-6,
            ),
            2e-6,
        )

    def test_generated_outputs_merge_in_selection_order(self) -> None:
        merged = merge_indexed_outputs(
            [[(0, "a"), (2, "c")], [(1, "b"), (3, "d")]], expected_count=4
        )
        self.assertEqual(merged, ["a", "b", "c", "d"])
        with self.assertRaises(ValueError):
            merge_indexed_outputs([[(0, "a")], [(0, "again")]], expected_count=2)


if __name__ == "__main__":
    unittest.main()
