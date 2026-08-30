from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from collections import Counter

from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES
from lowlight_restoration.foundir_flux2_protocol import (
    SOW_STAGE2_THRESHOLDS,
    aggregate_task_metrics,
    balanced_evaluation_subset,
    build_curriculum_pools,
    curriculum_record_index,
    fold_for_group,
    select_checkpoint,
    worst_normalized_margin,
)
from smoke.evaluate_foundir_five_type_flux2_tiled_ddp import tiled_restore, tile_starts
from smoke.train_foundir_five_type_flux2_full_ddp import CurriculumSampler
from smoke.freeze_foundir_five_type_evaluation_contract import contract_rows
from smoke.select_foundir_five_type_flux2_checkpoint import ROOT as WORKSPACE_ROOT, select


def synthetic_rows(split: str, groups_per_category: int = 12) -> list[dict[str, str]]:
    rows = []
    for category, degradations in FIVE_TYPE_CATEGORIES.items():
        for index in range(groups_per_category):
            rows.append(
                {
                    "record_id": f"{split}-{category}-{index}",
                    "split": split,
                    "source_group": f"{split}-group-{category}-{index}",
                    "category": category,
                    "composition_size": str(len(degradations)),
                }
            )
    return rows


class FoundIRFlux2ProtocolTests(unittest.TestCase):
    def test_native_tiling_covers_and_blends_identity_exactly(self) -> None:
        import numpy as np

        image = np.arange(700 * 900 * 3, dtype=np.uint32).reshape(700, 900, 3) % 256
        image = image.astype(np.uint8)
        restored, metadata = tiled_restore(
            image,
            lambda tile, _x, _y: tile.astype(np.float32),
            tile_size=512,
            overlap=128,
        )
        self.assertTrue(np.array_equal(np.rint(restored).astype(np.uint8), image))
        self.assertEqual(metadata["tile_count"], 6)
        self.assertEqual(tile_starts(900, 512, 128), [0, 194, 388])

    def test_curriculum_sampler_balances_composition_and_category(self) -> None:
        rows = synthetic_rows("train", groups_per_category=20)
        pools = build_curriculum_pools(rows)
        selected = [
            rows[
                curriculum_record_index(
                    pools,
                    maximum_composition_size=3,
                    presentation_index=index,
                    seed=5,
                )
            ]
            for index in range(330)
        ]
        composition = Counter(int(row["composition_size"]) for row in selected)
        self.assertEqual(composition, {1: 110, 2: 110, 3: 110})
        triple_categories = Counter(
            row["category"] for row in selected if row["composition_size"] == "3"
        )
        self.assertEqual(triple_categories, {"18Lowlight_Blur_Noise": 110})

    def test_fold_assignment_is_deterministic(self) -> None:
        first = fold_for_group("scene-a", seed=4, folds=3)
        self.assertEqual(first, fold_for_group("scene-a", seed=4, folds=3))
        self.assertIn(first, (0, 1, 2))

    def test_balanced_subset_is_fold_and_group_disjoint(self) -> None:
        # Generate enough groups, then duplicate each row to verify that only
        # one record per source component can enter the frozen subset.
        rows = synthetic_rows("validation", groups_per_category=80)
        rows.extend({**row, "record_id": row["record_id"] + "-duplicate"} for row in list(rows))
        subset = balanced_evaluation_subset(
            rows,
            split="validation",
            records_per_category_per_fold=3,
            folds=3,
            seed=19,
        )
        self.assertEqual(len(subset), 3 * 3 * len(FIVE_TYPE_CATEGORIES))
        self.assertEqual(len({row["source_group"] for row in subset}), len(subset))
        counts = Counter((row["evaluation_fold"], row["category"]) for row in subset)
        self.assertTrue(all(value == 3 for value in counts.values()))

    def test_task_metrics_include_composite_records(self) -> None:
        rows = [
            {"category": category, "psnr": 30.0, "lpips": 0.1}
            for category in FIVE_TYPE_CATEGORIES
        ]
        metrics = aggregate_task_metrics(rows)
        self.assertEqual(set(metrics), set(SOW_STAGE2_THRESHOLDS))
        self.assertGreater(metrics["low_light"]["records"], metrics["rain"]["records"])

    def test_checkpoint_selection_maximizes_worst_margin_then_earlier_step(self) -> None:
        def candidate(step: int, offset: float) -> dict[str, object]:
            task_metrics = {
                task: {
                    "psnr": thresholds["psnr"] + offset,
                    "lpips": thresholds["lpips"] - 0.01 * offset,
                }
                for task, thresholds in SOW_STAGE2_THRESHOLDS.items()
            }
            return {"step": step, "task_metrics": task_metrics}

        weak = candidate(100, -0.2)
        strong_early = candidate(200, 0.5)
        strong_late = candidate(300, 0.5)
        self.assertLess(worst_normalized_margin(weak["task_metrics"]), 0)
        self.assertIs(select_checkpoint([weak, strong_late, strong_early]), strong_early)

    def test_curriculum_sampler_resume_matches_uninterrupted_suffix(self) -> None:
        rows = synthetic_rows("train", groups_per_category=20)
        complete = list(
            CurriculumSampler(
                rows,
                maximum_composition_size=3,
                stage_presentations=96,
                stage_presentation_offset=0,
                stage_sampler_offset=0,
                rank=0,
                world=8,
                seed=7,
            )
        )
        resumed = list(
            CurriculumSampler(
                rows,
                maximum_composition_size=3,
                stage_presentations=64,
                stage_presentation_offset=32,
                stage_sampler_offset=32,
                rank=0,
                world=8,
                seed=7,
            )
        )
        self.assertEqual([record for record, _ in resumed], [record for record, _ in complete[4:]])
        self.assertEqual([presentation for _, presentation in resumed], [presentation for _, presentation in complete[4:]])

    def test_evaluation_contract_freezes_disjoint_development_and_internal_groups(self) -> None:
        rows = synthetic_rows("validation", groups_per_category=80)
        rows.extend(synthetic_rows("internal_test", groups_per_category=80))
        for row in rows:
            row.update(
                lq_path="lq.png",
                gt_path="gt.png",
                lq_sha256="a" * 64,
                gt_sha256="b" * 64,
            )
        selected = contract_rows(
            {
                "evaluation": {"folds": 3, "records_per_category_per_fold": 3},
                "data": {"split_seed": 11},
            },
            rows,
        )
        self.assertEqual(len(selected), 2 * 3 * 3 * len(FIVE_TYPE_CATEGORIES))
        development = {row["source_group"] for row in selected if row["evaluation_role"] == "development"}
        internal = {row["source_group"] for row in selected if row["evaluation_role"] == "sealed_internal_test"}
        self.assertFalse(development & internal)

    def test_file_based_selection_uses_development_only(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text("{}\n")
            config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
            results = []
            for step, offset in ((100, 0.1), (200, 0.4)):
                checkpoint = root / f"step_{step:06d}.safetensors"
                checkpoint.write_bytes(f"checkpoint-{step}".encode())
                task_metrics = {
                    task: {
                        "psnr": threshold["psnr"] + offset,
                        "lpips": threshold["lpips"] - offset * 0.01,
                    }
                    for task, threshold in SOW_STAGE2_THRESHOLDS.items()
                }
                result = root / f"result_{step}.json"
                result.write_text(
                    json.dumps(
                        {
                            "status": "development_evaluation_complete",
                            "config_sha256": config_sha,
                            "split": "validation",
                            "arm": "qwen",
                            "protected_foundir_test_accessed": False,
                            "checkpoint": checkpoint.relative_to(WORKSPACE_ROOT).as_posix(),
                            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            "summary": {"task_metrics": task_metrics},
                        }
                    )
                )
                results.append(result)
            payload = select(config, results)
            self.assertEqual(payload["selected_step"], 200)
            self.assertFalse(payload["sealed_internal_test_accessed"])


if __name__ == "__main__":
    unittest.main()
