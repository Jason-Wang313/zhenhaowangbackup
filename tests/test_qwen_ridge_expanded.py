from __future__ import annotations

import unittest
from pathlib import Path

from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_ridge_expanded import (
    source_dataset_counts,
    stratified_source_partition,
    stratified_unexposed_selection,
)
from train_qwen_lowlight_ridge_expanded_head import load_config, resolve_records

ROOT = Path(__file__).resolve().parents[1]


class QwenExpandedRidgeTests(unittest.TestCase):
    @staticmethod
    def sequences() -> tuple[list[list[PairRecord]], dict[str, str]]:
        sequences: list[list[PairRecord]] = []
        datasets: dict[str, str] = {}
        for dataset in ("div2k", "lsdir", "mio100"):
            for source in range(4):
                source_id = f"{dataset}_{source:02d}"
                datasets[source_id] = dataset
                sequences.append(
                    [
                        PairRecord(
                            pair_id=f"{source_id}_L{level}",
                            source_id=source_id,
                            clean_path=Path("clean.png"),
                            degraded_path=Path("degraded.png"),
                            project_level=f"L{level}",
                            seed=source,
                            split="train",
                            clean_sha256="a" * 64,
                            degraded_sha256=f"{source * 5 + level:064x}",
                        )
                        for level in range(1, 6)
                    ]
                )
        return sequences, datasets

    def test_stratified_partition_is_deterministic_and_exact(self) -> None:
        sequences, datasets = self.sequences()
        first = stratified_source_partition(
            sequences,
            source_datasets=datasets,
            selection_quotas={"div2k": 1, "lsdir": 1, "mio100": 1},
            seed=17,
        )
        second = stratified_source_partition(
            sequences,
            source_datasets=datasets,
            selection_quotas={"div2k": 1, "lsdir": 1, "mio100": 1},
            seed=17,
        )
        self.assertEqual(
            [[sequence[0].source_id for sequence in group] for group in first],
            [[sequence[0].source_id for sequence in group] for group in second],
        )
        fit, selection = first
        self.assertEqual(
            source_dataset_counts(fit, datasets), {"div2k": 3, "lsdir": 3, "mio100": 3}
        )
        self.assertEqual(
            source_dataset_counts(selection, datasets),
            {"div2k": 1, "lsdir": 1, "mio100": 1},
        )
        self.assertFalse(
            {sequence[0].source_id for sequence in fit}
            & {sequence[0].source_id for sequence in selection}
        )

    def test_unexposed_selection_excludes_every_declared_source(self) -> None:
        sequences, datasets = self.sequences()
        excluded = {"div2k_00", "lsdir_00", "mio100_00"}
        selected = stratified_unexposed_selection(
            sequences,
            source_datasets=datasets,
            excluded_source_ids=excluded,
            selection_quotas={"div2k": 1, "lsdir": 1, "mio100": 1},
            seed=23,
        )
        realized = {sequence[0].source_id for sequence in selected}
        self.assertFalse(realized & excluded)
        self.assertEqual(
            source_dataset_counts(selected, datasets),
            {"div2k": 1, "lsdir": 1, "mio100": 1},
        )
        with self.assertRaises(ValueError):
            stratified_unexposed_selection(
                sequences,
                source_datasets=datasets,
                excluded_source_ids={"missing_source"},
                selection_quotas={"div2k": 1, "lsdir": 1, "mio100": 1},
                seed=23,
            )

    def test_frozen_expanded_config_and_bundles_resolve(self) -> None:
        config, digest, _, failed = load_config(
            ROOT / "configs/qwen_lowlight_ridge_expanded_head.json"
        )
        self.assertEqual(len(digest), 64)
        self.assertEqual(config["features"]["extraction_batch_size"], 20)  # type: ignore[index]
        _, _, fit, selection, validation, exposed = resolve_records(config, failed)
        self.assertEqual((len(fit), len(selection), len(validation)), (6400, 1600, 250))
        self.assertEqual(len(exposed), 25)
        self.assertFalse({record.source_id for record in validation} & exposed)


if __name__ == "__main__":
    unittest.main()
