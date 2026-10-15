from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from smoke.prepare_cure_two_track_v1 import (
    COMMON,
    RESTORATION,
    ROOT,
    TRIPLES,
    VOCAB,
    prepare,
    split_name,
)


class CURETwoTrackPreparationTest(unittest.TestCase):
    def test_views_are_source_disjoint_and_strict_excludes_triples(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            base = Path(temporary)
            dataset = base / "dataset"
            output = base / "views"
            names = {}
            for candidate in range(100):
                name = f"source_{candidate:03d}.png"
                names.setdefault(split_name(name), name)
                if len(names) == 2:
                    break
            self.assertEqual(set(names), {"train", "development"})
            for index, name in enumerate(names.values()):
                stem = Path(name).stem
                main = dataset / "half_train/main_data"
                for category in (*VOCAB, "zero"):
                    path = main / category / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (2, 2), (index * 50 + len(category), 4, 7)).save(path)
                for category in RESTORATION:
                    parts = category.split("_")
                    factors = (parts[0], "zero") if len(parts) == 1 else (parts[0], parts[1])
                    for suffix in (*factors, "half"):
                        path = dataset / "half_train/sub_data" / category / stem / f"{stem}_{suffix}_.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        Image.new("RGB", (2, 2), (index + len(suffix), 3, 5)).save(path)
            test_main = dataset / "half_test/main_data"
            for category in VOCAB:
                path = test_main / category / "test_source.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2), (200 + len(category), 8, 9)).save(path)
            metadata = prepare(dataset, output, check_only=False)
            self.assertEqual(metadata["train_sources"], 1)
            self.assertEqual(metadata["development_sources"], 1)
            self.assertEqual(metadata["embedder_presentations_per_epoch_each_track"], 12)
            self.assertEqual(metadata["strict_triple_training_samples"], 0)
            self.assertEqual(metadata["view_link_counts"]["embedder_dev_common"], len(COMMON))
            for category in TRIPLES:
                self.assertFalse(list((output / "embedder_strict_train/main_data" / category).iterdir()))
            self.assertEqual(prepare(dataset, output, check_only=True), metadata)


if __name__ == "__main__":
    unittest.main()
