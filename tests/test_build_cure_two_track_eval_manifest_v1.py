from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from smoke.build_cure_two_track_eval_manifest_v1 import ROOT, TRIPLES, build


class CUREEvalManifestTest(unittest.TestCase):
    def test_seven_exact_same_source_targets_per_triple(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            for triple in TRIPLES:
                name = "sample.png"
                stem = "sample"
                for path in (
                    root / "main_data" / triple / name,
                    root / "main_data/clear" / name,
                ):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
                factors = triple.split("_")
                for retained in factors:
                    remaining = "_".join(factor for factor in factors if factor != retained)
                    for level, suffix in (("1deg", retained), ("2deg", remaining)):
                        path = root / "sub_data" / triple / stem / level / f"{stem}_{suffix}_.png"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
            content, metadata = build(root)
            rows = list(csv.DictReader(io.StringIO(content)))
            self.assertEqual(len(rows), 14)
            self.assertEqual(metadata["full_rows"], 2)
            self.assertEqual(metadata["selective_rows"], 12)
            self.assertEqual(len({row["id"] for row in rows}), 14)
            for row in rows:
                if row["kind"] == "selective":
                    self.assertIn(f"sub_data/{row['source_prompt']}/sample/", row["target_path"])


if __name__ == "__main__":
    unittest.main()
