from __future__ import annotations

import copy
import json
import unittest

from lowlight_restoration.multidegradation_data import ROOT
from smoke.generate_qwen_multidegradation_visibility_v2 import (
    expected_visibility_hash,
    render_verified_png,
)


class MultiDegradationVisibilityGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pilot = json.loads(
            (
                ROOT / "reports/qwen_multidegradation_visibility_v2_final_pilot.json"
            ).read_text()
        )
        result = pilot["results"][0]
        cls.record = {**result["record"], "visibility_evidence": result["visibility"]}

    def test_passing_pilot_render_matches_visibility_hash(self):
        expected = expected_visibility_hash(self.record)
        payload, result = render_verified_png(self.record)
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(result["output_sha256"], expected)
        self.assertEqual(result["visibility_full_sha256"], expected)

    def test_inconsistent_visibility_hash_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["visibility_evidence"][0]["full_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "inconsistent full hashes"):
            expected_visibility_hash(record)


if __name__ == "__main__":
    unittest.main()
