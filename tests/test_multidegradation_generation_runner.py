import json
import unittest

from smoke.generate_qwen_multidegradation_pilot import ROOT, generate_or_verify
from lowlight_restoration.multidegradation_data import read_jsonl


class MultiDegradationGenerationRunnerTests(unittest.TestCase):
    def test_existing_smoke_output_becomes_resumable_evidence(self):
        records = read_jsonl(ROOT / "manifests/qwen_multidegradation_pilot_v1.jsonl")
        record = records[0]
        first = generate_or_verify(record)
        second = generate_or_verify(record)
        self.assertEqual(first["output_sha256"], second["output_sha256"])
        self.assertTrue(second["reused"])
        evidence = (
            ROOT
            / "data/qwen_multidegradation_pilot_v1/.generation/train"
            / f"{record['record_id']}.json"
        )
        self.assertEqual(json.loads(evidence.read_text())["record_id"], record["record_id"])


if __name__ == "__main__":
    unittest.main()
