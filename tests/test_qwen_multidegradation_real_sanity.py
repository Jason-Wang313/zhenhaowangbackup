import json
import unittest

from lowlight_restoration.multidegradation_data import sha256
from smoke.finalize_qwen_multidegradation_real_sanity_v2 import ROOT, validate_records
from smoke.render_qwen_multidegradation_real_sanity import render_sheet


class QwenMultiDegradationRealSanityTests(unittest.TestCase):
    def test_frozen_v3_manifest_is_exact_hash_verified_and_preinference(self):
        path = ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json"
        if not path.is_file():
            self.skipTest("real sanity v3 finalizer has not run")
        payload = json.loads(path.read_text())
        self.assertEqual(
            payload["status"], "frozen_before_qwen_inference_official_sidd_bytes"
        )
        self.assertFalse(payload["foundir_used"])
        self.assertFalse(payload["validation"]["qwen_inference_before_freeze"])
        validation = validate_records(payload["records"])
        self.assertEqual(validation["records"], 25)
        self.assertFalse(validation["foundir_used"])
        self.assertEqual(
            {record["expected_primary_type"] for record in payload["records"]},
            {"blur", "noise", "rain", "haze", "low_light"},
        )

    def test_blur_uses_prompt_proven_blurred_after_side_from_five_scenes(self):
        payload = json.loads(
            (ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json").read_text()
        )
        records = [record for record in payload["records"] if record["expected_primary_type"] == "blur"]
        self.assertEqual(len(records), 5)
        self.assertEqual(len({record["source_key"] for record in records}), 5)
        for record in records:
            self.assertTrue(record["source_key"].endswith(":after"))
            self.assertIn("blur", record["prompt_proving_transform_direction"].lower())
            self.assertLess(record["after_to_before_sharpness_ratio"], 0.8)

    def test_noise_is_five_distinct_official_sidd_srgb_scenes(self):
        payload = json.loads(
            (ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json").read_text()
        )
        records = [record for record in payload["records"] if record["expected_primary_type"] == "noise"]
        self.assertEqual(len(records), 5)
        self.assertEqual(len({record["revision"] for record in records}), 5)
        self.assertTrue(all(record["dataset"] == "SIDD Full real noisy sRGB" for record in records))
        self.assertTrue(all("research_use_only" in record["license"] for record in records))

    def test_all_five_sidd_noise_files_match_exact_official_members(self):
        verification_path = (
            ROOT / "manifests/qwen_multidegradation_sidd_official_verification_v3.json"
        )
        verification = json.loads(verification_path.read_text())
        self.assertEqual(verification["status"], "all_five_exact_official_member_matches")
        self.assertFalse(verification["foundir_used"])
        self.assertEqual(len(verification["records"]), 5)
        for result in verification["records"]:
            self.assertTrue(result["exact_match"])
            self.assertEqual(result["local_sha256"], result["official_member_sha256"])
            self.assertEqual(sha256(ROOT / result["local_path"]), result["local_sha256"])

    def test_v1_is_retained_as_rejected_preinference_screening(self):
        payload = json.loads(
            (ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json").read_text()
        )
        rejection = payload["rejected_screening_evidence"]
        self.assertEqual(rejection["status"], "rejected_preinference_input_screening")
        self.assertTrue((ROOT / rejection["manifest"]).is_file())

    def test_renderer_makes_a_readable_five_row_sheet(self):
        path = ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json"
        if not path.is_file():
            self.skipTest("real sanity acquisition has not run")
        records = json.loads(path.read_text())["records"]
        subset = [record for record in records if record["expected_primary_type"] == "haze"]
        sheet = render_sheet(subset, "haze")
        self.assertEqual(sheet.size, (1500, 1620))


if __name__ == "__main__":
    unittest.main()
