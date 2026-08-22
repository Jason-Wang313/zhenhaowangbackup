import json
import unittest
from pathlib import Path

from smoke.run_foundir_lowlight_gate2_teachers import output_stem, read_config


ROOT = Path(__file__).resolve().parents[1]


class Gate2TeacherTests(unittest.TestCase):
    def test_contract_is_non_foundir_and_hash_validated(self):
        config = read_config(ROOT / "configs/foundir_lowlight_gate2_teacher_v1.json")
        self.assertFalse(config["protected_foundir_pixels_allowed"])
        self.assertTrue(config["specialist"]["raw_input_direct_route_forbidden"])
        self.assertEqual(config["generalist"]["sampling_steps"], 4)

    def test_output_stem_is_stable_and_path_safe(self):
        row = {"record_id": "family:scene/input", "family": "sdsd"}
        first = output_stem(7, row)
        self.assertEqual(first, output_stem(7, row))
        self.assertNotIn("/", first)
        self.assertTrue(first.startswith("0007_sdsd_"))


if __name__ == "__main__":
    unittest.main()
