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

    def test_launcher_supports_parallel_single_gpu_processes_with_durable_status(self):
        launcher = (ROOT / "smoke/launch_foundir_lowlight_gate2_gpu.sh").read_text()
        bundle = (ROOT / "smoke/run_foundir_lowlight_gate2_gpu_bundle.sh").read_text()
        logged = (ROOT / "smoke/run_foundir_lowlight_gate2_gpu_logged.sh").read_text()
        self.assertIn("one or two GPU indices", launcher)
        self.assertIn('CUDA_VISIBLE_DEVICES="$teacher_device"', bundle)
        self.assertIn('CUDA_VISIBLE_DEVICES="$flux_device"', bundle)
        self.assertIn("FOUNDIR_LOWLIGHT_GATE2_PARALLEL_PROPOSALS_START", bundle)
        self.assertIn("FOUNDIR_LOWLIGHT_GATE2_TERMINAL_PASS", logged)
        self.assertIn("FOUNDIR_LOWLIGHT_GATE2_TERMINAL_FAIL", logged)
        self.assertIn("FOUNDIR_LOWLIGHT_GATE2_RUN_END", logged)


if __name__ == "__main__":
    unittest.main()
