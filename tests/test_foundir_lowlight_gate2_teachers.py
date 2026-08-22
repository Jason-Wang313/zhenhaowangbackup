import json
import unittest
from pathlib import Path

from smoke.run_foundir_lowlight_gate2_teachers import (
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    output_stem,
    read_config,
)


ROOT = Path(__file__).resolve().parents[1]


class Gate2TeacherTests(unittest.TestCase):
    def test_contract_is_non_foundir_and_hash_validated(self):
        config = read_config(ROOT / "configs/foundir_lowlight_gate2_teacher_v1b.json")
        self.assertFalse(config["protected_foundir_pixels_allowed"])
        self.assertTrue(config["specialist"]["raw_input_direct_route_forbidden"])
        self.assertEqual(config["generalist"]["sampling_steps"], 4)
        self.assertEqual(config["generalist"]["large_image_policy"]["tile_size"], 1024)
        self.assertEqual(config["specialist"]["large_image_policy"]["tile_size"], 512)

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

    def test_preflight_is_fail_closed_and_uses_official_foundir_versions(self):
        launcher = (ROOT / "smoke/launch_foundir_lowlight_gate2_gpu.sh").read_text()
        self.assertIn("(\nset -euo pipefail\n", launcher)
        self.assertIn('transformers.__version__ == "4.47.1"', launcher)
        self.assertIn('tokenizers.__version__ == "0.21.0"', launcher)
        self.assertNotIn("{\necho \"FOUNDIR_LOWLIGHT_GATE2_PREFLIGHT_START", launcher)

    def test_specialist_dependency_directories_are_workspace_local(self):
        expected = {
            ".tools/foundir-tokenizers-0.21.0",
            ".tools/foundir-huggingface-hub-0.27.1",
            ".tools/foundir-transformers-4.47.1-v2",
        }
        actual = set()
        for path in SPECIALIST_DEPENDENCY_DIRECTORIES:
            path.resolve().relative_to(ROOT)
            actual.add(str(path.relative_to(ROOT)))
        self.assertEqual(actual, expected)

    def test_resume_runs_only_missing_specialist_then_metrics(self):
        launcher = (ROOT / "smoke/launch_foundir_lowlight_gate2_gpu_resume.sh").read_text()
        runner = (ROOT / "smoke/run_foundir_lowlight_gate2_gpu_resume.sh").read_text()
        logged = (ROOT / "smoke/run_foundir_lowlight_gate2_gpu_resume_logged.sh").read_text()
        self.assertIn("--stage specialist-resume", runner)
        self.assertNotIn("run_foundir_lowlight_gate2_historical_flux", runner)
        self.assertIn("WAITING_FOR_HISTORICAL", runner)
        self.assertIn("run_foundir_lowlight_gate2_metric_blend", runner)
        self.assertIn("set -euo pipefail", launcher)
        self.assertIn("RESUME_TERMINAL_PASS", logged)
        self.assertIn("RESUME_TERMINAL_FAIL", logged)


if __name__ == "__main__":
    unittest.main()
