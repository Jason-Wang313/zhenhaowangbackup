from __future__ import annotations

import inspect
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from smoke import foundir_five_type_runtime_preflight as runtime_preflight
from smoke.export_foundir_five_type_physical_v8_conditions_ddp import (
    center_generation_records,
    chunk_identity,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import evaluate_arms


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_five_type_flux2_dit_v2.json"


class FoundIRFiveTypeMainV2Tests(unittest.TestCase):
    def test_config_is_no_jpeg_no_ablation_physical_v8(self) -> None:
        config = json.loads(CONFIG.read_text())
        self.assertTrue(config["scope"]["exclude_jpeg"])
        self.assertEqual(config["scope"]["degradations"], ["blur", "noise", "rain", "haze", "low_light"])
        self.assertEqual(config["evaluation"]["condition_arms"], ["qwen"])
        self.assertEqual(config["full_training"]["generic_condition_probability"], 0.0)
        self.assertEqual(config["qwen_condition"]["method"], "per_type_dialogue_center_crop_448")
        self.assertIn("physical_v8", config["qwen_condition"]["training_config"])
        self.assertNotIn("postcodec", json.dumps(config).lower())

    def test_center_crop_is_exact_and_deterministic(self) -> None:
        row = {
            "record_id": "r1",
            "lq_path": "image.png",
            "width": "1920",
            "height": "1080",
        }
        result = center_generation_records([row], 448)[0]
        self.assertEqual(result["crop_box"], [736, 316, 1184, 764])
        self.assertEqual(result["output_path"], "image.png")

    def test_chunk_identity_binds_inputs_and_excludes_jpeg(self) -> None:
        rows = [
            {
                "record_id": "r1",
                "lq_sha256": "1" * 64,
                "gt_sha256": "2" * 64,
            }
        ]
        identity = chunk_identity(
            rows, config_sha="3" * 64, qwen_config_sha="4" * 64, adapter_sha="5" * 64
        )
        self.assertEqual(identity["query_count_per_image"], 5)
        self.assertFalse(identity["jpeg_included"])
        self.assertFalse(identity["protected_foundir_test_accessed"])

    def test_overfit_uses_configured_arms(self) -> None:
        source = inspect.getsource(evaluate_arms)
        self.assertIn('config["evaluation"].get("condition_arms"', source)
        self.assertIn("for arm in configured_arms", source)

    def test_main_runner_has_no_condition_ablation_commands(self) -> None:
        runner = (ROOT / "smoke/run_foundir_five_type_flux2_main_v2_8gpu.sh").read_text()
        self.assertIn("export_foundir_five_type_physical_v8_conditions_ddp.py", runner)
        self.assertNotIn("--arm generic", runner)
        self.assertNotIn("--arm shuffled_wrong", runner)
        self.assertNotIn("--arm type_oracle", runner)
        self.assertNotIn("postcodec", runner.lower())
        self.assertNotIn("MillionIRData/Test", runner)

    def test_evaluator_rejects_disabled_condition_arms(self) -> None:
        source = (ROOT / "smoke/evaluate_foundir_five_type_flux2_ddp.py").read_text()
        self.assertIn("condition arm {args.arm!r} is disabled", source)

    def test_specialist_probe_uses_an_isolated_pinned_import_path(self) -> None:
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}, clear=False):
            environment = runtime_preflight.specialist_probe_environment()
        entries = environment["PYTHONPATH"].split(os.pathsep)
        expected = [
            str(runtime_preflight.workspace_path(path))
            for path in runtime_preflight.SPECIALIST_DEPENDENCY_RELATIVES
        ]
        self.assertEqual(entries[: len(expected)], expected)
        self.assertEqual(entries[len(expected)], str(ROOT))
        self.assertEqual(entries[-1], "/existing/path")

    def test_specialist_probe_accepts_only_verified_child_marker(self) -> None:
        payload = {
            "versions": runtime_preflight.SPECIALIST_EXPECTED_VERSIONS,
            "architecture": "WaveMamba",
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=runtime_preflight.SPECIALIST_PROBE_MARKER + json.dumps(payload) + "\n",
            stderr="",
        )
        with patch.object(runtime_preflight.subprocess, "run", return_value=completed):
            self.assertEqual(runtime_preflight.verify_specialist_runtime(), payload)

        malformed = SimpleNamespace(returncode=0, stdout="no marker\n", stderr="")
        with patch.object(runtime_preflight.subprocess, "run", return_value=malformed), self.assertRaisesRegex(
            RuntimeError, "no pass marker"
        ):
            runtime_preflight.verify_specialist_runtime()

        failed = SimpleNamespace(returncode=1, stdout="", stderr="dependency failure")
        with patch.object(runtime_preflight.subprocess, "run", return_value=failed), self.assertRaisesRegex(
            RuntimeError, "dependency failure"
        ):
            runtime_preflight.verify_specialist_runtime()

    def test_launcher_requires_eight_explicit_devices(self) -> None:
        launcher = (ROOT / "smoke/launch_foundir_five_type_flux2_main_v2_8gpu.sh").read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7"', launcher)
        self.assertNotIn("--require-data", launcher)
        self.assertIn("install.complete", launcher)
        self.assertIn('"${1:-}" == "--check"', launcher)


if __name__ == "__main__":
    unittest.main()
