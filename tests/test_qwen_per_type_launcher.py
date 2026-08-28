from __future__ import annotations

import unittest
from pathlib import Path

from smoke.train_qwen_multidegradation_lora_ddp import evaluation_artifact_paths


ROOT = Path(__file__).resolve().parents[1]


class QwenPerTypeLauncherTests(unittest.TestCase):
    def test_candidate_validation_artifacts_do_not_collide(self):
        output = Path("/tmp/qwen-candidate-artifacts")
        h5_predictions, h5_result = evaluation_artifact_paths(
            output,
            split="validation",
            validation_stage_name="H5_per_type_dialogue_epoch5",
        )
        h6_predictions, h6_result = evaluation_artifact_paths(
            output,
            split="validation",
            validation_stage_name="H6_per_type_dialogue_epoch6",
        )
        h8_predictions, h8_result = evaluation_artifact_paths(
            output,
            split="validation",
            validation_stage_name=None,
        )
        self.assertEqual(len({h5_predictions, h6_predictions, h8_predictions}), 3)
        self.assertEqual(len({h5_result, h6_result, h8_result}), 3)
        self.assertTrue(h5_predictions.name.endswith("H5_per_type_dialogue_epoch5.json"))
        self.assertTrue(h6_predictions.name.endswith("H6_per_type_dialogue_epoch6.json"))
        self.assertEqual(h8_predictions.name, "validation_predictions.json")

    def test_runner_is_fail_closed_and_test_blind_until_validation_passes(self):
        runner = (
            ROOT / "smoke/run_qwen_multidegradation_per_type_rank32.sh"
        ).read_text()
        self.assertIn("0,1) NPROC=2", runner)
        self.assertIn("0,1,2,3) NPROC=4", runner)
        overfit = runner.index("train_qwen_multidegradation_overfit_ddp.py")
        validation = runner.index(
            "train_qwen_multidegradation_lora_ddp.py --config \"$CONFIG\" --variant"
        )
        selection = runner.index("select_qwen_multidegradation_lora.py")
        fixed_test = runner.index("--test-only")
        self.assertLess(overfit, validation)
        self.assertLess(validation, selection)
        self.assertLess(selection, fixed_test)
        self.assertIn("VALIDATION_SCIENTIFIC_FAILURE", runner)
        self.assertIn("exit 2", runner)
        self.assertIn("VALIDATION_ONLY_RECOVERY", runner)
        self.assertIn("validation_mode=(--validation-only)", runner)

    def test_launcher_has_persistent_tmux_or_nohup_evidence(self):
        launcher = (
            ROOT / "smoke/launch_qwen_multidegradation_per_type_rank32.sh"
        ).read_text()
        self.assertIn("remain-on-exit", launcher)
        self.assertIn("nohup env CUDA_VISIBLE_DEVICES", launcher)
        self.assertIn("qwen_multidegradation_per_type_rank32_v1.log", launcher)

    def test_continuation_launcher_supports_two_and_eight_h100s(self):
        runner = (
            ROOT
            / "smoke/run_qwen_multidegradation_per_type_rank32_continue.sh"
        ).read_text()
        launcher = (
            ROOT
            / "smoke/launch_qwen_multidegradation_per_type_rank32_continue.sh"
        ).read_text()
        for text in (runner, launcher):
            self.assertIn("0,1)", text)
            self.assertIn("0,1,2,3,4,5,6,7)", text)
        self.assertIn("ACCUMULATION=4", runner)
        self.assertIn("ACCUMULATION=1", runner)
        self.assertIn("global=200", runner)
        self.assertIn("H5_per_type_dialogue_epoch5", runner)
        self.assertIn("H6_per_type_dialogue_epoch6", runner)
        self.assertIn("--test-only", runner)
        self.assertIn("remain-on-exit", launcher)
        self.assertIn("nohup env CUDA_VISIBLE_DEVICES", launcher)


if __name__ == "__main__":
    unittest.main()
