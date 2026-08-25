from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QwenPerTypeLauncherTests(unittest.TestCase):
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

    def test_launcher_has_persistent_tmux_or_nohup_evidence(self):
        launcher = (
            ROOT / "smoke/launch_qwen_multidegradation_per_type_rank32.sh"
        ).read_text()
        self.assertIn("remain-on-exit", launcher)
        self.assertIn("nohup env CUDA_VISIBLE_DEVICES", launcher)
        self.assertIn("qwen_multidegradation_per_type_rank32_v1.log", launcher)


if __name__ == "__main__":
    unittest.main()
