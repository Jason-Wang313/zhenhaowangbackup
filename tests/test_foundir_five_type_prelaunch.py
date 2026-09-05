from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from lowlight_restoration.foundir_five_type import (
    DEGRADATION_ORDER,
    FIVE_TYPE_CATEGORIES,
)
from smoke import foundir_five_type_runtime_preflight as runtime_preflight
from smoke.evaluate_foundir_five_type_qwen_training_gate import (
    calculate,
    checks,
    evaluate_candidates,
)
from smoke.export_foundir_five_type_qwen_per_type_pilot import verify_existing


ROOT = Path(__file__).resolve().parents[1]


def exact_prediction(record_id: str, category: str) -> dict[str, object]:
    truth = set(FIVE_TYPE_CATEGORIES[category])
    return {
        "record_id": record_id,
        "category": category,
        "composition_size": len(truth),
        "valid": True,
        "levels": {
            name: 2 if name in truth else 1 for name in DEGRADATION_ORDER
        },
    }


class FoundIRFiveTypePrelaunchTests(unittest.TestCase):
    def test_metrics_are_exact_for_all_categories(self) -> None:
        predictions = [
            exact_prediction(f"record-{index}", category)
            for index, category in enumerate(FIVE_TYPE_CATEGORIES)
        ]
        metrics = calculate(predictions)
        self.assertEqual(metrics["valid_output_rate"], 1.0)
        self.assertEqual(metrics["type_micro_f1"], 1.0)
        self.assertEqual(metrics["type_macro_f1"], 1.0)
        self.assertEqual(metrics["degradation_set_exact"], 1.0)
        self.assertEqual(metrics["minimum_type_recall"], 1.0)

    def test_invalid_answer_counts_as_a_miss(self) -> None:
        row = exact_prediction("r1", "01Blur")
        row.update(valid=False, levels=None)
        metrics = calculate([row])
        self.assertEqual(metrics["valid_output_rate"], 0.0)
        self.assertEqual(metrics["type_micro_recall"], 0.0)
        self.assertEqual(metrics["degradation_set_exact"], 0.0)
        self.assertEqual(metrics["per_type"]["blur"]["fn"], 1)

    def test_thresholds_and_selection_are_fail_closed(self) -> None:
        exact = [exact_prediction("r1", "01Blur")]
        wrong = [exact_prediction("r1", "05Noise")]
        wrong[0]["category"] = "01Blur"
        thresholds = {
            "minimum_valid_output_rate": 0.99,
            "minimum_type_micro_f1": 0.80,
            "minimum_type_macro_f1": 0.10,
            "minimum_degradation_set_exact": 0.60,
            "minimum_each_type_recall": 0.0,
        }
        results, selected = evaluate_candidates(
            {"joint": wrong, "per_type": exact},
            records=1,
            thresholds=thresholds,
            selection_rule=(
                "degradation_set_exact",
                "minimum_type_recall",
                "type_micro_f1",
                "valid_output_rate",
            ),
        )
        self.assertFalse(results["joint"]["automatic_gate_pass"])
        self.assertTrue(results["per_type"]["automatic_gate_pass"])
        self.assertEqual(selected, "per_type")
        self.assertTrue(all(checks(calculate(exact), thresholds).values()))

    def test_candidate_order_and_cardinality_must_be_identical(self) -> None:
        first = [
            exact_prediction("r1", "01Blur"),
            exact_prediction("r2", "05Noise"),
        ]
        with self.assertRaisesRegex(RuntimeError, "identical records"):
            evaluate_candidates(
                {"joint": first, "per_type": list(reversed(first))},
                records=2,
                thresholds={
                    "minimum_valid_output_rate": 0.0,
                    "minimum_type_micro_f1": 0.0,
                    "minimum_type_macro_f1": 0.0,
                    "minimum_degradation_set_exact": 0.0,
                    "minimum_each_type_recall": 0.0,
                },
                selection_rule=("degradation_set_exact",),
            )

    def test_offline_environment_and_alexnet_hash_are_enforced(self) -> None:
        environment = {
            "TORCH_HOME": str(ROOT / "models/torch"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
        with patch.dict(os.environ, environment, clear=False), patch.object(
            runtime_preflight,
            "sha256",
            return_value=runtime_preflight.ALEXNET_SHA256,
        ):
            result = runtime_preflight.verify_environment()
        self.assertEqual(result["alexnet_sha256"], runtime_preflight.ALEXNET_SHA256)

        broken = dict(environment, HF_DATASETS_OFFLINE="0")
        with patch.dict(os.environ, broken, clear=False), self.assertRaisesRegex(
            RuntimeError, "offline environment"
        ):
            runtime_preflight.verify_environment()

        with patch.dict(os.environ, environment, clear=False), patch.object(
            runtime_preflight, "sha256", return_value="0" * 64
        ), self.assertRaisesRegex(RuntimeError, "AlexNet"):
            runtime_preflight.verify_environment()

    def test_per_type_cache_identity_is_immutable(self) -> None:
        identity = {
            "schema_version": 1,
            "status": "frozen_per_type_qwen_foundir_training_predictions",
            "records": 1,
            "protected_foundir_test_accessed": False,
        }
        payload = {
            **identity,
            "valid_records": 1,
            "predictions": [exact_prediction("r1", "01Blur")],
        }
        temporary = ROOT / "logs/.test_foundir_five_type_per_type_identity.json"
        try:
            temporary.write_text(json.dumps(payload))
            self.assertEqual(verify_existing(temporary, identity), payload)
            payload["records"] = 2
            temporary.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                verify_existing(temporary, identity)
        finally:
            temporary.unlink(missing_ok=True)

    def test_every_gpu_runner_exports_offline_workspace_caches(self) -> None:
        runners = (
            "smoke/run_foundir_five_type_prelaunch_8gpu.sh",
            "smoke/run_foundir_five_type_stage0_8gpu.sh",
            "smoke/run_foundir_five_type_flux2_full_8gpu.sh",
        )
        for relative in runners:
            with self.subTest(runner=relative):
                text = (ROOT / relative).read_text()
                self.assertIn('export TORCH_HOME="$ROOT/models/torch"', text)
                self.assertIn("HF_DATASETS_OFFLINE=1", text)
                self.assertIn('TRITON_CACHE_DIR="$ROOT/.cache/triton/foundir_five_type"', text)

    def test_prelaunch_order_and_full_training_blocker_are_explicit(self) -> None:
        prelaunch = (ROOT / "smoke/run_foundir_five_type_prelaunch_8gpu.sh").read_text()
        runtime = prelaunch.index("foundir_five_type_runtime_preflight.py")
        canary = prelaunch.index("foundir_five_type_distributed_canary.py")
        manifest = prelaunch.index("build_foundir_five_type_manifest.py")
        per_type = prelaunch.index("export_foundir_five_type_qwen_per_type_pilot.py")
        gate = prelaunch.index("evaluate_foundir_five_type_qwen_training_gate.py")
        self.assertLess(runtime, canary)
        self.assertLess(canary, manifest)
        self.assertLess(manifest, per_type)
        self.assertLess(per_type, gate)

        launcher = (ROOT / "smoke/launch_foundir_five_type_prelaunch_8gpu.sh").read_text()
        self.assertIn('"${1:-}" == "--check"', launcher)
        self.assertIn("FOUNDIR_FIVE_TYPE_PRELAUNCH_LAUNCHER_CHECK_PASS", launcher)

        for relative in (
            "smoke/run_foundir_five_type_stage0_8gpu.sh",
            "smoke/run_foundir_five_type_flux2_full_8gpu.sh",
            "smoke/launch_foundir_five_type_flux2_full_8gpu.sh",
        ):
            text = (ROOT / relative).read_text()
            self.assertIn("--require-selected joint", text)


if __name__ == "__main__":
    unittest.main()
