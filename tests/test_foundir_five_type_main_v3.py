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
    OPTIMIZED_EXPORT_ENGINE,
    center_generation_records,
    chunk_belongs_to_partition,
    chunk_identity,
    optimized_batch_size,
    record_indices_for_rank,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import (
    aggregate_sow_overfit_metrics,
    balanced_overfit_evaluation_records,
    evaluate_arms,
    sow_overfit_checks,
)
from smoke.recheck_foundir_five_type_flux2_overfit_sow_ddp import (
    validate_historical_source,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_five_type_flux2_dit_v3.json"


class FoundIRFiveTypeMainV3Tests(unittest.TestCase):
    def test_config_is_no_jpeg_no_ablation_physical_v8(self) -> None:
        config = json.loads(CONFIG.read_text())
        self.assertTrue(config["scope"]["exclude_jpeg"])
        self.assertEqual(config["scope"]["degradations"], ["blur", "noise", "rain", "haze", "low_light"])
        self.assertEqual(config["evaluation"]["condition_arms"], ["qwen"])
        self.assertEqual(config["full_training"]["generic_condition_probability"], 0.0)
        self.assertEqual(config["qwen_condition"]["method"], "per_type_dialogue_center_crop_448")
        self.assertIn("physical_v8", config["qwen_condition"]["training_config"])
        self.assertNotIn("postcodec", json.dumps(config).lower())
        self.assertTrue(config["pixel_refinement"]["enabled"])
        self.assertEqual(config["pixel_refinement"]["expected_trainable_parameters"], 33251)
        self.assertEqual(config["model"]["restoration_adapter"], "bounded_pixel_refinement_after_vae_decode")
        self.assertTrue(config["overfit"]["require_restored_lpips_improvement"])
        self.assertEqual(config["overfit"]["output_subdirectory"], "native_reference_overfit_sow")
        self.assertNotIn("high_frequency", json.dumps(config))
        self.assertNotIn("hf_correlation", json.dumps(config))

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

    def test_optimized_export_shards_records_once_across_eight_ranks(self) -> None:
        assignments = [record_indices_for_rank(512, rank, 8) for rank in range(8)]
        self.assertEqual([len(values) for values in assignments], [64] * 8)
        self.assertEqual(sorted(value for values in assignments for value in values), list(range(512)))
        self.assertEqual(len({value for values in assignments for value in values}), 512)
        self.assertEqual(OPTIMIZED_EXPORT_ENGINE, "record_sharded_per_type_batch_v1")

    def test_multi_instance_chunks_are_disjoint_and_complete(self) -> None:
        assignments = [
            {index for index in range(1339) if chunk_belongs_to_partition(index, 8, partition)}
            for partition in range(8)
        ]
        self.assertEqual(set().union(*assignments), set(range(1339)))
        self.assertEqual(sum(len(values) for values in assignments), 1339)
        for left in range(8):
            for right in range(left + 1, 8):
                self.assertFalse(assignments[left] & assignments[right])

    def test_optimized_batch_uses_one_call_per_type_for_a_full_chunk(self) -> None:
        h100 = SimpleNamespace(total_memory=80 * 1024**3)
        with patch("torch.cuda.get_device_properties", return_value=h100):
            self.assertEqual(optimized_batch_size(SimpleNamespace(), 64), 32)
            self.assertEqual(optimized_batch_size(SimpleNamespace(), 17), 17)
        h200 = SimpleNamespace(total_memory=140 * 1024**3)
        with patch("torch.cuda.get_device_properties", return_value=h200):
            self.assertEqual(optimized_batch_size(SimpleNamespace(), 64), 64)
        too_small = SimpleNamespace(total_memory=74 * 1024**3)
        with patch("torch.cuda.get_device_properties", return_value=too_small), self.assertRaisesRegex(
            RuntimeError, "at least 75 GiB"
        ):
            optimized_batch_size(SimpleNamespace(), 64)

    def test_overfit_uses_configured_arms(self) -> None:
        source = inspect.getsource(evaluate_arms)
        self.assertIn('config["evaluation"].get("condition_arms"', source)
        self.assertIn("for arm in configured_arms", source)
        self.assertIn("input_lpips", source)
        self.assertIn("restored_lpips", source)
        self.assertNotIn("high_frequency", source)

    def test_overfit_gate_uses_psnr_and_lpips(self) -> None:
        rows = [
            {
                "arm": "qwen",
                "input_psnr": 20.0,
                "decoded_psnr": 23.0,
                "restored_psnr": 25.0,
                "input_lpips": 0.30,
                "decoded_lpips": 0.20,
                "restored_lpips": 0.10,
            }
        ]
        metrics = aggregate_sow_overfit_metrics(rows, ["qwen"])
        self.assertAlmostEqual(metrics["qwen"]["psnr_gain_db"], 5.0)
        self.assertAlmostEqual(metrics["qwen"]["lpips_improvement"], 0.20)
        config = json.loads(CONFIG.read_text())
        checks = sow_overfit_checks(
            loss_ratio=0.1, arm_metrics=metrics, overfit=config["overfit"]
        )
        self.assertEqual(
            checks,
            {
                "loss_ratio": True,
                "restored_psnr_improves": True,
                "restored_lpips_improves": True,
            },
        )

    def test_sow_recheck_binds_immutable_v3_checkpoint(self) -> None:
        config = json.loads(CONFIG.read_text())
        checkpoint, result = validate_historical_source(config)
        self.assertEqual(checkpoint.name, "step_000200.safetensors")
        self.assertFalse(result["protected_foundir_test_accessed"])
        self.assertLess(result["probe_loss_ratio"], config["overfit"]["require_loss_ratio_at_most"])

    def test_overfit_evaluation_is_category_balanced(self) -> None:
        records = [
            {"record_id": f"{category}_{index}", "category": category}
            for category in sorted(json.loads(CONFIG.read_text())["data"]["expected_categories"])
            for index in range(12)
        ]
        selected = balanced_overfit_evaluation_records(records, 32, 17)
        self.assertEqual(len(selected), 32)
        self.assertEqual(
            {row["category"] for row in selected},
            set(json.loads(CONFIG.read_text())["data"]["expected_categories"]),
        )
        counts = {}
        for row in selected:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_main_runner_has_no_condition_ablation_commands(self) -> None:
        runner = (ROOT / "smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh").read_text()
        self.assertIn("export_foundir_five_type_physical_v8_conditions_ddp.py", runner)
        self.assertNotIn("--arm generic", runner)
        self.assertNotIn("--arm shuffled_wrong", runner)
        self.assertNotIn("--arm type_oracle", runner)
        self.assertNotIn("postcodec", runner.lower())
        self.assertNotIn("MillionIRData/Test", runner)
        exporter = (ROOT / "smoke/export_foundir_five_type_physical_v8_conditions_ddp.py").read_text()
        self.assertIn("FOUNDIR_PHYSICAL_V8_OPTIMIZATION_VALIDATION", exporter)
        self.assertIn("--benchmark-only", exporter)
        self.assertIn("FOUNDIR_PHYSICAL_V8_OPTIMIZATION_BENCHMARK_PASS", exporter)
        self.assertIn("exact_condition_equivalence", exporter)
        self.assertIn("generate_outputs_record_sharded", exporter)
        self.assertIn("FOUNDIR_PHYSICAL_V8_PARTITION_PASS", exporter)
        partition_runner = (ROOT / "smoke/run_foundir_qwen_export_partition_group_h200.sh").read_text()
        partition_launcher = (ROOT / "smoke/launch_foundir_qwen_export_partition_group_h200.sh").read_text()
        partition_stopper = (ROOT / "smoke/stop_foundir_five_type_flux2_main_v3_for_partition.sh").read_text()
        self.assertIn("PARTITIONS=8", partition_runner)
        self.assertIn("WORKERS_PER_INSTANCE=4", partition_runner)
        self.assertIn("--partition-index", partition_runner)
        self.assertIn("NVIDIA\\ H200", partition_runner)
        self.assertIn("STOPPED_FOR_PARTITION", partition_stopper)
        self.assertIn("refusing to signal unrelated PID", partition_stopper)
        self.assertNotIn("kill -KILL", partition_stopper)
        self.assertIn("stop_foundir_five_type_flux2_main_v3_for_partition.sh", partition_launcher)
        self.assertNotIn("MillionIRData/Test", partition_runner + partition_launcher)
        for relative in (
            "smoke/train_foundir_five_type_flux2_overfit_ddp.py",
            "smoke/recheck_foundir_five_type_flux2_overfit_sow_ddp.py",
            "smoke/train_foundir_five_type_flux2_full_ddp.py",
            "smoke/evaluate_foundir_five_type_flux2_ddp.py",
            "smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertIn("pixel_refinement", source)
            if "overfit" in relative:
                self.assertNotIn("high_frequency", source)
                self.assertNotIn("hf_correlation", source)
        stage0 = (ROOT / "smoke/run_foundir_five_type_stage0_v3_8gpu.sh").read_text()
        self.assertIn("recheck_foundir_five_type_flux2_overfit_sow_ddp.py", stage0)
        self.assertNotIn("evaluate_foundir_five_type_vae_ceiling_ddp.py", stage0)

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
        launcher = (ROOT / "smoke/launch_foundir_five_type_flux2_main_v3_8gpu.sh").read_text()
        self.assertIn('CUDA_VISIBLE_DEVICES:-}" == "0,1,2,3,4,5,6,7"', launcher)
        self.assertNotIn("--require-data", launcher)
        self.assertIn("install.complete", launcher)
        self.assertIn('"${1:-}" == "--check"', launcher)


if __name__ == "__main__":
    unittest.main()
