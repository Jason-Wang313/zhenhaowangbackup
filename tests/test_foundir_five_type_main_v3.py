from __future__ import annotations

import inspect
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from smoke import foundir_five_type_runtime_preflight as runtime_preflight
from smoke import evaluate_foundir_five_type_released_baselines_ddp as released_baseline
from smoke.freeze_foundir_five_type_flux2_main_v3_report import validate_released_baseline
from smoke.preflight_foundir_flux2_training_geometry import verify_evaluation_pairs
from smoke.evaluate_foundir_five_type_flux2_ddp import write_or_verify_png
from smoke.export_foundir_five_type_physical_v8_conditions_ddp import (
    OPTIMIZED_EXPORT_ENGINE,
    UNDERSIZED_INPUT_POLICY,
    center_generation_records,
    chunk_belongs_to_partition,
    chunk_identity,
    generation_geometry_summary,
    optimized_batch_size,
    record_indices_for_rank,
    resized_dimensions,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import (
    aggregate_sow_overfit_metrics,
    balanced_overfit_evaluation_records,
    evaluate_arms,
    sow_overfit_checks,
)
from smoke.train_qwen_multidegradation_lora_ddp import open_images
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
        self.assertNotIn("resize_size", result)

    def test_undersized_input_is_resized_before_exact_center_crop(self) -> None:
        row = {
            "record_id": "narrow",
            "lq_path": "image.png",
            "width": "1534",
            "height": "406",
        }
        result = center_generation_records([row], 448)[0]
        self.assertEqual(resized_dimensions(1534, 406, 448), (1693, 448))
        self.assertEqual(result["resize_size"], [1693, 448])
        self.assertEqual(result["crop_box"], [622, 0, 1070, 448])
        self.assertEqual(result["undersized_input_policy"], UNDERSIZED_INPUT_POLICY)

    def test_geometry_preflight_scans_all_dimensions(self) -> None:
        rows = [
            {"record_id": "regular", "width": "1920", "height": "1080"},
            {"record_id": "narrow", "width": "674", "height": "266"},
        ]
        summary = generation_geometry_summary(rows, 448)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["undersized_records"], 1)
        self.assertEqual(summary["minimum_width"], 674)
        self.assertEqual(summary["minimum_height"], 266)
        self.assertEqual(summary["undersized_input_policy"], UNDERSIZED_INPUT_POLICY)
        with self.assertRaisesRegex(RuntimeError, "non-positive manifest dimensions"):
            generation_geometry_summary(
                [{"record_id": "invalid", "width": "0", "height": "448"}], 448
            )

    def test_qwen_loader_resizes_before_crop(self) -> None:
        source = Image.new("RGB", (1534, 406), (32, 64, 96))
        record = {
            "record_id": "narrow",
            "output_path": "unused.png",
            "resize_size": [1693, 448],
            "crop_box": [622, 0, 1070, 448],
        }
        with patch(
            "smoke.train_qwen_multidegradation_lora_ddp.workspace_path",
            return_value=Path("unused.png"),
        ), patch(
            "smoke.train_qwen_multidegradation_lora_ddp.Image.open",
            return_value=source,
        ):
            images = open_images([record])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (448, 448))
        self.assertEqual(images[0].getpixel((0, 0)), (32, 64, 96))

    def test_chunk_identity_binds_inputs_and_excludes_jpeg(self) -> None:
        rows = [
            {
                "record_id": "r1",
                "lq_sha256": "1" * 64,
                "gt_sha256": "2" * 64,
                "width": "1920",
                "height": "1080",
            }
        ]
        identity = chunk_identity(
            rows, config_sha="3" * 64, qwen_config_sha="4" * 64, adapter_sha="5" * 64
        )
        self.assertEqual(identity["query_count_per_image"], 5)
        self.assertFalse(identity["jpeg_included"])
        self.assertFalse(identity["protected_foundir_test_accessed"])

    def test_undersized_chunk_identity_binds_fallback_policy(self) -> None:
        rows = [
            {
                "record_id": "narrow",
                "lq_sha256": "1" * 64,
                "gt_sha256": "2" * 64,
                "width": "1534",
                "height": "406",
            }
        ]
        identity = chunk_identity(
            rows, config_sha="3" * 64, qwen_config_sha="4" * 64, adapter_sha="5" * 64
        )
        self.assertEqual(identity["undersized_input_policy"], UNDERSIZED_INPUT_POLICY)
        self.assertEqual(identity["undersized_record_ids"], ["narrow"])

    def test_partition_launcher_binds_shared_qwen_image_loader(self) -> None:
        launcher = (ROOT / "smoke/launch_foundir_qwen_export_partition_group_h200.sh").read_text()
        self.assertIn("smoke/train_qwen_multidegradation_lora_ddp.py", launcher)

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
        self.assertIn('! -f "$OUTPUT/sealed_internal_consumption.json"', runner)
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

    def test_released_weather_checkpoint_uses_restormer(self) -> None:
        source = (ROOT / "smoke/evaluate_foundir_five_type_released_baselines_ddp.py").read_text()
        self.assertIn("build_weather_specialist", source)
        self.assertIn("run_weather_specialist", source)
        self.assertNotIn(
            "weather = build_specialist(released_config(config, weather_path), device)",
            source,
        )

    def test_v3_ddp_entrypoints_bind_cuda_device_before_nccl(self) -> None:
        for relative in (
            "smoke/build_foundir_flux_qwen3_prompt_cache_ddp.py",
            "smoke/recheck_foundir_five_type_flux2_overfit_sow_ddp.py",
            "smoke/train_foundir_five_type_flux2_overfit_ddp.py",
            "smoke/train_foundir_five_type_flux2_full_ddp.py",
            "smoke/evaluate_foundir_five_type_flux2_ddp.py",
            "smoke/evaluate_foundir_five_type_flux2_tiled_ddp.py",
            "smoke/evaluate_foundir_five_type_released_baselines_ddp.py",
            "smoke/train_qwen_multidegradation_lora_ddp.py",
        ):
            source = (ROOT / relative).read_text()
            self.assertLess(
                source.index("torch.cuda.set_device(local_rank)"),
                source.index("dist.init_process_group("),
                relative,
            )
            self.assertIn("device_id=device", source, relative)

    def test_released_baseline_has_complete_sidecar_recovery(self) -> None:
        evaluator = (
            ROOT / "smoke/evaluate_foundir_five_type_released_baselines_ddp.py"
        ).read_text()
        runner = (ROOT / "smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh").read_text()
        self.assertIn("finalize_existing_sidecars", evaluator)
        self.assertIn("--finalize-existing-if-complete", evaluator)
        self.assertGreaterEqual(runner.count("--finalize-existing-if-complete"), 2)
        self.assertIn("FOUNDIR_RELEASED_BASELINE_RESUME_PENDING", evaluator)

    def test_released_baseline_producer_matches_final_freezer_schema(self) -> None:
        rows = []
        for index, category in enumerate(
            json.loads(CONFIG.read_text())["data"]["expected_categories"]
        ):
            row = {
                "record_id": f"record-{index}",
                "source_group": f"group-{index}",
                "category": category,
                "composition_size": len(category.split("_")),
                "evaluation_fold": 0,
                "candidates": {},
            }
            for candidate in released_baseline.expected_candidate_names(row):
                row["candidates"][candidate] = {
                    "category": category,
                    "psnr": 30.0,
                    "lpips": 0.1,
                    "ssim": 0.9,
                    "output_path": f"unused/{candidate}-{index}.png",
                    "output_sha256": "a" * 64,
                }
            rows.append(row)
        payload = released_baseline.build_result_payload(
            rows,
            config_sha="b" * 64,
            conditions_sha="c" * 64,
            split="validation",
        )
        self.assertEqual(payload["summary"], payload["candidate_summaries"]["generalist"])
        self.assertEqual(payload["summary"]["records"], len(rows))
        task_metrics = validate_released_baseline(
            payload,
            label="synthetic baseline",
            split="validation",
            expected_records=len(rows),
            conditions_sha="c" * 64,
        )
        self.assertEqual(set(task_metrics), {"blur", "noise", "rain", "haze", "low_light"})
        payload_without_summary = dict(payload)
        payload_without_summary.pop("summary")
        with self.assertRaisesRegex(RuntimeError, "generalist summary contract"):
            validate_released_baseline(
                payload_without_summary,
                label="synthetic baseline",
                split="validation",
                expected_records=len(rows),
                conditions_sha="c" * 64,
            )

    def test_released_baseline_rejects_nonfinite_aggregate(self) -> None:
        rows = [
            {
                "category": category,
                "candidates": {
                    "generalist": {
                        "category": category,
                        "psnr": float("nan") if index == 0 else 30.0,
                        "lpips": 0.1,
                        "ssim": 0.9,
                    }
                },
            }
            for index, category in enumerate(
                json.loads(CONFIG.read_text())["data"]["expected_categories"]
            )
        ]
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            released_baseline.build_result_payload(
                rows,
                config_sha="b" * 64,
                conditions_sha="c" * 64,
                split="validation",
            )

    def test_released_sealed_baseline_requires_exact_consumption_chain(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary_root = Path(directory)
            output = temporary_root / "output"
            sealed_dir = output / "sealed_internal/qwen"
            sealed_dir.mkdir(parents=True)
            selection_path = output / "checkpoint_selection.json"
            selection = {
                "status": "frozen_development_selected_checkpoint",
                "config_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "checkpoint": "checkpoint.safetensors",
                "sealed_internal_test_accessed": False,
                "protected_foundir_test_accessed": False,
            }
            selection_path.write_text(json.dumps(selection))
            sealed = {
                "status": "sealed_internal_test_complete",
                "config_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "checkpoint": "checkpoint.safetensors",
                "conditions_sha256": "c" * 64,
                "split": "internal_test",
                "arm": "qwen",
                "protected_foundir_test_accessed": False,
            }
            (sealed_dir / "result.json").write_text(json.dumps(sealed))
            consumption = {
                "status": "sealed_internal_test_consumed",
                "config_sha256": "a" * 64,
                "checkpoint_sha256": "b" * 64,
                "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                "output": "output/sealed_internal/qwen",
                "protected_foundir_test_accessed": False,
            }
            consumption_path = output / "sealed_internal_consumption.json"
            consumption_path.write_text(json.dumps(consumption))
            with patch.object(released_baseline, "ROOT", temporary_root):
                digest = released_baseline.verify_sealed_release_order(
                    {"runtime": {"output_dir": "output"}}, "a" * 64, "c" * 64
                )
                self.assertEqual(digest, hashlib.sha256(consumption_path.read_bytes()).hexdigest())
                consumption["selection_sha256"] = "0" * 64
                consumption_path.write_text(json.dumps(consumption))
                with self.assertRaisesRegex(RuntimeError, "consumption identity changed"):
                    released_baseline.verify_sealed_release_order(
                        {"runtime": {"output_dir": "output"}}, "a" * 64, "c" * 64
                    )

    def test_evaluation_geometry_is_checked_before_gpu_inference(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            Image.new("RGB", (20, 18), (1, 2, 3)).save(root / "lq.png")
            Image.new("RGB", (20, 18), (4, 5, 6)).save(root / "gt.png")
            rows = [
                {
                    "evaluation_role": role,
                    "record_id": role,
                    "lq_path": "lq.png",
                    "gt_path": "gt.png",
                }
                for role in ("development", "sealed_internal_test")
            ]
            summary = verify_evaluation_pairs(rows, 16, root=root)
            self.assertEqual(summary["evaluation_records"], 2)
            self.assertEqual(
                summary["evaluation_minimum_side_by_role"],
                {"development": 18, "sealed_internal_test": 18},
            )
            with self.assertRaisesRegex(RuntimeError, "unaligned or undersized"):
                verify_evaluation_pairs(rows, 19, root=root)

    def test_exact_orphan_png_is_resume_safe_but_differing_png_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "partial.png"
            pixels = np.zeros((8, 8, 3), dtype=np.uint8)
            write_or_verify_png(path, pixels, label="test")
            write_or_verify_png(path, pixels.copy(), label="test")
            differing = pixels.copy()
            differing[0, 0, 0] = 1
            with self.assertRaisesRegex(RuntimeError, "differing orphan"):
                write_or_verify_png(path, differing, label="test")

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
            "generalist_architecture": "ResidualDiffusion",
            "architecture": "WaveMamba",
            "weather_architecture": "Restormer",
            "strict_checkpoint_load": True,
            "wrapper_forward": True,
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
