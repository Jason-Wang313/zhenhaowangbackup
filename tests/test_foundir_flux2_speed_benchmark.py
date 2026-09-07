from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from smoke.preflight_foundir_flux2_training_geometry import geometry_summary
from smoke.select_foundir_flux2_speed_benchmark import relative_loss_delta, select
from smoke.train_foundir_five_type_flux2_full_ddp import (
    TRAINING_UNDERSIZED_PAIR_POLICY,
    flow_random_inputs,
    prepare_aligned_pair,
    resized_pair_dimensions,
    synchronized_loader_batches,
)


ROOT = Path(__file__).resolve().parents[1]


class FoundIRFlux2SpeedBenchmarkTests(unittest.TestCase):
    def test_training_pair_resize_is_aligned_and_normal_pairs_are_untouched(self) -> None:
        normal_lq = np.zeros((512, 700, 3), dtype=np.uint8)
        normal_gt = np.ones((512, 700, 3), dtype=np.uint8)
        observed_lq, observed_gt = prepare_aligned_pair(normal_lq, normal_gt, 512, "normal")
        self.assertIs(observed_lq, normal_lq)
        self.assertIs(observed_gt, normal_gt)

        narrow_lq = np.full((494, 1920, 3), 32, dtype=np.uint8)
        narrow_gt = np.full((494, 1920, 3), 96, dtype=np.uint8)
        self.assertEqual(resized_pair_dimensions(1920, 494, 512), (1990, 512))
        observed_lq, observed_gt = prepare_aligned_pair(narrow_lq, narrow_gt, 512, "narrow")
        self.assertEqual(observed_lq.shape, (512, 1990, 3))
        self.assertEqual(observed_gt.shape, (512, 1990, 3))
        self.assertTrue(np.all(observed_lq == 32))
        self.assertTrue(np.all(observed_gt == 96))

    def test_unaligned_pair_fails_before_crop(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unaligned FoundIR pair"):
            prepare_aligned_pair(
                np.zeros((512, 512, 3), dtype=np.uint8),
                np.zeros((511, 512, 3), dtype=np.uint8),
                512,
                "bad",
            )

    def test_geometry_preflight_reports_all_undersized_rows(self) -> None:
        rows = [
            {
                "record_id": "normal",
                "category": "01Blur",
                "split": "train",
                "width": "1920",
                "height": "1080",
            },
            {
                "record_id": "narrow",
                "category": "10Rain",
                "split": "internal_test",
                "width": "674",
                "height": "266",
            },
        ]
        result = geometry_summary(rows, 512)
        self.assertEqual(result["undersized_records"], 1)
        self.assertEqual(result["undersized_categories"], {"10Rain": 1})
        self.assertEqual(result["undersized_splits"], {"internal_test": 1})
        self.assertEqual(result["training_undersized_pair_policy"], TRAINING_UNDERSIZED_PAIR_POLICY)

    def test_randomness_is_independent_of_microbatch_grouping(self) -> None:
        all_noise, all_timesteps = flow_random_inputs(
            (4, 2, 3), [0, 8, 16, 24], seed=17, device=torch.device("cpu")
        )
        first_noise, first_timesteps = flow_random_inputs(
            (2, 2, 3), [0, 8], seed=17, device=torch.device("cpu")
        )
        second_noise, second_timesteps = flow_random_inputs(
            (2, 2, 3), [16, 24], seed=17, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(all_noise, torch.cat([first_noise, second_noise])))
        self.assertTrue(torch.equal(all_timesteps, torch.cat([first_timesteps, second_timesteps])))

    def test_loader_failure_is_collectively_reported_before_forward(self) -> None:
        def fail():
            yield {"ok": True}
            raise RuntimeError("decode failed")

        def gather(errors, error):
            errors[0] = error

        with patch("torch.distributed.all_reduce"), patch(
            "torch.distributed.get_world_size", return_value=1
        ), patch("torch.distributed.all_gather_object", side_effect=gather), self.assertRaisesRegex(
            RuntimeError, "synchronized DataLoader failure.*decode failed"
        ):
            synchronized_loader_batches(iter(fail()), 2, device=torch.device("cpu"))

    def test_selector_requires_loss_agreement_and_material_speedup(self) -> None:
        common = {
            "status": "speed_benchmark_complete",
            "config_sha256": "a" * 64,
            "git_revision": "b" * 40,
            "gpu_family": "H200",
            "world_size": 8,
            "global_batch_size": 32,
            "optimizer_steps": 20,
            "warmup_steps_excluded": 5,
            "mean_losses": [1.0, 0.9],
            "peak_memory_allocated_bytes_per_rank": [10] * 8,
            "total_memory_bytes_per_rank": [100] * 8,
            "protected_foundir_test_accessed": False,
            "checkpoint_written": False,
            "per_presentation_randomness_independent_of_microbatch_grouping": True,
        }
        baseline = {
            **common,
            "benchmark_name": "baseline",
            "micro_batch_size_per_rank": 2,
            "gradient_accumulation_steps": 2,
            "data_loader_workers_per_rank": 4,
            "mean_seconds_per_step": 4.0,
            "global_samples_per_second": 8.0,
        }
        fast = {
            **common,
            "benchmark_name": "fast",
            "micro_batch_size_per_rank": 4,
            "gradient_accumulation_steps": 1,
            "data_loader_workers_per_rank": 8,
            "mean_seconds_per_step": 2.0,
            "global_samples_per_second": 16.0,
            "mean_losses": [1.01, 0.91],
        }
        divergent = {
            **fast,
            "benchmark_name": "divergent",
            "mean_seconds_per_step": 1.0,
            "global_samples_per_second": 32.0,
            "mean_losses": [2.0, 2.0],
        }
        result = select([baseline, fast, divergent])
        self.assertEqual(result["selected"], "fast")
        self.assertAlmostEqual(result["selected_speedup_vs_baseline"], 2.0)
        self.assertEqual(result["gpu_family"], "H200")
        self.assertEqual(result["config_sha256"], "a" * 64)
        self.assertGreater(relative_loss_delta([1.0], [2.0]), 0.05)

    def test_benchmark_launchers_never_access_official_test(self) -> None:
        source = "\n".join(
            (ROOT / relative).read_text()
            for relative in (
                "smoke/run_foundir_flux2_speed_benchmark_8gpu.sh",
                "smoke/launch_foundir_flux2_speed_benchmark_8gpu.sh",
            )
        )
        self.assertNotIn("MillionIRData/Test", source)
        self.assertIn("--benchmark-steps 20", source)
        self.assertIn("candidate_${candidate_micro}x${candidate_accum}_w8", source)
        main_runner = (ROOT / "smoke/run_foundir_five_type_flux2_main_v3_8gpu.sh").read_text()
        self.assertIn("speed_benchmark_selection_${training_family}.json", main_runner)
        self.assertIn("--approved-speed-selection", main_runner)


if __name__ == "__main__":
    unittest.main()
