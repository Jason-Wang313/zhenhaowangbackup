from __future__ import annotations

import unittest
from pathlib import Path

from lowlight_restoration.qwen_scaling import (
    build_scaling_decision,
    expected_accumulation_steps,
    load_scaling_config,
    validate_scaling_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def summary(world_size: int, seconds_per_step: float, startup: float) -> dict[str, object]:
    global_batch = 40
    return {
        "status": "passed",
        "world_size": world_size,
        "gradient_accumulation_steps": 8 // world_size,
        "measured_seconds_per_step": seconds_per_step,
        "measured_global_samples_per_second": global_batch / seconds_per_step,
        "shell_to_measurement_start_seconds": startup,
        "warmup_seconds": seconds_per_step * 3,
        "measured_losses": [0.8 - index * 0.01 for index in range(10)],
        "maximum_allocated_gpu_memory_gib_by_rank": [20.0] * world_size,
    }


class QwenScalingTests(unittest.TestCase):
    def test_checked_in_scaling_contract_is_valid(self) -> None:
        config, digest, _ = load_scaling_config(
            ROOT / "configs/qwen_lowlight_supervised_scaling_1_2_4_8.json"
        )
        validate_scaling_contract(config)
        self.assertEqual(len(digest), 64)

    def test_accumulation_preserves_global_source_count(self) -> None:
        self.assertEqual(
            [
                expected_accumulation_steps(source_count=8, world_size=world)
                for world in (1, 2, 4, 8)
            ],
            [8, 4, 2, 1],
        )
        with self.assertRaises(ValueError):
            expected_accumulation_steps(source_count=8, world_size=3)

    def test_projection_selects_fastest_end_to_end_world_size(self) -> None:
        decision = build_scaling_decision(
            [
                summary(1, 16.0, 80.0),
                summary(2, 8.5, 60.0),
                summary(4, 4.8, 50.0),
                summary(8, 3.0, 40.0),
            ],
            projection_optimizer_steps=1000,
            minimum_multi_gpu_speedup=1.05,
            maximum_relative_loss_deviation=0.05,
        )
        self.assertEqual(decision["selected_world_size"], 8)
        self.assertTrue(decision["multi_gpu_throughput_benefit_passed"])
        self.assertTrue(decision["multi_gpu_projected_wall_time_benefit_passed"])
        self.assertTrue(decision["cross_world_loss_equivalence_passed"])
        self.assertAlmostEqual(
            decision["rows"][-1]["throughput_speedup_over_one_gpu"],
            16.0 / 3.0,
        )

    def test_missing_or_failed_world_size_is_rejected(self) -> None:
        summaries = [summary(1, 16.0, 20.0), summary(2, 8.0, 30.0)]
        with self.assertRaises(ValueError):
            build_scaling_decision(
                summaries,
                projection_optimizer_steps=1000,
                minimum_multi_gpu_speedup=1.05,
                maximum_relative_loss_deviation=0.05,
            )
        summaries = [
            summary(1, 16.0, 20.0),
            summary(2, 8.0, 30.0),
            summary(4, 4.0, 40.0),
            summary(8, 3.0, 60.0),
        ]
        summaries[-1]["status"] = "failed"
        with self.assertRaises(ValueError):
            build_scaling_decision(
                summaries,
                projection_optimizer_steps=1000,
                minimum_multi_gpu_speedup=1.05,
                maximum_relative_loss_deviation=0.05,
            )

    def test_cross_world_loss_divergence_fails_equivalence(self) -> None:
        summaries = [
            summary(1, 16.0, 60.0),
            summary(2, 8.0, 40.0),
            summary(4, 4.0, 30.0),
            summary(8, 3.0, 25.0),
        ]
        summaries[-1]["measured_losses"] = [1.2] * 10
        decision = build_scaling_decision(
            summaries,
            projection_optimizer_steps=1000,
            minimum_multi_gpu_speedup=1.05,
            maximum_relative_loss_deviation=0.05,
        )
        self.assertFalse(decision["cross_world_loss_equivalence_passed"])


if __name__ == "__main__":
    unittest.main()
