from __future__ import annotations

import unittest
from pathlib import Path

from lowlight_restoration.qwen_scaling import (
    load_scaling_config,
    validate_scaling_contract,
)
from smoke.qwen_lowlight_supervised_ddp_preflight import load_ddp_config
from smoke.validate_qwen_lowlight_supervised_scaling import validate_world_summary

ROOT = Path(__file__).resolve().parents[1]


class QwenScalingValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scaling, cls.scaling_sha, _ = load_scaling_config(
            ROOT / "configs/qwen_lowlight_supervised_scaling_1_2_4_8.json"
        )
        validate_scaling_contract(cls.scaling)
        cls.base, _, _ = load_ddp_config(
            ROOT / "configs/qwen_lowlight_supervised_overfit40_ddp8.json"
        )
        cls.preflight = {
            "base_preflight": {
                "model_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5"
            }
        }
        cls.preflight_sha = "a" * 64

    def make_summary(self, world_size: int) -> dict[str, object]:
        data = self.scaling["data"]
        training = self.scaling["training"]
        runtime = self.scaling["runtime"]
        model = self.base["model"]
        assert all(
            isinstance(section, dict)
            for section in (data, training, runtime, model)
        )
        accumulation = data["expected_gradient_accumulation_steps"]
        assert isinstance(accumulation, dict)
        measured_steps = int(training["measured_optimizer_steps"])
        measured_seconds = 10.0
        losses = [0.8 - index * 0.01 for index in range(measured_steps)]
        audit = {
            "distributed_all_ranks_passed": True,
            "rank_results": [
                {"rank": rank, "passed": True} for rank in range(world_size)
            ],
        }
        versions = {
            "python": runtime["required_python"],
            "pytorch": runtime["required_pytorch"],
            "transformers": runtime["required_transformers"],
        }
        return {
            "schema_version": 1,
            "status": "passed",
            "scope": "qwen_lowlight_supervised_distributed_scaling_benchmark",
            "scaling_config_sha256": self.scaling_sha,
            "base_overfit_config_sha256": self.scaling[
                "base_overfit_config_sha256"
            ],
            "preflight_sha256": self.preflight_sha,
            "git_revision": "b" * 40,
            "launcher_attempt_id": "20260819T120000Z-1234",
            "model_revision": self.preflight["base_preflight"]["model_revision"],
            "selection_bundle_sha256": self.scaling["selection_bundle_sha256"],
            "parameter_groups": model["expected_parameter_groups"],
            "world_size": world_size,
            "physical_cuda_device_set": ",".join(
                str(index) for index in range(world_size)
            ),
            "micro_batch_size_per_rank": data["micro_batch_size_per_rank"],
            "gradient_accumulation_steps": accumulation[str(world_size)],
            "global_batch_size": data["global_batch_size"],
            "warmup_optimizer_steps": training["warmup_optimizer_steps"],
            "measured_optimizer_steps": measured_steps,
            "measured_sample_presentations": measured_steps
            * int(data["global_batch_size"]),
            "initialization_seconds_this_process": 10.0,
            "local_preprocessing_seconds_this_process": 1.0,
            "warmup_seconds": 3.0,
            "measured_training_seconds": measured_seconds,
            "measured_seconds_per_step": measured_seconds / measured_steps,
            "measured_global_samples_per_second": measured_steps
            * int(data["global_batch_size"])
            / measured_seconds,
            "shell_to_measurement_start_seconds": 15.0,
            "shell_to_summary_seconds": 26.0,
            "measured_losses": losses,
            "final_measured_loss": losses[-1],
            "gradient_audit": audit,
            "frozen_parameter_audit": audit,
            "maximum_allocated_gpu_memory_gib_by_rank": [20.0] * world_size,
            "environment_by_rank": [
                {
                    "rank": rank,
                    "local_rank": rank,
                    "world_size": world_size,
                    "gpu": "NVIDIA H200",
                    "gpu_total_memory_gib": 139.0,
                    "cuda": "12.8",
                    "versions": versions,
                }
                for rank in range(world_size)
            ],
            "gate_checks": {
                "gradient_scope_passed": True,
                "frozen_parameters_unchanged": True,
                "distributed_model_equivalence_passed": True,
                "global_batch_semantics_passed": True,
                "finite_measured_losses": True,
            },
            "automatic_gate_pass": True,
            "synthetic_internal_only": True,
            "scientific_pilot_result": False,
            "test_split_used": False,
            "official_foundir_used": False,
            "base_preflight_large_hashes_checked": True,
        }

    def test_valid_world_summaries_pass(self) -> None:
        for world_size in (1, 2, 4, 8):
            with self.subTest(world_size=world_size):
                validate_world_summary(
                    self.make_summary(world_size),
                    world_size=world_size,
                    scaling=self.scaling,
                    scaling_sha=self.scaling_sha,
                    base=self.base,
                    preflight=self.preflight,
                    preflight_sha=self.preflight_sha,
                )

    def test_changed_device_mapping_fails_closed(self) -> None:
        summary = self.make_summary(4)
        summary["physical_cuda_device_set"] = "4,5,6,7"
        with self.assertRaises(RuntimeError):
            validate_world_summary(
                summary,
                world_size=4,
                scaling=self.scaling,
                scaling_sha=self.scaling_sha,
                base=self.base,
                preflight=self.preflight,
                preflight_sha=self.preflight_sha,
            )


if __name__ == "__main__":
    unittest.main()
