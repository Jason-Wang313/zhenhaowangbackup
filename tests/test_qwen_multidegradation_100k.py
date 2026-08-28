import json
import unittest
from pathlib import Path

from lowlight_restoration.multidegradation_data import (
    BLUR_PARAMETERS_EXTREME_V6,
    ROOT,
    sha256,
)


class QwenMultidegradation100kTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_contract_path = (
            ROOT / "configs/qwen_multidegradation_100k_data_v1.json"
        )
        cls.data_contract = json.loads(cls.data_contract_path.read_text())
        cls.metadata = json.loads(
            (ROOT / "manifests/qwen_multidegradation_100k_blur_v6.json").read_text()
        )
        cls.joint = json.loads(
            (
                ROOT
                / "configs/qwen_multidegradation_100k_rank64_joint_v1.json"
            ).read_text()
        )
        cls.per_type = json.loads(
            (
                ROOT
                / "configs/qwen_multidegradation_100k_rank64_per_type_v1.json"
            ).read_text()
        )

    def test_manifest_is_exactly_100k_train_plus_disjoint_10k_splits(self):
        validation = self.metadata["validation"]
        self.assertEqual(
            validation["split_records"],
            {"train": 100000, "validation": 10000, "test": 10000},
        )
        self.assertEqual(validation["unique_sources_used"], 2000)
        self.assertTrue(validation["source_disjoint"])
        manifest = ROOT / self.metadata["manifest"]
        self.assertEqual(sha256(manifest), self.metadata["manifest_sha256"])

    def test_blur_v6_is_bound_to_the_human_inspection_artifacts(self):
        self.assertEqual(
            BLUR_PARAMETERS_EXTREME_V6,
            {2: (22, 18.0), 3: (30, 24.0), 4: (90, 72.0), 5: (200, 160.0)},
        )
        blur = self.data_contract["blur"]
        self.assertEqual(
            sha256(ROOT / blur["preview"]), blur["preview_sha256"]
        )
        self.assertEqual(
            sha256(ROOT / blur["preview_report"]),
            blur["preview_report_sha256"],
        )
        self.assertEqual(
            self.metadata["data_contract_sha256"], sha256(self.data_contract_path)
        )

    def test_two_methods_change_the_question_format_only(self):
        self.assertEqual(self.joint["model"], self.per_type["model"])
        self.assertEqual(self.joint["training"], self.per_type["training"] | {
            "total_optimizer_steps": self.joint["training"]["total_optimizer_steps"]
        })
        self.assertEqual(
            self.joint["prompt"]["target_schema"],
            "fixed_five_slot_l1_absent_l2_l5_v1",
        )
        self.assertEqual(
            self.per_type["prompt"]["target_schema"],
            "per_type_dialogue_l1_absent_l2_l5_v1",
        )
        self.assertEqual(
            [stage["source_records"] for stage in self.joint["data"]["curriculum"]],
            [40000, 70000, 100000],
        )
        self.assertEqual(
            [stage["source_records"] for stage in self.per_type["data"]["curriculum"]],
            [40000, 70000, 100000],
        )
        for joint_stage, per_type_stage in zip(
            self.joint["data"]["curriculum"],
            self.per_type["data"]["curriculum"],
            strict=True,
        ):
            self.assertEqual(per_type_stage["records"], joint_stage["records"] * 5)

    def test_rank_alpha_mse_and_hardware_contracts_are_locked(self):
        for config in (self.joint, self.per_type):
            model = config["model"]
            self.assertEqual(
                (
                    model["lora_rank"],
                    model["lora_alpha"],
                    model["language_lora_rank"],
                    model["language_lora_alpha"],
                ),
                (64, 64, 64, 64),
            )
            self.assertEqual(
                config["gate"]["maximum_severity_typed_mse_with_missing_penalty"],
                1.0,
            )
            self.assertEqual(
                config["distributed"]["supported_world_sizes"], [1, 2, 4, 8]
            )
            for world in (1, 2, 4, 8):
                micro = config["training"][
                    "micro_batch_size_per_rank_by_world_size"
                ][str(world)]
                accumulation = config["training"][
                    "gradient_accumulation_steps_by_world_size"
                ][str(world)]
                self.assertEqual(world * micro * accumulation, 200)


if __name__ == "__main__":
    unittest.main()
