import json
import unittest
from collections import Counter
from pathlib import Path

from lowlight_restoration.multidegradation_data import (
    BLUR_PARAMETERS_EXTREME_V6,
    BLUR_PARAMETERS_MILD_V7,
    NOISE_SIGMA_RANGES_MILD_V7,
    ROOT,
    sha256,
    _balanced_levels,
)
from smoke.build_qwen_multidegradation_100k import (
    V8_STATUS,
    load_contract,
    v8_non_target_nuisance,
    v8_operator_parameters,
)
from smoke.generate_qwen_multidegradation_100k import source_grouped_chunks


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
        cls.v7_data_contract_path = (
            ROOT / "configs/qwen_multidegradation_100k_mild_v7_data.json"
        )
        cls.v7_data_contract = json.loads(cls.v7_data_contract_path.read_text())
        cls.v7_metadata = json.loads(
            (
                ROOT / "manifests/qwen_multidegradation_100k_mild_v7.json"
            ).read_text()
        )
        cls.joint_v2 = json.loads(
            (
                ROOT
                / "configs/qwen_multidegradation_100k_rank64_joint_v2.json"
            ).read_text()
        )
        cls.per_type_v2 = json.loads(
            (
                ROOT
                / "configs/qwen_multidegradation_100k_rank64_per_type_v2.json"
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

    def test_source_grouped_generation_preserves_every_record(self):
        records = [
            {"record_id": "a1", "source_path": "a.png", "source_sha256": "1"},
            {"record_id": "b1", "source_path": "b.png", "source_sha256": "2"},
            {"record_id": "a2", "source_path": "a.png", "source_sha256": "1"},
            {"record_id": "a3", "source_path": "a.png", "source_sha256": "1"},
        ]
        grouped = source_grouped_chunks(records, 2)
        self.assertEqual(
            sorted(row["record_id"] for chunk in grouped for row in chunk),
            ["a1", "a2", "a3", "b1"],
        )
        self.assertTrue(
            all(len({row["source_path"] for row in chunk}) == 1 for chunk in grouped)
        )
        self.assertTrue(all(len(chunk) <= 2 for chunk in grouped))

    def test_mild_v7_strengths_are_the_advisor_directed_non_overwriting_corpus(self):
        self.assertEqual(
            BLUR_PARAMETERS_MILD_V7,
            {2: (1, 1.0), 3: (2, 1.6), 4: (4, 3.2), 5: (8, 6.4)},
        )
        self.assertEqual(
            NOISE_SIGMA_RANGES_MILD_V7,
            {2: (1.0, 3.0), 3: (3.0, 6.0), 4: (6.0, 12.0), 5: (12.0, 25.0)},
        )
        self.assertEqual(
            self.v7_data_contract["status"],
            "advisor_directed_mild_blur_noise_v7_100k_contract",
        )
        self.assertEqual(
            self.v7_metadata["data_contract_sha256"],
            sha256(self.v7_data_contract_path),
        )
        self.assertEqual(
            self.v7_metadata["validation"]["split_records"],
            {"train": 100000, "validation": 10000, "test": 10000},
        )

    def test_v2_methods_are_bound_only_to_mild_v7_and_keep_scientific_settings(self):
        pairs = (
            (self.joint, self.joint_v2, "joint"),
            (self.per_type, self.per_type_v2, "per_type"),
        )
        for old, new, method in pairs:
            for section in (
                "contract",
                "model",
                "processor",
                "prompt",
                "training",
                "overfit",
                "distributed",
                "generation",
                "gate",
            ):
                self.assertEqual(old[section], new[section], (method, section))
            self.assertNotIn("initialization", new)
            self.assertEqual(
                new["data_contract"],
                {
                    "path": "configs/qwen_multidegradation_100k_mild_v7_data.json",
                    "sha256": sha256(self.v7_data_contract_path),
                },
            )
            self.assertEqual(
                new["data"]["manifest"],
                "manifests/qwen_multidegradation_100k_mild_v7.jsonl",
            )
            self.assertEqual(
                new["data"]["manifest_sha256"],
                self.v7_metadata["manifest_sha256"],
            )
            self.assertEqual(
                new["data"]["generation_result"],
                "reports/qwen_multidegradation_100k_mild_v7_generation.json",
            )
            self.assertEqual(
                new["data"]["generation_evidence_root"],
                "data/qwen_multidegradation_100k_mild_v7/.generation",
            )
            self.assertEqual(new["data"]["curriculum"], old["data"]["curriculum"])
            self.assertEqual(
                new["runtime"]["output_root"],
                f"checkpoints/qwen_multidegradation_100k_rank64_{method}_v2",
            )
            self.assertEqual(new["runtime"]["minimum_gpu_memory_gib"], 78)

    def test_v2_joint_and_per_type_remain_a_fair_prompt_only_comparison(self):
        self.assertEqual(self.joint_v2["model"], self.per_type_v2["model"])
        self.assertEqual(self.joint_v2["training"], self.per_type_v2["training"] | {
            "total_optimizer_steps": self.joint_v2["training"]["total_optimizer_steps"]
        })
        self.assertEqual(
            self.joint_v2["data"]["expected_split_records"],
            self.per_type_v2["data"]["expected_split_records"],
        )
        for joint_stage, per_type_stage in zip(
            self.joint_v2["data"]["curriculum"],
            self.per_type_v2["data"]["curriculum"],
            strict=True,
        ):
            self.assertEqual(per_type_stage["source_records"], joint_stage["source_records"])
            self.assertEqual(per_type_stage["records"], joint_stage["records"] * 5)

    def test_v8_family_level_assignment_is_exactly_balanced(self):
        for name in ("blur", "noise"):
            counts = Counter()
            for index in range(12):
                level = _balanced_levels((name,), index)[name]
                parameters = v8_operator_parameters(
                    name,
                    level,
                    index=index,
                    combination_index=0,
                    seed=123 + index,
                    source={},
                )
                counts[(parameters["family"], level)] += 1
            self.assertEqual(len(counts), 12)
            self.assertEqual(set(counts.values()), {1})

    def test_v8_contract_and_hard_negative_clean_semantics(self):
        path = ROOT / "configs/qwen_multidegradation_100k_physical_v8_data.json"
        contract, digest = load_contract(path)
        self.assertEqual(contract["status"], V8_STATUS)
        self.assertEqual(len(digest), 64)
        self.assertEqual(
            v8_non_target_nuisance(
                index=2,
                combination_index=0,
                size=0,
                seed=9,
                degradations=(),
            )["family"],
            "none",
        )
        observed = {
            v8_non_target_nuisance(
                index=index * 12,
                combination_index=0,
                size=1,
                seed=100 + index,
                degradations=("rain",),
            )["family"]
            for index in range(4)
        }
        self.assertEqual(observed, {"none", "jpeg", "resample"})


if __name__ == "__main__":
    unittest.main()
