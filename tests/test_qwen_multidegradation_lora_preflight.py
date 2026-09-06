import json
import unittest
from copy import deepcopy
from pathlib import Path

from smoke.qwen_multidegradation_lora_preflight import (
    ROOT,
    accumulation_steps_for_world,
    load_config,
    validate_contract,
    validate_model_inventory,
    validate_meta_architecture,
    validate_software_runtime,
)


class QwenMultiDegradationLoraPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "configs/qwen_multidegradation_lora_v1.json"
        cls.config, cls.sha = load_config(cls.path)

    def test_static_contract_and_inventory(self):
        validate_contract(self.config)
        self.assertEqual(validate_software_runtime(self.config)["transformers"], "5.14.1")
        result = validate_model_inventory(self.config, full_hashes=False)
        self.assertEqual(result["revision"], "cc594898137f460bfe9f0759e9844b3ce807cfb5")

    def test_rejects_device_or_curriculum_drift(self):
        changed = deepcopy(self.config)
        changed["runtime"]["supported_visible_devices"] = {"0,1": 2}
        with self.assertRaises(ValueError):
            validate_contract(changed)
        changed = deepcopy(self.config)
        changed["data"]["curriculum"][2]["maximum_composition_size"] = 2
        with self.assertRaises(ValueError):
            validate_contract(changed)

    def test_two_and_four_gpu_contract_preserve_global_batch(self):
        training = self.config["training"]
        for world, expected_accumulation in ((2, 2), (4, 1)):
            accumulation = accumulation_steps_for_world(self.config, world)
            self.assertEqual(accumulation, expected_accumulation)
            self.assertEqual(
                training["micro_batch_size_per_rank"] * world * accumulation,
                training["global_batch_size"],
            )

    def test_installed_qwen_meta_architecture_matches_frozen_scope(self):
        result = validate_meta_architecture(self.config)
        self.assertEqual(result["matched_lora_module_count"], 160)
        self.assertEqual(result["lora_trainable_parameters"], 11151360)
        self.assertEqual(result["merger_trainable_parameters"], 44574464)
    def test_config_is_machine_serializable(self):
        json.dumps(self.config, sort_keys=True)
        self.assertEqual(len(self.sha), 64)

    def test_validation_correction_is_hash_pinned_and_test_blind(self):
        path = ROOT / "configs/qwen_multidegradation_lora_correction1_v1.json"
        correction, correction_sha = load_config(path)
        validate_contract(correction)
        self.assertEqual(len(correction_sha), 64)
        self.assertEqual(correction["model"]["variants"], ["merger_trainable"])
        self.assertEqual(
            correction["data"]["curriculum"],
            [
                {
                    "name": "D_full_mixture_consolidation",
                    "maximum_composition_size": 3,
                    "records": 9600,
                    "epochs": 2,
                }
            ],
        )
        self.assertFalse(
            (
                ROOT
                / correction["initialization"]["base_fixed_test_result"]
            ).exists()
        )

    def test_semantic_loss_correction_is_hash_pinned_and_test_blind(self):
        path = ROOT / "configs/qwen_multidegradation_lora_correction2_v1.json"
        correction, correction_sha = load_config(path)
        validate_contract(correction)
        self.assertEqual(len(correction_sha), 64)
        self.assertEqual(
            correction["data"]["curriculum"],
            [
                {
                    "name": "E_semantic_decision_consolidation",
                    "maximum_composition_size": 3,
                    "records": 9600,
                    "epochs": 1,
                }
            ],
        )
        self.assertEqual(
            correction["training"]["semantic_token_loss"]["semantic_weight"],
            4.0,
        )
        self.assertFalse(
            (
                ROOT
                / correction["initialization"]["base_fixed_test_result"]
            ).exists()
        )

    def test_crossmodal_decoder_correction_is_hash_pinned_and_test_blind(self):
        path = ROOT / "configs/qwen_multidegradation_lora_correction3_v1.json"
        correction, correction_sha = load_config(path)
        validate_contract(correction)
        self.assertEqual(len(correction_sha), 64)
        self.assertEqual(correction["model"]["expected_lora_modules"], 272)
        self.assertEqual(correction["model"]["expected_language_lora_modules"], 112)
        self.assertEqual(
            correction["initialization"]["allow_missing_adapter_prefixes"],
            ["model.language_model."],
        )
        self.assertEqual(correction["training"]["total_optimizer_steps"], 300)
        self.assertFalse(
            (
                ROOT
                / correction["initialization"]["base_fixed_test_result"]
            ).exists()
        )

    def test_fixed_slot_correction_is_hash_pinned_and_test_blind(self):
        path = ROOT / "configs/qwen_multidegradation_lora_correction4_v1.json"
        correction, correction_sha = load_config(path)
        validate_contract(correction)
        self.assertEqual(len(correction_sha), 64)
        self.assertEqual(
            correction["prompt"]["target_schema"],
            "fixed_five_slot_l1_absent_l2_l5_v1",
        )
        self.assertEqual(
            correction["training"]["semantic_token_loss"]["semantic_token_ids"],
            [16, 17, 18, 19, 20],
        )
        self.assertEqual(correction["training"]["total_optimizer_steps"], 600)
        self.assertFalse(
            (
                ROOT
                / correction["initialization"]["base_fixed_test_result"]
            ).exists()
        )

    def test_advisor_per_type_rank32_retraining_contract(self):
        path = ROOT / "configs/qwen_multidegradation_per_type_rank32_v1.json"
        config, config_sha = load_config(path)
        validate_contract(config)
        self.assertEqual(len(config_sha), 64)
        self.assertEqual(config["model"]["lora_rank"], 32)
        self.assertEqual(config["model"]["language_lora_rank"], 32)
        self.assertEqual(config["training"]["micro_batch_size_per_rank"], 25)
        self.assertEqual(config["training"]["global_batch_size"], 100)
        self.assertEqual(config["training"]["total_optimizer_steps"], 1440)
        self.assertEqual(
            config["data"]["expected_dialogue_split_records"],
            {"train": 48000, "validation": 6000, "test": 6000},
        )
        result = validate_meta_architecture(config)
        self.assertEqual(result["matched_lora_module_count"], 272)
        self.assertEqual(result["visual_lora_trainable_parameters"], 22302720)
        self.assertEqual(result["language_lora_trainable_parameters"], 20185088)
        self.assertEqual(result["trainable_parameters"], 87062272)

    def test_advisor_rank32_continuation_preserves_two_and_eight_gpu_semantics(self):
        path = (
            ROOT
            / "configs/qwen_multidegradation_per_type_rank32_continue_v2.json"
        )
        config, config_sha = load_config(path)
        validate_contract(config)
        self.assertEqual(len(config_sha), 64)
        self.assertEqual(config["distributed"]["supported_world_sizes"], [2, 8])
        self.assertEqual(config["training"]["global_batch_size"], 200)
        for world, accumulation in ((2, 4), (8, 1)):
            self.assertEqual(accumulation_steps_for_world(config, world), accumulation)
            self.assertEqual(
                config["training"]["micro_batch_size_per_rank"]
                * world
                * accumulation,
                200,
            )
        self.assertEqual(config["training"]["initial_optimizer_steps"], 1440)
        self.assertEqual(config["training"]["continuation_optimizer_steps"], 1200)
        self.assertEqual(config["training"]["total_optimizer_steps"], 2640)
        self.assertEqual(
            config["gate"]["minimum_triple_degradation_set_exact_match_rate"],
            0.8,
        )
        self.assertFalse(
            (ROOT / config["initialization"]["base_fixed_test_result"]).exists()
        )

    def test_visibility_v2_rank32_contract_is_training_calibrated_and_test_blind(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_visibility_v2_rank32.json"
        )
        validate_contract(config)
        self.assertEqual(config["distributed"]["supported_world_sizes"], [2, 8])
        self.assertEqual(
            config["training"]["semantic_token_loss"]["semantic_token_ids"],
            [16, 17, 18, 19, 20],
        )
        self.assertEqual(
            config["decision_calibration"]["fit_split"],
            "reserved_training_sources_only",
        )
        self.assertFalse(
            config["decision_calibration"]["validation_or_test_used_for_fit"]
        )
        fixed_test = (
            ROOT
            / config["runtime"]["output_root"]
            / "merger_trainable/fixed_test_result.json"
        )
        self.assertFalse(fixed_test.exists())

    def test_v8_physical_continuation_is_hash_bound_and_test_blind(self):
        config, config_sha = load_config(
            ROOT
            / "configs/qwen_multidegradation_100k_rank64_per_type_physical_v8.json"
        )
        validate_contract(config)
        self.assertEqual(len(config_sha), 64)
        self.assertEqual(config["model"]["lora_rank"], 64)
        self.assertEqual(config["model"]["lora_alpha"], 64)
        self.assertIsNone(config["initialization"]["optimizer"])
        self.assertEqual(config["training"]["total_optimizer_steps"], 2500)
        self.assertEqual(len(config["data"]["curriculum"]), 5)
        self.assertEqual(
            [stage["source_start"] for stage in config["data"]["curriculum"]],
            [0, 20000, 40000, 60000, 80000],
        )
        self.assertEqual(
            config["training"]["presence_auxiliary_loss"],
            {
                "enabled": True,
                "weight": 1.0,
                "level_token_ids": [16, 17, 18, 19, 20],
            },
        )
        self.assertFalse(
            (
                ROOT
                / config["runtime"]["output_root"]
                / "merger_trainable/fixed_test_result.json"
            ).exists()
        )

    def test_v8_rejects_presence_loss_or_initialization_drift(self):
        config, _ = load_config(
            ROOT
            / "configs/qwen_multidegradation_100k_rank64_per_type_physical_v8.json"
        )
        changed = deepcopy(config)
        changed["training"]["presence_auxiliary_loss"]["weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "presence auxiliary"):
            validate_contract(changed)
        changed = deepcopy(config)
        changed["initialization"]["optimizer"] = "optimizer.pt"
        with self.assertRaisesRegex(ValueError, "fresh optimizer"):
            validate_contract(changed)


if __name__ == "__main__":
    unittest.main()
