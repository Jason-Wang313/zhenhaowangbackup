import json
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from smoke.train_qwen_multidegradation_lora_ddp import (
    calibrated_per_type_outputs,
    create_optimizer,
    decoder_presence_auxiliary_loss,
    gate_checks,
    left_padded_generation,
    learning_rate_scale,
    rank_microbatch,
    resume_stage,
    semantic_assistant_token_loss,
    stage_records,
    training_loss,
    validate_completed_stage_checkpoints,
)
from lowlight_restoration.qwen_multidegradation import DEGRADATIONS
from smoke.train_qwen_multidegradation_overfit_ddp import expand_per_type_records
from smoke.qwen_multidegradation_lora_preflight import ROOT, load_config
from lowlight_restoration.multidegradation_data import read_jsonl
from smoke.train_qwen_multidegradation_overfit_ddp import select_overfit_records


class TinyScope(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2))
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.merger = nn.Linear(2, 2)
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)


class TinyCrossmodalScope(TinyScope):
    def __init__(self):
        super().__init__()
        self.model.language_model = nn.Module()
        self.model.language_model.lora_A = nn.Parameter(torch.ones(2))


class QwenMultiDegradationTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, _ = load_config(ROOT / "configs/qwen_multidegradation_lora_v1.json")
        cls.records = read_jsonl(ROOT / "manifests/qwen_multidegradation_pilot_v1.jsonl")

    def test_learning_rate_schedule(self):
        self.assertAlmostEqual(learning_rate_scale(1, 100, 10, 0.1), 0.1)
        self.assertAlmostEqual(learning_rate_scale(10, 100, 10, 0.1), 1.0)
        self.assertAlmostEqual(learning_rate_scale(100, 100, 10, 0.1), 0.1)

    def test_generation_padding_is_left_and_always_restored(self):
        processor = SimpleNamespace(
            tokenizer=SimpleNamespace(padding_side="right")
        )
        with left_padded_generation(processor):
            self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(processor.tokenizer.padding_side, "right")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with left_padded_generation(processor):
                self.assertEqual(processor.tokenizer.padding_side, "left")
                raise RuntimeError("generation failed")
        self.assertEqual(processor.tokenizer.padding_side, "right")

    def test_progressive_stages_retain_prior_compositions(self):
        stages = self.config["data"]["curriculum"]
        prior = set()
        for stage in stages:
            selected = stage_records(self.records, stage, self.config["training"]["seed"])
            current = {record["record_id"] for record in selected}
            self.assertTrue(prior.issubset(current))
            self.assertEqual(len(selected), stage["records"])
            prior = current

    def test_correction_replays_the_complete_mixture_twice(self):
        correction, _ = load_config(
            ROOT / "configs/qwen_multidegradation_lora_correction1_v1.json"
        )
        stage = correction["data"]["curriculum"][0]
        first = stage_records(
            self.records, stage, correction["training"]["seed"], epoch=0
        )
        second = stage_records(
            self.records, stage, correction["training"]["seed"], epoch=1
        )
        self.assertEqual(len(first), 9600)
        self.assertEqual(
            {record["record_id"] for record in first},
            {record["record_id"] for record in second},
        )
        self.assertNotEqual(
            [record["record_id"] for record in first[:32]],
            [record["record_id"] for record in second[:32]],
        )
        self.assertEqual(correction["training"]["total_optimizer_steps"], 600)

    def test_semantic_correction_replays_one_complete_mixture(self):
        correction, _ = load_config(
            ROOT / "configs/qwen_multidegradation_lora_correction2_v1.json"
        )
        stage = correction["data"]["curriculum"][0]
        selected = stage_records(
            self.records, stage, correction["training"]["seed"], epoch=0
        )
        self.assertEqual(len(selected), 9600)
        self.assertEqual(len({record["record_id"] for record in selected}), 9600)
        self.assertEqual(correction["training"]["total_optimizer_steps"], 300)

    def test_fixed_slot_correction_replays_complete_mixture_twice(self):
        correction, _ = load_config(
            ROOT / "configs/qwen_multidegradation_lora_correction4_v1.json"
        )
        stage = correction["data"]["curriculum"][0]
        first = stage_records(
            self.records, stage, correction["training"]["seed"], epoch=0
        )
        second = stage_records(
            self.records, stage, correction["training"]["seed"], epoch=1
        )
        self.assertEqual(len(first), len(second), 9600)
        self.assertEqual(
            {record["record_id"] for record in first},
            {record["record_id"] for record in second},
        )
        self.assertNotEqual(first[:32], second[:32])
        self.assertEqual(correction["training"]["total_optimizer_steps"], 600)

    def test_per_type_retraining_covers_every_image_type_and_balances_batches(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_per_type_rank32_v1.json"
        )
        stage = config["data"]["curriculum"][0]
        dialogues = stage_records(
            self.records,
            stage,
            config["training"]["seed"],
            target_schema=config["prompt"]["target_schema"],
            global_batch_size=config["training"]["global_batch_size"],
        )
        self.assertEqual(len(dialogues), 48000)
        self.assertEqual(
            len(
                {
                    (record["record_id"], record["query_degradation"])
                    for record in dialogues
                }
            ),
            48000,
        )
        for offset in range(0, len(dialogues), 100):
            batch = dialogues[offset : offset + 100]
            self.assertEqual(
                {
                    name: sum(record["query_degradation"] == name for record in batch)
                    for name in DEGRADATIONS
                },
                {name: 20 for name in DEGRADATIONS},
            )

    def test_continuation_global_batch_200_balances_all_five_questions(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_per_type_rank32_continue_v2.json"
        )
        dialogues = stage_records(
            self.records,
            config["data"]["curriculum"][0],
            config["training"]["seed"],
            target_schema=config["prompt"]["target_schema"],
            global_batch_size=config["training"]["global_batch_size"],
        )
        self.assertEqual(len(dialogues), 48000)
        for offset in range(0, len(dialogues), 200):
            batch = dialogues[offset : offset + 200]
            self.assertEqual(
                {
                    name: sum(record["query_degradation"] == name for record in batch)
                    for name in DEGRADATIONS
                },
                {name: 40 for name in DEGRADATIONS},
            )

    def test_visibility_v2_excludes_calibration_sources_and_balances_batches(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_visibility_v2_rank32.json"
        )
        records = read_jsonl(ROOT / config["data"]["manifest"])
        partition = json.loads(
            (ROOT / config["data"]["calibration_partition"]["path"]).read_text()
        )
        dialogues = stage_records(
            records,
            config["data"]["curriculum"][0],
            config["training"]["seed"],
            target_schema=config["prompt"]["target_schema"],
            global_batch_size=config["training"]["global_batch_size"],
            excluded_source_ids=partition["calibration_source_ids"],
        )
        self.assertEqual(len(dialogues), 42000)
        self.assertFalse(
            {record["source_id"] for record in dialogues}
            & set(partition["calibration_source_ids"])
        )
        for offset in range(0, len(dialogues), 200):
            batch = dialogues[offset : offset + 200]
            self.assertEqual(
                {
                    name: sum(record["query_degradation"] == name for record in batch)
                    for name in DEGRADATIONS
                },
                {name: 40 for name in DEGRADATIONS},
            )

    def test_calibrated_candidate_outputs_remain_decoder_dialogue_json(self):
        scores = [
            {
                degradation: {
                    "candidate_scores": [0.0, 1.0, 2.0, 3.0, 4.0]
                }
                for degradation in DEGRADATIONS
            }
        ]
        calibration = {
            "calibration_schema": "qwen_per_type_candidate_affine_v1",
            "validation_or_test_used_for_fit": False,
            "degradation_order": list(DEGRADATIONS),
            "temperature": {name: 1.0 for name in DEGRADATIONS},
            "class_bias": {name: [0.0] * 5 for name in DEGRADATIONS},
        }
        outputs, details = calibrated_per_type_outputs(scores, calibration)
        self.assertEqual(
            json.loads(outputs[0]),
            {name: 5 for name in DEGRADATIONS},
        )
        self.assertTrue(
            all(
                details[0][name]["raw_output"].endswith("L5.")
                for name in DEGRADATIONS
            )
        )

    def test_per_type_overfit_expansion_asks_all_five_questions(self):
        selected = select_overfit_records(self.records, seed=2026082506)
        dialogues = expand_per_type_records(selected, seed=2026082506)
        self.assertEqual(len(dialogues), 160)
        self.assertEqual(
            {
                name: sum(record["query_degradation"] == name for record in dialogues)
                for name in DEGRADATIONS
            },
            {name: 32 for name in DEGRADATIONS},
        )

    def test_semantic_loss_only_upweights_frozen_target_ids(self):
        labels = torch.tensor([[-100, 1, 2, 3]])
        logits = torch.zeros((1, 4, 5), dtype=torch.float32)
        logits[0, 0, 1] = 2.0
        logits[0, 1, 2] = 1.0
        logits[0, 2, 3] = 3.0
        loss = semantic_assistant_token_loss(
            logits,
            labels,
            semantic_token_ids=[2],
            semantic_weight=4.0,
        )
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 5), labels[:, 1:].reshape(-1), reduction="none"
        )
        expected = (token_losses[0] + 4 * token_losses[1] + token_losses[2]) / 6
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_semantic_loss_accepts_per_query_type_weights(self):
        labels = torch.tensor([[-100, 2], [-100, 2]])
        logits = torch.zeros((2, 2, 4), dtype=torch.float32)
        logits[0, 0, 2] = 1.0
        logits[1, 0, 2] = 3.0
        loss = semantic_assistant_token_loss(
            logits,
            labels,
            semantic_token_ids=[2],
            semantic_weight=8.0,
            example_semantic_weights=[12.0, 8.0],
        )
        token_losses = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1), reduction="none"
        )
        expected = (12 * token_losses[0] + 8 * token_losses[1]) / 20
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

    def test_decoder_presence_loss_uses_l1_vs_pooled_l2_l5_logits(self):
        labels = torch.tensor([[-100, 10], [-100, 13]])
        logits = torch.zeros((2, 2, 20), dtype=torch.float32, requires_grad=True)
        logits.data[0, 0, [10, 11, 12, 13, 14]] = torch.tensor(
            [3.0, 0.0, 0.0, 0.0, 0.0]
        )
        logits.data[1, 0, [10, 11, 12, 13, 14]] = torch.tensor(
            [0.0, 0.0, 0.0, 3.0, 0.0]
        )
        loss = decoder_presence_auxiliary_loss(
            logits,
            labels,
            level_token_ids=[10, 11, 12, 13, 14],
        )
        expected_scores = torch.stack(
            (
                torch.stack((torch.tensor(3.0), torch.log(torch.tensor(4.0)))),
                torch.stack(
                    (
                        torch.tensor(0.0),
                        torch.log(torch.exp(torch.tensor(3.0)) + 3.0),
                    )
                ),
            )
        )
        expected = torch.nn.functional.cross_entropy(
            expected_scores, torch.tensor([0, 1])
        )
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_training_loss_combines_decoder_presence_without_a_new_head(self):
        labels = torch.tensor([[-100, 10]])
        logits = torch.zeros((1, 2, 20), dtype=torch.float32, requires_grad=True)
        output = SimpleNamespace(logits=logits)
        training = {
            "semantic_token_loss": {
                "enabled": True,
                "semantic_token_ids": [10, 11, 12, 13, 14],
                "semantic_weight": 8.0,
                "semantic_weight_by_degradation": None,
            },
            "presence_auxiliary_loss": {
                "enabled": True,
                "weight": 1.0,
                "level_token_ids": [10, 11, 12, 13, 14],
            },
        }
        combined = training_loss(output, labels, training)
        semantic = semantic_assistant_token_loss(
            logits,
            labels,
            semantic_token_ids=[10, 11, 12, 13, 14],
            semantic_weight=8.0,
        )
        presence = decoder_presence_auxiliary_loss(
            logits,
            labels,
            level_token_ids=[10, 11, 12, 13, 14],
        )
        self.assertAlmostEqual(combined.item(), (semantic + presence).item(), places=6)

    def test_resumable_source_shards_are_disjoint_and_cover_one_epoch(self):
        records = [
            {
                "record_id": f"r{index:03d}",
                "source_id": f"s{index:03d}",
                "split": "train",
                "composition_size": index % 4,
                "degradations": [],
            }
            for index in range(40)
        ]
        selected_ids = []
        for shard in range(4):
            stage = {
                "name": f"shard{shard}",
                "source_order_key": "one_epoch",
                "source_start": shard * 10,
                "maximum_composition_size": 3,
                "source_records": 10,
                "records": 50,
            }
            dialogues = stage_records(
                records,
                stage,
                17,
                target_schema="per_type_dialogue_l1_absent_l2_l5_v1",
                global_batch_size=10,
            )
            ids = {record["record_id"] for record in dialogues}
            self.assertEqual(len(ids), 10)
            selected_ids.extend(ids)
        self.assertEqual(len(selected_ids), 40)
        self.assertEqual(len(set(selected_ids)), 40)

    def test_hard_triple_sampler_preserves_mixture_and_exact_quotas(self):
        records = []
        for size in range(4):
            combos = [()] if size == 0 else list(itertools.combinations(DEGRADATIONS, size))
            for index, combo in enumerate(combos):
                records.append(
                    {
                        "record_id": f"c{size}_{index}",
                        "source_id": f"source_c{size}_{index}",
                        "split": "train",
                        "composition_size": size,
                        "degradations": list(combo),
                    }
                )
        triple_quotas = {
            "+".join(combo): (
                480
                if "+".join(combo)
                in {
                    "blur+noise+rain",
                    "blur+noise+low_light",
                    "noise+rain+low_light",
                    "noise+haze+low_light",
                }
                else 240
            )
            for combo in itertools.combinations(DEGRADATIONS, 3)
        }
        stage = {
            "name": "test_hard_triple",
            "maximum_composition_size": 3,
            "source_records": 8400,
            "records": 42000,
            "source_sampling": {
                "schema": "hard_triple_mixture_v1",
                "composition_source_quotas": {
                    "0": 600,
                    "1": 2200,
                    "2": 2240,
                    "3": 3360,
                },
                "triple_combination_source_quotas": triple_quotas,
            },
        }
        dialogues = stage_records(
            records,
            stage,
            2026082839,
            target_schema="per_type_dialogue_l1_absent_l2_l5_v1",
            global_batch_size=200,
        )
        self.assertEqual(len(dialogues), 42000)
        source_presentations = {
            size: sum(int(row["composition_size"]) == size for row in dialogues) // 5
            for size in range(4)
        }
        self.assertEqual(source_presentations, {0: 600, 1: 2200, 2: 2240, 3: 3360})
        for combo, quota in triple_quotas.items():
            observed = sum(
                "+".join(row["degradations"]) == combo for row in dialogues
            ) // 5
            self.assertEqual(observed, quota)

    def test_baseline_training_loss_is_unchanged(self):
        class Output:
            loss = torch.tensor(1.25)

        labels = torch.tensor([[-100, 1]])
        self.assertIs(training_loss(Output(), labels, self.config["training"]), Output.loss)

    def test_optimizer_uses_separate_merger_rate(self):
        model = TinyScope()
        optimizer = create_optimizer(model, self.config)
        rates = {group["kind"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(rates["lora"], self.config["training"]["lora_learning_rate"])
        self.assertEqual(rates["merger"], self.config["training"]["merger_learning_rate"])

    def test_crossmodal_optimizer_uses_separate_decoder_rate(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_lora_correction3_v1.json"
        )
        optimizer = create_optimizer(TinyCrossmodalScope(), config)
        rates = {group["kind"]: group["lr"] for group in optimizer.param_groups}
        self.assertEqual(
            rates["language_lora"],
            config["training"]["language_lora_learning_rate"],
        )
        self.assertEqual(rates["lora"], config["training"]["lora_learning_rate"])
        self.assertEqual(rates["merger"], config["training"]["merger_learning_rate"])

    def test_gate_checks_include_triples_and_gradients(self):
        metrics = {
            "valid_structured_output_rate": 1.0,
            "type_macro_f1": 1.0,
            "type_micro_f1": 1.0,
            "degradation_set_exact_match_rate": 1.0,
            "combined_label_exact_match_rate": 1.0,
            "severity_typed_accuracy": 1.0,
            "severity_typed_mae_with_missing_penalty": 0.0,
            "by_composition_size": {"3": {"degradation_set_exact_match_rate": 1.0}},
        }
        self.assertTrue(all(gate_checks(metrics, self.config, True).values()))
        self.assertFalse(gate_checks(metrics, self.config, False)["gradient_scope"])

    def test_v2b_validation_and_protected_test_use_frozen_distinct_triple_gates(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_visibility_v2b_hard_triple.json"
        )
        metrics = {
            "valid_structured_output_rate": 1.0,
            "type_macro_f1": 1.0,
            "type_micro_f1": 1.0,
            "degradation_set_exact_match_rate": 1.0,
            "combined_label_exact_match_rate": 1.0,
            "severity_typed_accuracy": 1.0,
            "severity_typed_mae_with_missing_penalty": 0.0,
            "per_type_valid_response_rate": 1.0,
            "per_type_level_accuracy": 1.0,
            "by_composition_size": {
                "3": {"degradation_set_exact_match_rate": 0.81}
            },
        }
        self.assertFalse(gate_checks(metrics, config, True, split="validation")["triple_set_exact"])
        self.assertTrue(gate_checks(metrics, config, True, split="test")["triple_set_exact"])

    def test_overfit_selection_covers_all_types_levels_and_compositions(self):
        selected = select_overfit_records(
            self.records, seed=self.config["overfit"]["selection_seed"]
        )
        self.assertEqual(len(selected), 32)
        self.assertEqual(
            {size: sum(record["composition_size"] == size for record in selected) for size in range(4)},
            {0: 8, 1: 8, 2: 8, 3: 8},
        )
        covered = {
            (name, level)
            for record in selected
            for name, level in record["levels"].items()
        }
        self.assertEqual(len(covered), 20)

    def test_two_and_four_gpu_microbatches_cover_the_same_global_batch(self):
        records = [{"index": index} for index in range(32)]
        for world, accumulation in ((2, 2), (4, 1)):
            selected = [
                record["index"]
                for accumulation_index in range(accumulation)
                for rank in range(world)
                for record in rank_microbatch(
                    records,
                    rank=rank,
                    world=world,
                    micro=8,
                    accumulation_index=accumulation_index,
                )
            ]
            self.assertEqual(selected, list(range(32)))

    def test_validation_only_recovery_verifies_every_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stages = [
                {"name": "epoch1", "records": 20, "epochs": 1},
                {"name": "epoch2", "records": 20, "epochs": 1},
            ]
            for index, stage in enumerate(stages, start=1):
                directory = output / "checkpoints" / stage["name"]
                directory.mkdir(parents=True)
                adapter = directory / "adapter.safetensors"
                optimizer = directory / "optimizer.pt"
                adapter.write_bytes(f"adapter-{index}".encode())
                optimizer.write_bytes(f"optimizer-{index}".encode())
                metadata = {
                    "config_sha256": "config",
                    "variant": "merger_trainable",
                    "world_size": 2,
                    "gradient_accumulation_steps": 2,
                    "global_step": index * 2,
                    "adapter_sha256": __import__("hashlib").sha256(
                        adapter.read_bytes()
                    ).hexdigest(),
                    "optimizer_sha256": __import__("hashlib").sha256(
                        optimizer.read_bytes()
                    ).hexdigest(),
                }
                (directory / "metadata.json").write_text(json.dumps(metadata))
            evidence = validate_completed_stage_checkpoints(
                output=output,
                stages=stages,
                config_sha="config",
                variant="merger_trainable",
                world_size=2,
                accumulation_steps=2,
                global_batch_size=10,
            )
            self.assertEqual([item["global_step"] for item in evidence], [2, 4])
            (output / "checkpoints/epoch2/adapter.safetensors").write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                validate_completed_stage_checkpoints(
                    output=output,
                    stages=stages,
                    config_sha="config",
                    variant="merger_trainable",
                    world_size=2,
                    accumulation_steps=2,
                    global_batch_size=10,
                )

    def test_external_continuation_restores_optimizer_and_step_offset(self):
        config, _ = load_config(
            ROOT / "configs/qwen_multidegradation_per_type_rank32_continue_v2.json"
        )
        model = TinyCrossmodalScope()
        optimizer = create_optimizer(model, config)
        model.lora_A.grad = torch.ones_like(model.lora_A)
        model.model.visual.merger.weight.grad = torch.ones_like(
            model.model.visual.merger.weight
        )
        model.model.visual.merger.bias.grad = torch.ones_like(
            model.model.visual.merger.bias
        )
        model.model.language_model.lora_A.grad = torch.ones_like(
            model.model.language_model.lora_A
        )
        optimizer.step()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = root / "adapter.safetensors"
            optimizer_path = root / "optimizer.pt"
            adapter.write_bytes(b"adapter")
            torch.save(optimizer.state_dict(), optimizer_path)
            initialization = {
                "adapter": str(adapter),
                "adapter_sha256": __import__("hashlib").sha256(
                    adapter.read_bytes()
                ).hexdigest(),
                "optimizer": str(optimizer_path),
                "optimizer_sha256": __import__("hashlib").sha256(
                    optimizer_path.read_bytes()
                ).hexdigest(),
                "initial_optimizer_steps": 1440,
            }
            resumed_model = TinyCrossmodalScope()
            resumed_optimizer = create_optimizer(resumed_model, config)
            with (
                patch(
                    "smoke.train_qwen_multidegradation_lora_ddp.workspace_path",
                    side_effect=lambda value: Path(value),
                ),
                patch(
                    "smoke.train_qwen_multidegradation_lora_ddp.load_qwen_adapter"
                ) as load_adapter,
            ):
                completed, step = resume_stage(
                    resumed_model,
                    resumed_optimizer,
                    output=root / "new-output",
                    stages=config["data"]["curriculum"],
                    config_sha="config",
                    variant="merger_trainable",
                    world_size=8,
                    accumulation_steps=1,
                    global_batch_size=200,
                    initialization=initialization,
                    allow_world_size_change=True,
                )
            self.assertEqual(completed, 0)
            self.assertEqual(step, 1440)
            load_adapter.assert_called_once_with(resumed_model, adapter)
            self.assertEqual(len(resumed_optimizer.state), len(optimizer.state))

    def test_checkpoint_validation_allows_two_to_eight_gpu_resume_at_same_global_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            stage = {"name": "epoch4", "records": 48000, "epochs": 1}
            directory = output / "checkpoints/epoch4"
            directory.mkdir(parents=True)
            adapter = directory / "adapter.safetensors"
            optimizer = directory / "optimizer.pt"
            adapter.write_bytes(b"adapter")
            optimizer.write_bytes(b"optimizer")
            metadata = {
                "config_sha256": "config",
                "variant": "merger_trainable",
                "world_size": 2,
                "gradient_accumulation_steps": 4,
                "global_batch_size": 200,
                "global_step": 1680,
                "adapter_sha256": __import__("hashlib").sha256(
                    adapter.read_bytes()
                ).hexdigest(),
                "optimizer_sha256": __import__("hashlib").sha256(
                    optimizer.read_bytes()
                ).hexdigest(),
            }
            (directory / "metadata.json").write_text(json.dumps(metadata))
            evidence = validate_completed_stage_checkpoints(
                output=output,
                stages=[stage],
                config_sha="config",
                variant="merger_trainable",
                world_size=8,
                accumulation_steps=1,
                global_batch_size=200,
                initial_global_step=1440,
                allow_world_size_change=True,
            )
            self.assertEqual(evidence[0]["global_step"], 1680)


if __name__ == "__main__":
    unittest.main()
