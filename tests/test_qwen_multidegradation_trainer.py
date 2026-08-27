import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from smoke.train_qwen_multidegradation_lora_ddp import (
    create_optimizer,
    gate_checks,
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
