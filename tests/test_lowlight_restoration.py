from __future__ import annotations

import csv
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from lowlight_restoration.config import TrainConfig, load_train_config
from lowlight_restoration.data import LowLightPairDataset, load_pair_manifest
from lowlight_restoration.flow import build_reference_flow_batch
from lowlight_restoration.lora import (
    LoRALinear,
    inject_lora,
    load_lora_checkpoint,
    save_lora_checkpoint,
)
from smoke.evaluate_lowlight_overfit import (
    require_finite_numbers as require_finite_evaluation_metrics,
    validate_reusable_evaluation,
)
from train_flux2_lowlight import (
    ROOT,
    checkpoint_paths,
    deterministic_index,
    sha256,
    write_training_summary,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.qkv = nn.Linear(4, 12, bias=False)
        self.block.proj = nn.Linear(12, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block.proj(self.block.qkv(value))


def pack_image(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, height, width = latents.shape
    tokens = latents.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    ids = torch.zeros(batch, height * width, 4, dtype=torch.int64)
    index = 0
    for row in range(height):
        for column in range(width):
            ids[:, index, 1] = row
            ids[:, index, 2] = column
            index += 1
    return tokens, ids


class LoRATests(unittest.TestCase):
    def test_zero_initialization_preserves_baseline_and_gradients_are_scoped(self) -> None:
        torch.manual_seed(7)
        model = TinyModel()
        value = torch.randn(2, 3, 4)
        baseline = model(value).detach()
        names = inject_lora(model, ["block.qkv", "block.proj"], rank=2, alpha=2)
        self.assertEqual(names, ["block.qkv", "block.proj"])
        torch.testing.assert_close(model(value), baseline, rtol=0, atol=0)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertEqual(
            trainable,
            [
                "block.qkv.lora_A.weight",
                "block.qkv.lora_B.weight",
                "block.proj.lora_A.weight",
                "block.proj.lora_B.weight",
            ],
        )
        model(value).sum().backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))

    def test_checkpoint_round_trip(self) -> None:
        torch.manual_seed(11)
        model = TinyModel()
        inject_lora(model, ["block.*"], rank=2, alpha=2)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.lora_B.weight.normal_()
        expected = {name: value.detach().clone() for name, value in model.named_parameters() if "lora_" in name}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_lora_checkpoint(model, path, {"step": 3, "kind": "test"})
            with torch.no_grad():
                for name, value in model.named_parameters():
                    if "lora_" in name:
                        value.zero_()
            load_lora_checkpoint(model, path)
            for name, value in model.named_parameters():
                if "lora_" in name:
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


class FlowTests(unittest.TestCase):
    def test_reference_flow_formula_and_layout(self) -> None:
        clean = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        degraded = clean * 0.25
        noise = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]])
        batch = build_reference_flow_batch(
            clean,
            degraded,
            noise,
            torch.tensor([0.25]),
            pack_image,
        )
        expected_noisy = 0.75 * clean + 0.25 * noise
        expected_velocity = noise - clean
        expected_noisy_tokens, _ = pack_image(expected_noisy)
        expected_velocity_tokens, _ = pack_image(expected_velocity)
        torch.testing.assert_close(batch.model_tokens[:, :4], expected_noisy_tokens)
        torch.testing.assert_close(batch.model_tokens[:, 4:], pack_image(degraded)[0])
        torch.testing.assert_close(batch.target_velocity, expected_velocity_tokens)
        self.assertTrue(torch.all(batch.model_ids[:, :4, 0] == 0))
        self.assertTrue(torch.all(batch.model_ids[:, 4:, 0] == 10))


class DatasetTests(unittest.TestCase):
    def test_aligned_crop_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = np.zeros((32, 32, 3), dtype=np.uint8)
            clean[..., 0] = np.arange(32, dtype=np.uint8)[None, :]
            clean[..., 1] = np.arange(32, dtype=np.uint8)[:, None]
            degraded = clean // 2
            Image.fromarray(clean).save(root / "clean.png")
            Image.fromarray(degraded).save(root / "degraded.png")
            manifest = root / "pairs.csv"
            fields = [
                "pair_id",
                "source_id",
                "clean_path",
                "degraded_path",
                "degradation_type",
                "project_level",
                "seed",
                "split",
                "clean_sha256",
                "degraded_sha256",
            ]
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "sample_L3",
                        "source_id": "sample",
                        "clean_path": "clean.png",
                        "degraded_path": "degraded.png",
                        "degradation_type": "low_light",
                        "project_level": "L3",
                        "seed": "1",
                        "split": "train",
                        "clean_sha256": "a" * 64,
                        "degraded_sha256": "b" * 64,
                    }
                )
            records = load_pair_manifest(manifest, root=root)
            dataset = LowLightPairDataset(records, split="train", crop_size=16, seed=99)
            first = dataset[0]
            second = dataset[0]
            torch.testing.assert_close(first["clean"], second["clean"])
            torch.testing.assert_close(first["degraded"], second["degraded"])
            torch.testing.assert_close(first["crop_xy"], second["crop_xy"])
            self.assertEqual(first["project_level"], "L3")
            self.assertEqual(first["level_index"], 3)


class TrainingControlTests(unittest.TestCase):
    def test_config_rejects_invalid_runtime_and_boolean_numeric_values(self) -> None:
        base = json.loads((ROOT / "configs/lowlight_flux2_overfit16.json").read_text())
        cases = [
            (("training", "learning_rate"), True, "learning_rate"),
            (("training", "reference_time_coordinate"), -1, "non-negative"),
            (("lora", "dropout"), 1.0, "dropout"),
            (("runtime", "minimum_vram_gib"), 0, "positive integer"),
            (("inference", "num_steps"), 0, "positive integer"),
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            for index, (keys, value, message) in enumerate(cases):
                raw = copy.deepcopy(base)
                raw[keys[0]][keys[1]] = value
                path = Path(directory) / f"invalid_{index}.json"
                path.write_text(json.dumps(raw))
                with self.subTest(keys=keys, value=value):
                    with self.assertRaisesRegex(ValueError, message):
                        load_train_config(path)

    def test_deterministic_index_ordering_is_epoch_stable(self) -> None:
        length = 7
        seed = 2026081502
        first = [deterministic_index(position, length, seed) for position in range(length * 2)]
        second = [deterministic_index(position, length, seed) for position in range(length * 2)]
        self.assertEqual(first, second)
        self.assertEqual(sorted(first[:length]), list(range(length)))
        self.assertEqual(sorted(first[length:]), list(range(length)))
        with self.assertRaises(ValueError):
            deterministic_index(0, 0, seed)

    def test_interrupted_summary_recovers_final_checkpoint_metric(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output_dir = Path(directory)
            config_path = output_dir / "source_config.json"
            raw = {"run_name": "recovery_test"}
            config_path.write_text(json.dumps(raw))
            config = TrainConfig(config_path, raw)
            checkpoint, optimizer, metadata = checkpoint_paths(output_dir, 4)
            checkpoint.write_bytes(b"adapter checkpoint")
            optimizer.write_bytes(b"optimizer checkpoint")
            metric = {
                "step": 4,
                "loss": 1.5,
                "probe_loss": 0.75,
                "grad_norm": 0.5,
            }
            metadata.write_text(
                json.dumps(
                    {
                        "step": 4,
                        "config_sha256": config.sha256,
                        "checkpoint_sha256": sha256(checkpoint),
                        "optimizer_sha256": sha256(optimizer),
                        "metric": metric,
                    }
                )
            )
            (output_dir / "probe_baseline.json").write_text(
                json.dumps(
                    {
                        "step": 0,
                        "config_sha256": config.sha256,
                        "probe_loss": 2.0,
                    }
                )
            )
            self.assertFalse((output_dir / "metrics.jsonl").exists())
            summary = write_training_summary(output_dir, config, 12, 4)
            recovered = json.loads((output_dir / "metrics.jsonl").read_text())
            self.assertEqual(recovered, metric)
            self.assertEqual(summary["checkpoint_sha256"], sha256(checkpoint))
            self.assertTrue(summary["loss_decreased"])


class EvaluationReuseTests(unittest.TestCase):
    def test_non_finite_metric_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite metric"):
            require_finite_evaluation_metrics({"restored_psnr": float("nan")})

    def test_failed_evaluation_is_never_reused(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.safetensors"
            restored = root / "restored.png"
            contact = root / "contact.jpg"
            checkpoint.write_bytes(b"checkpoint")
            restored.write_bytes(b"restored")
            contact.write_bytes(b"contact")
            existing = {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "config_sha256": "a" * 64,
                "samples": 1,
                "guidance": 1.0,
                "evaluation_name": "evaluation_test",
                "automatic_gate_pass": False,
                "per_image": [{"restored_path": str(restored.relative_to(ROOT))}],
                "contact_sheets": [str(contact.relative_to(ROOT))],
            }
            with self.assertRaisesRegex(RuntimeError, "failed automatic gate"):
                validate_reusable_evaluation(
                    existing,
                    checkpoint=checkpoint,
                    config_sha256="a" * 64,
                    samples=1,
                    guidance=1.0,
                    evaluation_name="evaluation_test",
                )


if __name__ == "__main__":
    unittest.main()
