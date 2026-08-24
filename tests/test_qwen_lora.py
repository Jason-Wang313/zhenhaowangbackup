import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lowlight_restoration.qwen_lora import (
    configure_qwen_visual_lora,
    load_qwen_adapter,
    save_qwen_adapter,
    trainable_state_dict,
)


class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(4, 12)
        self.proj = nn.Linear(4, 4)


class FakeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 8)
        self.up_proj = nn.Linear(4, 8)
        self.down_proj = nn.Linear(8, 4)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttention()
        self.mlp = FakeMlp()


class FakeVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])
        self.merger = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4))


class FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()
        self.language_model = nn.Linear(4, 4)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInner()
        self.lm_head = nn.Linear(4, 4)


class QwenLoraTests(unittest.TestCase):
    def test_frozen_merger_scope(self):
        model = FakeQwen()
        audit = configure_qwen_visual_lora(
            model, rank=2, alpha=4, dropout=0.0, merger_trainable=False
        )
        self.assertEqual(audit["matched_lora_module_count"], 10)
        self.assertEqual(audit["merger_trainable_parameters"], 0)
        self.assertTrue(all("lora_" in name for name in audit["trainable_names"]))
        self.assertFalse(model.model.language_model.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)

    def test_trainable_merger_scope(self):
        model = FakeQwen()
        audit = configure_qwen_visual_lora(
            model, rank=2, alpha=4, dropout=0.0, merger_trainable=True
        )
        self.assertGreater(audit["merger_trainable_parameters"], 0)
        self.assertTrue(
            any(name.startswith("model.visual.merger.") for name in audit["trainable_names"])
        )

    def test_adapter_round_trip_and_immutable_publish(self):
        model = FakeQwen()
        configure_qwen_visual_lora(
            model, rank=2, alpha=4, dropout=0.0, merger_trainable=True
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data.uniform_(-0.2, 0.2)
        expected = {key: value.clone() for key, value in trainable_state_dict(model).items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            summary = save_qwen_adapter(model, path, {"variant": "test"})
            self.assertEqual(len(summary["adapter_sha256"]), 64)
            restored = FakeQwen()
            configure_qwen_visual_lora(
                restored, rank=2, alpha=4, dropout=0.0, merger_trainable=True
            )
            load_qwen_adapter(restored, path)
            for key, value in trainable_state_dict(restored).items():
                self.assertTrue(torch.equal(value, expected[key]), key)
            with self.assertRaises(FileExistsError):
                save_qwen_adapter(model, path, {"variant": "test"})


if __name__ == "__main__":
    unittest.main()
