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


class FakeDecoderAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 2)
        self.v_proj = nn.Linear(4, 2)
        self.o_proj = nn.Linear(4, 4)


class FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeDecoderAttention()


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeDecoderLayer(), FakeDecoderLayer()])


class FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = FakeVisual()
        self.language_model = FakeLanguage()


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
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.model.language_model.parameters()
            )
        )
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

    def test_crossmodal_decoder_attention_lora_scope(self):
        model = FakeQwen()
        audit = configure_qwen_visual_lora(
            model,
            rank=2,
            alpha=4,
            dropout=0.0,
            merger_trainable=True,
            language_rank=2,
            language_alpha=4,
        )
        self.assertEqual(audit["matched_visual_lora_module_count"], 10)
        self.assertEqual(audit["matched_language_lora_module_count"], 8)
        self.assertGreater(audit["language_lora_trainable_parameters"], 0)
        self.assertTrue(
            all(
                ".lora_" in name
                for name in audit["trainable_names"]
                if name.startswith("model.language_model.")
            )
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

    def test_visual_adapter_can_initialize_new_decoder_lora(self):
        source = FakeQwen()
        configure_qwen_visual_lora(
            source, rank=2, alpha=4, dropout=0.0, merger_trainable=True
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual.safetensors"
            save_qwen_adapter(source, path, {"variant": "visual"})
            expanded = FakeQwen()
            configure_qwen_visual_lora(
                expanded,
                rank=2,
                alpha=4,
                dropout=0.0,
                merger_trainable=True,
                language_rank=2,
                language_alpha=4,
            )
            load_qwen_adapter(
                expanded,
                path,
                allow_missing_prefixes=("model.language_model.",),
            )
            for name, value in trainable_state_dict(source).items():
                self.assertTrue(
                    torch.equal(value, trainable_state_dict(expanded)[name]), name
                )


if __name__ == "__main__":
    unittest.main()
