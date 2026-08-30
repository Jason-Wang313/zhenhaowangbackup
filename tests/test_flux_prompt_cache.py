from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from lowlight_restoration.flux_prompt_cache import PromptContextCache, prompt_sha256
from smoke.build_foundir_flux_qwen3_prompt_cache_ddp import cache_path, publish_context, verify_entry


class FluxPromptCacheTests(unittest.TestCase):
    def test_disk_cache_verifies_and_evicts_lru(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = ["restore blur", "restore noise"]
            entries = {}
            for index, prompt in enumerate(prompts):
                path = cache_path(root, prompt)
                entries[prompt_sha256(prompt)] = publish_context(
                    path,
                    prompt,
                    torch.full((3, 4), float(index), dtype=torch.bfloat16),
                    torch.zeros((3, 4), dtype=torch.int64),
                )
            index_path = root / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "status": "frozen_flux_qwen3_prompt_context_cache",
                        "prompts": 2,
                        "entries": entries,
                    }
                )
            )
            cache = PromptContextCache(index_path, maximum_resident=1)
            context, ids = cache.get(prompts[0], torch.device("cpu"))
            self.assertEqual(context.shape, (1, 3, 4))
            self.assertEqual(ids.shape, (1, 3, 4))
            cache.get(prompts[1], torch.device("cpu"))
            self.assertEqual(len(cache.resident), 1)
            self.assertIn(prompt_sha256(prompts[1]), cache.resident)

    def test_unknown_prompt_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "frozen_flux_qwen3_prompt_context_cache",
                        "prompts": 0,
                        "entries": {},
                    }
                )
            )
            cache = PromptContextCache(path)
            with self.assertRaises(KeyError):
                cache.get("missing", torch.device("cpu"))

    def test_orphan_tensor_recovers_missing_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = "restore haze"
            path = cache_path(root, prompt)
            path.parent.mkdir(parents=True)
            save_file(
                {
                    "context": torch.ones((3, 4), dtype=torch.bfloat16),
                    "context_ids": torch.zeros((3, 4), dtype=torch.int64),
                },
                str(path),
            )
            entry = verify_entry(path, prompt)
            self.assertTrue(path.with_suffix(path.suffix + ".json").is_file())
            self.assertEqual(entry["path"], path.relative_to(root).as_posix())


if __name__ == "__main__":
    unittest.main()
