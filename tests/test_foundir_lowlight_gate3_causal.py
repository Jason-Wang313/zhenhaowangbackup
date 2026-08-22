from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from lowlight_restoration.causal_condition import (
    CausalToneAdapter,
    group_derangement,
    measured_image_features,
    oracle_pair_features,
    qwen_condition_from_scores,
)


class Gate3CausalTests(unittest.TestCase):
    def test_gpu_execution_fields_are_complete(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/foundir_lowlight_gate3_causal_v1.json").read_text())
        runtime = config["runtime"]
        for key in (
            "seed", "visible_devices", "world_size", "qwen_python",
            "adapter_metric_python", "minimum_gpu_memory_mib",
        ):
            self.assertIn(key, runtime)
        self.assertEqual(runtime["world_size"], 4)
        self.assertEqual(runtime["visible_devices"], "0,1,2,3")
        self.assertEqual(config["training"]["global_batch_size"], runtime["world_size"] * config["training"]["batch_size_per_rank"])
        self.assertEqual(config["training"]["sample_presentations"], config["training"]["optimizer_steps"] * config["training"]["global_batch_size"])

    def test_feature_dimensions_are_stable(self) -> None:
        source = np.arange(32 * 40 * 3, dtype=np.uint16).reshape(32, 40, 3).astype(np.uint8)
        target = np.flip(source, axis=1).copy()
        self.assertEqual(measured_image_features(source).shape, (76,))
        self.assertEqual(oracle_pair_features(source, target).shape, (40,))
        qwen = qwen_condition_from_scores({str(level): float(level) for level in range(1, 6)}, [0.1] * 32)
        self.assertEqual(qwen.shape, (40,))
        self.assertAlmostEqual(float(qwen[:5].sum()), 1.0, places=6)

    def test_zero_start_adapter_preserves_frozen_tone(self) -> None:
        adapter = CausalToneAdapter(76, 40, 40, width=16)
        output = adapter(torch.randn(3, 76), torch.randn(3, 40), None)
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_shuffle_crosses_groups_but_not_families(self) -> None:
        rows = [
            {"record_id": f"{family}-{group}-{index}", "family": family, "group_id": group}
            for family in ("a", "b")
            for group in ("g1", "g2", "g3")
            for index in range(2)
        ]
        by_id = {row["record_id"]: row for row in rows}
        mapping = group_derangement(rows, 7)
        self.assertEqual(set(mapping), set(by_id))
        for destination, source in mapping.items():
            self.assertEqual(by_id[destination]["family"], by_id[source]["family"])
            self.assertNotEqual(by_id[destination]["group_id"], by_id[source]["group_id"])


if __name__ == "__main__":
    unittest.main()
