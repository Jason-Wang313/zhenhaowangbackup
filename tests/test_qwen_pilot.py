from __future__ import annotations

import unittest
from pathlib import Path

from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.qwen_pilot import (
    augment_understanding_metrics,
    epoch_local_order,
    load_pilot_config,
    metric_gate_checks,
    records_by_split,
    select_best_validation_checkpoint,
    training_position,
    validate_pilot_contract,
)
from lowlight_restoration.qwen_supervised import (
    canonical_target,
    evaluate_generated_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen_lowlight_supervised_pilot_8k_ddp8.json"


class QwenPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.config_sha, _ = load_pilot_config(CONFIG)

    def test_checked_in_contract_and_split_bundles(self) -> None:
        validate_pilot_contract(self.config)
        self.assertEqual(len(self.config_sha), 64)
        records = load_pair_manifest(ROOT / "manifests/low_light_pilot_10000.csv")
        splits = records_by_split(records, self.config)
        self.assertEqual(
            {name: len(values) for name, values in splits.items()},
            {"train": 8000, "validation": 1000, "test": 1000},
        )

    def test_epoch_order_is_a_deterministic_local_permutation(self) -> None:
        first = epoch_local_order(
            local_source_count=200, seed=2026081905, epoch_index=0, rank=3
        )
        repeated = epoch_local_order(
            local_source_count=200, seed=2026081905, epoch_index=0, rank=3
        )
        next_epoch = epoch_local_order(
            local_source_count=200, seed=2026081905, epoch_index=1, rank=3
        )
        self.assertEqual(first, repeated)
        self.assertEqual(sorted(first), list(range(200)))
        self.assertNotEqual(first, next_epoch)

    def test_training_position_respects_epoch_boundaries(self) -> None:
        self.assertEqual(training_position(1, steps_per_epoch=200), (0, 0))
        self.assertEqual(training_position(200, steps_per_epoch=200), (0, 199))
        self.assertEqual(training_position(201, steps_per_epoch=200), (1, 0))
        self.assertEqual(training_position(1000, steps_per_epoch=200), (4, 199))

    def test_validation_selection_uses_loss_then_earlier_step(self) -> None:
        selected = select_best_validation_checkpoint(
            [
                {"step": 200, "teacher_forced_loss": 0.4},
                {"step": 400, "teacher_forced_loss": 0.2},
                {"step": 600, "teacher_forced_loss": 0.2},
            ],
            allowed_steps=[200, 400, 600],
        )
        self.assertEqual(selected["step"], 400)
        with self.assertRaises(ValueError):
            select_best_validation_checkpoint(
                [{"step": 200, "teacher_forced_loss": 0.4}],
                allowed_steps=[200, 400],
            )

    def test_augmented_perfect_metrics_pass_pilot_thresholds(self) -> None:
        records = load_pair_manifest(ROOT / "manifests/low_light_pilot_10000.csv")
        source = records[0].source_id
        chosen = [record for record in records if record.source_id == source]
        rows, base = evaluate_generated_outputs(
            chosen, [canonical_target(record.level_index) for record in chosen]
        )
        metrics = augment_understanding_metrics(rows, base)
        self.assertEqual(metrics["degradation_type_macro_f1"], 1.0)
        self.assertEqual(metrics["severity_macro_f1"], 1.0)
        gate = self.config["gate"]
        assert isinstance(gate, dict)
        self.assertTrue(all(metric_gate_checks(metrics, gate, prefix="test").values()))
        malformed = dict(metrics)
        malformed["per_level"] = dict(metrics["per_level"])
        malformed["per_level"]["3"] = None
        with self.assertRaises(ValueError):
            metric_gate_checks(malformed, gate, prefix="test")


if __name__ == "__main__":
    unittest.main()
