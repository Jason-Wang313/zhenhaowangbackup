from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lowlight_restoration.data import PairRecord
from lowlight_restoration.condition_adapter import LowLightLevelContext
from lowlight_restoration.qwen_condition import (
    FALLBACK_LEVEL,
    LowLightCondition,
    evaluate_predictions,
    parse_qwen_condition,
)
from predict_qwen_lowlight import reusable_record, select_records


class QwenConditionTests(unittest.TestCase):
    @staticmethod
    def records() -> list[PairRecord]:
        digest = "0" * 64
        return [
            PairRecord(
                pair_id=f"pair_L{level}_{index:02d}",
                source_id=f"source_L{level}_{index:02d}",
                clean_path=Path("clean.png"),
                degraded_path=Path("degraded.png"),
                project_level=f"L{level}",
                seed=index,
                split="test",
                clean_sha256=digest,
                degraded_sha256=digest,
            )
            for level in range(1, 6)
            for index in range(10)
        ]

    def test_exact_schema_parses_and_canonicalizes(self) -> None:
        condition = parse_qwen_condition(
            '{"degradations":["low_light"],"levels":{"low_light":5}}'
        )
        self.assertEqual(condition, LowLightCondition(5))
        self.assertEqual(
            condition.canonical_json(),
            '{"degradations":["low_light"],"levels":{"low_light":5}}',
        )

    def test_schema_rejects_repairs_extra_fields_and_boolean_levels(self) -> None:
        invalid = [
            "```json\n{\"degradations\":[\"low_light\"],\"levels\":{\"low_light\":2}}\n```",
            '{"degradations":["low_light"],"levels":{"low_light":2},"caption":"dark"}',
            '{"degradations":["low_light","noise"],"levels":{"low_light":2}}',
            '{"degradations":["low_light"],"levels":{"low_light":true}}',
            '{"degradations":["low_light"],"levels":{"low_light":0}}',
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_qwen_condition(value)

    def test_metrics_use_frozen_fallback_for_invalid_outputs(self) -> None:
        records = [
            {"true_level": 1, "predicted_level": 1, "valid_structured_output": True},
            {"true_level": 2, "predicted_level": 3, "valid_structured_output": True},
            {"true_level": FALLBACK_LEVEL, "predicted_level": None, "valid_structured_output": False},
            {"true_level": 4, "predicted_level": 4, "valid_structured_output": True},
            {"true_level": 5, "predicted_level": 5, "valid_structured_output": True},
        ]
        metrics = evaluate_predictions(records)
        self.assertEqual(metrics["valid_structured_output_rate"], 0.8)
        self.assertEqual(metrics["severity_accuracy"], 0.8)
        self.assertEqual(metrics["severity_mean_absolute_error"], 0.2)

    def test_stratified_selection_is_balanced_and_deterministic(self) -> None:
        first = select_records(self.records(), split="test", sample_count=25, seed=17)
        second = select_records(self.records(), split="test", sample_count=25, seed=17)
        self.assertEqual([record.pair_id for record in first], [record.pair_id for record in second])
        self.assertEqual(
            {level: sum(record.level_index == level for record in first) for level in range(1, 6)},
            {level: 5 for level in range(1, 6)},
        )

    def test_reusable_prediction_is_fail_closed(self) -> None:
        pair = self.records()[0]
        payload = {
            "pair_id": pair.pair_id,
            "true_level": pair.level_index,
            "degraded_sha256": pair.degraded_sha256,
            "config_sha256": "a" * 64,
            "model_revision": "b" * 40,
            "raw_output": '{"degradations":["low_light"],"levels":{"low_light":1}}',
            "valid_structured_output": True,
            "predicted_level": 1,
            "used_level": 1,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            path.write_text(json.dumps(payload))
            reused = reusable_record(
                path,
                pair=pair,
                config_sha256="a" * 64,
                model_revision="b" * 40,
            )
            self.assertEqual(reused["predicted_level"], 1)
            payload["true_level"] = 2
            path.write_text(json.dumps(payload))
            with self.assertRaises(RuntimeError):
                reusable_record(
                    path,
                    pair=pair,
                    config_sha256="a" * 64,
                    model_revision="b" * 40,
                )

    def test_context_none_is_exact_baseline_and_levels_append_one_token(self) -> None:
        import torch

        torch.manual_seed(5)
        adapter = LowLightLevelContext(context_dim=8)
        context = torch.randn(2, 3, 8)
        context_ids = torch.zeros(2, 3, 4, dtype=torch.int64)
        context_ids[..., 3] = torch.arange(3)
        baseline_context, baseline_ids = adapter(context, context_ids, None)
        self.assertIs(baseline_context, context)
        self.assertIs(baseline_ids, context_ids)
        conditioned, conditioned_ids = adapter(
            context,
            context_ids,
            torch.tensor([1, 5], dtype=torch.int64),
        )
        self.assertEqual(conditioned.shape, (2, 4, 8))
        self.assertEqual(conditioned_ids.shape, (2, 4, 4))
        torch.testing.assert_close(conditioned[:, :3], context, rtol=0, atol=0)
        torch.testing.assert_close(conditioned_ids[:, :3], context_ids, rtol=0, atol=0)
        self.assertEqual(conditioned_ids[:, -1, 3].tolist(), [3, 3])
        self.assertFalse(torch.equal(conditioned[0, -1], conditioned[1, -1]))


if __name__ == "__main__":
    unittest.main()
