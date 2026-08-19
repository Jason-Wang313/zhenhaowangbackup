from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

from lowlight_restoration.data import PairRecord
from lowlight_restoration.qwen_supervised import (
    canonical_target,
    count_parameter_scope,
    evaluate_generated_outputs,
    parse_target,
    record_bundle_sha256,
    select_complete_sources,
)


def record(source: str, level: int, *, split: str = "train") -> PairRecord:
    digest = f"{level}" * 64
    return PairRecord(
        pair_id=f"{source}_L{level}",
        source_id=source,
        clean_path=Path(f"{source}_clean.png"),
        degraded_path=Path(f"{source}_L{level}.png"),
        project_level=f"L{level}",
        seed=level,
        split=split,
        clean_sha256=digest[:64],
        degraded_sha256=digest[:64],
    )


class QwenSupervisedSchemaTests(unittest.TestCase):
    def test_clean_l1_has_no_active_degradation(self) -> None:
        target = canonical_target(1)
        self.assertEqual(target, '{"degradations":[],"levels":{}}')
        self.assertEqual(parse_target(target), 1)

    def test_active_levels_have_independent_lowlight_level(self) -> None:
        for level in range(2, 6):
            target = canonical_target(level)
            self.assertEqual(
                json.loads(target),
                {
                    "degradations": ["low_light"],
                    "levels": {"low_light": level},
                },
            )
            self.assertEqual(parse_target(target), level)

    def test_forbidden_lowlight_l1_and_extra_fields_fail(self) -> None:
        invalid = [
            '{"degradations":["low_light"],"levels":{"low_light":1}}',
            '{"degradations":[],"levels":{},"severity":1}',
            '{"degradations":["noise"],"levels":{"noise":2}}',
            "not json",
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_target(value)

    def test_complete_source_selection_is_balanced_and_deterministic(self) -> None:
        records = [
            record(source, level) for source in ("a", "b", "c") for level in range(1, 6)
        ]
        selected = select_complete_sources(
            records, split="train", source_count=2, seed=123
        )
        repeated = select_complete_sources(
            list(reversed(records)), split="train", source_count=2, seed=123
        )
        self.assertEqual(
            [item.pair_id for item in selected], [item.pair_id for item in repeated]
        )
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            {
                level: sum(item.level_index == level for item in selected)
                for level in range(1, 6)
            },
            {level: 2 for level in range(1, 6)},
        )
        self.assertEqual(record_bundle_sha256(selected), record_bundle_sha256(repeated))

    def test_generated_metrics_keep_validity_and_canonical_form_separate(self) -> None:
        records = [record("a", 1), record("b", 2)]
        rows, metrics = evaluate_generated_outputs(
            records,
            [
                canonical_target(1),
                '{"levels":{"low_light":2}, "degradations":["low_light"]}',
            ],
        )
        self.assertTrue(rows[1]["valid_structured_output"])
        self.assertFalse(rows[1]["canonical_exact_match"])
        self.assertEqual(metrics["valid_structured_output_rate"], 1.0)
        self.assertEqual(metrics["severity_accuracy"], 1.0)
        self.assertEqual(metrics["canonical_exact_match_rate"], 0.5)

    def test_parameter_scope_matches_current_transformers_names(self) -> None:
        named = [
            (
                "model.visual.patch_embed.proj.weight",
                torch.nn.Parameter(torch.zeros(2, 3)),
            ),
            ("model.visual.blocks.0.norm1.weight", torch.nn.Parameter(torch.zeros(5))),
            ("model.visual.merger.mlp.0.weight", torch.nn.Parameter(torch.zeros(7, 2))),
            (
                "model.language_model.embed_tokens.weight",
                torch.nn.Parameter(torch.zeros(11, 3)),
            ),
            ("lm_head.weight", torch.nn.Parameter(torch.zeros(13, 3))),
        ]
        self.assertEqual(
            count_parameter_scope(named),
            {
                "visual.patch_embed": 6,
                "visual.blocks": 5,
                "visual.merger": 14,
                "language_model": 33,
                "lm_head": 39,
            },
        )


if __name__ == "__main__":
    unittest.main()
