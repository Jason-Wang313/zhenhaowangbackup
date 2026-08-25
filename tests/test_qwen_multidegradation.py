import json
import unittest

from lowlight_restoration.qwen_multidegradation import (
    DEGRADATIONS,
    FIXED_FIVE_SLOT_SCHEMA,
    DegradationLabel,
    canonical_fixed_five_slot_target,
    canonical_restoration_text,
    canonical_target,
    evaluate_outputs,
    make_label,
    parse_fixed_five_slot_target,
    parse_target,
)


class QwenMultiDegradationSchemaTests(unittest.TestCase):
    def test_fixed_five_slot_schema_is_independent_and_canonical(self):
        label = make_label(
            ["blur", "haze", "low_light"],
            {"blur": 2, "haze": 4, "low_light": 5},
        )
        target = canonical_fixed_five_slot_target(label)
        self.assertEqual(
            target,
            '{"blur":2,"noise":1,"rain":1,"haze":4,"low_light":5}',
        )
        self.assertEqual(parse_fixed_five_slot_target(target), label)
        self.assertEqual(
            canonical_fixed_five_slot_target(make_label([], {})),
            '{"blur":1,"noise":1,"rain":1,"haze":1,"low_light":1}',
        )

    def test_fixed_five_slot_schema_rejects_drift_and_four_active_types(self):
        rejected = [
            '{"noise":1,"blur":1,"rain":1,"haze":1,"low_light":1}',
            '{"blur":0,"noise":1,"rain":1,"haze":1,"low_light":1}',
            '{"blur":true,"noise":1,"rain":1,"haze":1,"low_light":1}',
            '{"blur":2,"noise":2,"rain":2,"haze":2,"low_light":1}',
        ]
        for raw in rejected:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_fixed_five_slot_target(raw)
    def test_clean_is_only_l1_representation(self):
        clean = make_label([], {})
        self.assertEqual(canonical_target(clean), '{"degradations":[],"levels":{}}')
        self.assertEqual(parse_target(canonical_target(clean)), clean)
        for name in DEGRADATIONS:
            with self.assertRaises(ValueError):
                make_label([name], {name: 1})

    def test_single_double_triple_canonical_order(self):
        cases = [
            (["noise"], {"noise": 2}),
            (["low_light", "blur"], {"low_light": 5, "blur": 3}),
            (["haze", "rain", "noise"], {"haze": 4, "rain": 2, "noise": 5}),
        ]
        expected_names = [("noise",), ("blur", "low_light"), ("noise", "rain", "haze")]
        for (names, levels), expected in zip(cases, expected_names, strict=True):
            label = make_label(names, levels)
            self.assertEqual(label.degradations, expected)
            self.assertEqual(parse_target(canonical_target(label)), label)

    def test_rejects_four_types_and_non_independent_level_maps(self):
        with self.assertRaises(ValueError):
            make_label(DEGRADATIONS[:4], {name: 2 for name in DEGRADATIONS[:4]})
        with self.assertRaises(ValueError):
            make_label(["blur", "noise"], {"blur": 2})
        with self.assertRaises(ValueError):
            make_label([], {"noise": 2})

    def test_strict_parser_rejects_noncanonical_json(self):
        rejected = [
            '{"levels":{},"degradations":[]}',
            '{ "degradations":[],"levels":{} }',
            '{"degradations":["noise","blur"],"levels":{"noise":2,"blur":2}}',
            '{"degradations":["blur"],"levels":{"blur":2},"extra":1}',
            '{"degradations":["fog"],"levels":{"fog":2}}',
            '{"degradations":["blur"],"levels":{"blur":true}}',
            '```json\n{"degradations":[],"levels":{}}\n```',
            '{"degradations":[],"levels":{}}\n',
        ]
        for raw in rejected:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_target(raw)

    def test_restoration_text_is_deterministic_and_complete(self):
        label = make_label(["low_light", "noise"], {"low_light": 5, "noise": 3})
        text = canonical_restoration_text(label)
        self.assertEqual(
            text,
            "Restore the image by correcting noise at moderate strength (L3), "
            "low_light at extreme strength (L5).",
        )
        self.assertIn("clean", canonical_restoration_text(make_label([], {})))


class QwenMultiDegradationMetricTests(unittest.TestCase):
    def setUp(self):
        self.truths = [
            make_label([], {}),
            make_label(["blur"], {"blur": 2}),
            make_label(["noise", "rain"], {"noise": 3, "rain": 4}),
            make_label(
                ["haze", "low_light", "blur"],
                {"haze": 5, "low_light": 4, "blur": 3},
            ),
        ]

    def test_perfect_metrics_cover_all_compositions(self):
        _, metrics = evaluate_outputs(
            self.truths, [canonical_target(label) for label in self.truths]
        )
        for key in (
            "valid_structured_output_rate",
            "degradation_set_exact_match_rate",
            "combined_label_exact_match_rate",
            "type_micro_f1",
            "type_macro_f1",
            "severity_typed_accuracy",
        ):
            self.assertEqual(metrics[key], 1.0, key)
        self.assertEqual(metrics["severity_typed_mae_with_missing_penalty"], 0.0)
        self.assertEqual(metrics["active_severity_support"], 6)
        for size in range(4):
            self.assertEqual(metrics["by_composition_size"][str(size)]["samples"], 1)

    def test_fixed_five_slot_outputs_use_the_same_frozen_metrics(self):
        outputs = [canonical_fixed_five_slot_target(label) for label in self.truths]
        rows, metrics = evaluate_outputs(
            self.truths,
            outputs,
            target_schema=FIXED_FIVE_SLOT_SCHEMA,
        )
        self.assertTrue(all(row["combined_label_exact"] for row in rows))
        self.assertEqual(metrics["combined_label_exact_match_rate"], 1.0)

    def test_type_and_severity_failures_are_not_hidden(self):
        outputs = [
            canonical_target(self.truths[0]),
            canonical_target(make_label(["noise"], {"noise": 2})),
            canonical_target(make_label(["noise", "rain"], {"noise": 4, "rain": 4})),
            "not-json",
        ]
        rows, metrics = evaluate_outputs(self.truths, outputs)
        self.assertEqual(metrics["valid_structured_output_rate"], 0.75)
        self.assertEqual(metrics["degradation_set_exact_match_rate"], 0.5)
        self.assertEqual(metrics["combined_label_exact_match_rate"], 0.25)
        self.assertLess(metrics["type_micro_f1"], 1.0)
        self.assertLess(metrics["severity_typed_accuracy"], 1.0)
        self.assertGreater(metrics["severity_typed_mae_with_missing_penalty"], 0.0)
        self.assertIsNotNone(rows[-1]["parse_error"])

    def test_metric_json_is_machine_serializable(self):
        rows, metrics = evaluate_outputs(
            self.truths, [canonical_target(label) for label in self.truths]
        )
        json.dumps({"rows": rows, "metrics": metrics}, sort_keys=True)

    def test_rejects_mismatched_cardinality(self):
        with self.assertRaises(ValueError):
            evaluate_outputs(self.truths, [])
        with self.assertRaises(ValueError):
            evaluate_outputs([], [])


if __name__ == "__main__":
    unittest.main()
