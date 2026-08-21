from __future__ import annotations

import unittest

from lowlight_restoration.exposure_ordering import (
    build_exposure_stacks,
    completed_exposure_result_status,
    evaluate_exposure_ordering,
    expected_level_from_nll,
    exposure_pairs,
    load_frozen_exposure_stacks,
    spearman_correlation,
)


def candidate(
    *,
    component: str,
    filename: str,
    sha256: str,
    mean_luma: float,
    subset: str = "MEF",
) -> dict[str, str]:
    return {
        "path": f"data/{subset}/{filename}",
        "subset": subset,
        "filename": filename,
        "sha256": sha256,
        "bytes": "100",
        "width": "64",
        "height": "32",
        "mode": "RGB",
        "mean_luma": str(mean_luma),
        "component_id": component,
    }


class ExposureOrderingTests(unittest.TestCase):
    def test_stack_selection_deduplicates_and_orders(self) -> None:
        rows = [
            candidate(component="stack", filename="bright.png", sha256="b" * 64, mean_luma=80),
            candidate(component="stack", filename="dark.png", sha256="a" * 64, mean_luma=20),
            candidate(
                component="stack",
                filename="dark_copy.png",
                sha256="a" * 64,
                mean_luma=20,
                subset="NPE",
            ),
            candidate(
                component="singleton",
                filename="singleton.png",
                sha256="c" * 64,
                mean_luma=10,
            ),
            candidate(
                component="excluded",
                filename="excluded_one.png",
                sha256="d" * 64,
                mean_luma=10,
            ),
            candidate(
                component="excluded",
                filename="excluded_two.png",
                sha256="e" * 64,
                mean_luma=20,
            ),
        ]
        stacks = build_exposure_stacks(rows, excluded_components={"excluded"})
        self.assertEqual(list(stacks), ["stack"])
        stack = stacks["stack"]
        self.assertEqual([image.filename for image in stack], ["dark.png", "bright.png"])
        self.assertEqual([image.darkness_rank for image in stack], [2, 1])
        self.assertAlmostEqual(stack[0].ev_darker_than_brightest, 2.0)
        self.assertEqual(len(exposure_pairs(stack, minimum_ev_gap=1.0)), 1)
        self.assertEqual(len(exposure_pairs(stack, minimum_ev_gap=3.0)), 0)

    def test_ambiguous_luma_order_is_rejected(self) -> None:
        rows = [
            candidate(component="stack", filename="one.png", sha256="a" * 64, mean_luma=20),
            candidate(component="stack", filename="two.png", sha256="b" * 64, mean_luma=20),
        ]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            build_exposure_stacks(rows)

    def test_frozen_manifest_round_trip_validation(self) -> None:
        rows = [
            {
                "stack_id": "stack",
                "image_id": "dark",
                "path": "data/dark.png",
                "sha256": "a" * 64,
                "subset": "MEF",
                "filename": "dark.png",
                "bytes": "100",
                "width": "64",
                "height": "32",
                "mode": "RGB",
                "mean_luma": "20",
                "order_dark_to_bright": "0",
                "darkness_rank": "2",
                "ev_darker_than_brightest": "2",
            },
            {
                "stack_id": "stack",
                "image_id": "bright",
                "path": "data/bright.png",
                "sha256": "b" * 64,
                "subset": "MEF",
                "filename": "bright.png",
                "bytes": "100",
                "width": "64",
                "height": "32",
                "mode": "RGB",
                "mean_luma": "80",
                "order_dark_to_bright": "1",
                "darkness_rank": "1",
                "ev_darker_than_brightest": "0",
            },
        ]
        stacks, identities = load_frozen_exposure_stacks(rows)
        self.assertEqual(len(stacks["stack"]), 2)
        self.assertEqual(identities, {"dark": "a" * 64, "bright": "b" * 64})
        rows[1]["ev_darker_than_brightest"] = "1"
        with self.assertRaisesRegex(ValueError, "EV differs"):
            load_frozen_exposure_stacks(rows)

    def test_soft_level_and_rank_metrics(self) -> None:
        expected, probabilities, entropy = expected_level_from_nll(
            {"1": 4.0, "2": 3.0, "3": 2.0, "4": 1.0, "5": 0.0}
        )
        self.assertGreater(expected, 4.0)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(entropy, 0.0)
        self.assertAlmostEqual(spearman_correlation([3, 2, 1], [5, 3, 1]), 1.0)
        self.assertAlmostEqual(spearman_correlation([3, 2, 1], [1, 3, 5]), -1.0)

    def test_evaluation_detects_monotonic_and_inverted_predictions(self) -> None:
        rows = [
            candidate(component="stack", filename="dark.png", sha256="a" * 64, mean_luma=10),
            candidate(component="stack", filename="middle.png", sha256="b" * 64, mean_luma=20),
            candidate(component="stack", filename="bright.png", sha256="c" * 64, mean_luma=40),
        ]
        stacks = build_exposure_stacks(rows)

        def nll(center: int) -> dict[str, float]:
            return {str(level): float((level - center) ** 2) for level in range(1, 6)}

        good = evaluate_exposure_ordering(
            stacks,
            {"a" * 64: nll(5), "b" * 64: nll(3), "c" * 64: nll(1)},
            minimum_ev_gaps=(1.0,),
        )
        primary = good["pairwise"]["minimum_ev_1"]
        self.assertEqual(primary["continuous_concordance_with_half_credit_for_ties"], 1.0)
        self.assertEqual(good["median_stack_spearman"], 1.0)

        bad = evaluate_exposure_ordering(
            stacks,
            {"a" * 64: nll(1), "b" * 64: nll(3), "c" * 64: nll(5)},
            minimum_ev_gaps=(1.0,),
        )
        primary = bad["pairwise"]["minimum_ev_1"]
        self.assertEqual(primary["continuous_concordance_with_half_credit_for_ties"], 0.0)
        self.assertEqual(bad["median_stack_spearman"], -1.0)

    def test_completed_result_status_distinguishes_failure_from_error(self) -> None:
        base = {
            "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "selected_visual_sha256": "c" * 64,
            "official_foundir_used": False,
            "absolute_real_l1_l5_accuracy_claimed": False,
            "downstream_restoration_gain_claimed": False,
            "relative_exposure_evidence_only": True,
            "gate_checks": {"one": True, "two": True},
            "automatic_gate_pass": True,
            "status": "passed",
        }
        arguments = {
            "config_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
        }
        self.assertEqual(completed_exposure_result_status(base, **arguments), 0)
        base["gate_checks"]["two"] = False
        base["automatic_gate_pass"] = False
        base["status"] = "failed"
        self.assertEqual(completed_exposure_result_status(base, **arguments), 2)
        base["status"] = "passed"
        with self.assertRaisesRegex(ValueError, "status"):
            completed_exposure_result_status(base, **arguments)


if __name__ == "__main__":
    unittest.main()
