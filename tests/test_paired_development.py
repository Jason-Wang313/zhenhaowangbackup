from __future__ import annotations

import unittest

import numpy as np

from lowlight_restoration.paired_development import (
    DevelopmentSource,
    clustered_metric_summary,
    fit_oracle_gamma_color,
    gradient_alignment_diagnostic,
    peak_signal_noise_ratio,
    rgb_structural_similarity,
    select_paired_development_rows,
)


def parent_row(pair_id: str, *, split: str, level: str, subset: str = "usable") -> dict[str, str]:
    return {
        "pair_id": pair_id,
        "source_id": f"scene_{pair_id}",
        "clean_path": f"data/{pair_id}_target.png",
        "degraded_path": f"data/{pair_id}_input.png",
        "project_level": level,
        "split": split,
        "clean_sha256": (pair_id[0] if pair_id else "a") * 64,
        "degraded_sha256": (pair_id[-1] if pair_id else "b") * 64,
        "dataset": "dataset",
        "subset": subset,
    }


class PairedDevelopmentTests(unittest.TestCase):
    def test_selection_excludes_clean_and_test_records(self) -> None:
        rows = (
            parent_row("ab", split="validation", level="L5"),
            parent_row("cd", split="validation", level="L1"),
            parent_row("ef", split="test", level="L5"),
        )
        selected = select_paired_development_rows(
            (
                DevelopmentSource(
                    source_name="source",
                    rows=rows,
                    required_dataset="dataset",
                    required_split="validation",
                    required_subset="usable",
                    target_kind="paired_reference",
                ),
            )
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["development_id"], "source:ab")
        self.assertEqual(selected[0]["proxy_project_level"], "L5")

    def test_oracle_global_photometric_fit_improves_known_transform(self) -> None:
        grid = np.linspace(0.01, 0.8, 32 * 24 * 3, dtype=np.float32).reshape(24, 32, 3)
        target = np.empty_like(grid)
        target[..., 0] = np.clip(grid[..., 0] ** 0.5 * 0.9 + 0.04, 0, 1)
        target[..., 1] = np.clip(grid[..., 1] ** 0.5 * 1.0 + 0.02, 0, 1)
        target[..., 2] = np.clip(grid[..., 2] ** 0.5 * 1.1, 0, 1)
        initial = peak_signal_noise_ratio(target, grid)
        fit = fit_oracle_gamma_color(
            grid,
            target,
            gamma_candidates=(0.4, 0.5, 0.6, 1.0),
            maximum_fit_pixels=1024,
        )
        restored = peak_signal_noise_ratio(target, fit["prediction"])
        self.assertEqual(fit["gamma"], 0.5)
        self.assertGreater(restored, initial + 20.0)

    def test_invalid_gamma_grid_is_rejected(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "gamma"):
            fit_oracle_gamma_color(image, image, gamma_candidates=(0.5, 0.5))

    def test_identity_metrics_and_alignment(self) -> None:
        generator = np.random.default_rng(7)
        image = generator.random((48, 64, 3), dtype=np.float32)
        self.assertEqual(peak_signal_noise_ratio(image, image), 100.0)
        self.assertAlmostEqual(rgb_structural_similarity(image, image), 1.0)
        diagnostic = gradient_alignment_diagnostic(image, image)
        self.assertLess(diagnostic["estimated_shift_magnitude_pixels"], 0.05)
        self.assertGreater(diagnostic["gradient_correlation_shifted"], 0.999)

    def test_clustered_summary_distinguishes_record_and_scene_weighting(self) -> None:
        rows = (
            {"dataset": "a", "scene_id": "a:1", "score": 1.0},
            {"dataset": "a", "scene_id": "a:1", "score": 3.0},
            {"dataset": "a", "scene_id": "a:2", "score": 10.0},
        )
        summary = clustered_metric_summary(
            rows,
            fields=("score",),
            bootstrap_seed=17,
            bootstrap_resamples=100,
        )
        point = summary["point_estimates"]
        self.assertAlmostEqual(point["record_weighted"]["score"], 14.0 / 3.0)
        self.assertAlmostEqual(point["scene_balanced"]["score"], 6.0)
        interval = summary["confidence_intervals"]["scene_balanced"]["score"]
        self.assertLessEqual(interval["lower"], interval["estimate"])
        self.assertGreaterEqual(interval["upper"], interval["estimate"])


if __name__ == "__main__":
    unittest.main()
