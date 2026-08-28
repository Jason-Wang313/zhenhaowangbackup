from __future__ import annotations

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path

from lowlight_restoration.multidegradation_data import (
    ROOT,
    SPLIT_COMPOSITION_COUNTS,
    assign_source_splits,
    build_pilot_records,
    load_source_inventory,
    write_new_or_verify,
)
from lowlight_restoration.multidegradation_visibility_v2 import (
    build_candidate_record,
    load_visibility_anchors,
    ordered_source_pool,
    physical_order,
)
from lowlight_restoration.qwen_multidegradation import DEGRADATIONS
from smoke.build_qwen_multidegradation_visibility_v2 import (
    build_group_tasks,
    load_production_contract,
    validate_visibility_records,
)


class MultiDegradationVisibilityBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = (
            ROOT
            / "configs/qwen_multidegradation_visibility_v2_production_correction1.json"
        )
        cls.config = load_production_contract(cls.config_path)
        cls.inventory = load_source_inventory(ROOT / cls.config["source_manifest"])
        cls.assignments = assign_source_splits(
            cls.inventory,
            seed=int(cls.config["seed"]),
        )
        policy = cls.config["visibility_policy"]
        thresholds = load_visibility_anchors(
            ROOT / cls.config["visibility_anchor"],
            metric=str(policy["primary_metric"]),
            quantile=str(policy["anchor_quantile"]),
        )
        multiplier = float(policy["anchor_multiplier"])
        cls.thresholds = {
            key: value * multiplier for key, value in thresholds.items()
        }

    def test_production_contract_is_bound_to_actual_passing_pilot_config(self):
        pilot = json.loads((ROOT / self.config["passing_pilot"]).read_text())
        self.assertEqual(pilot["config"], self.config["passing_pilot_config"])
        self.assertEqual(
            pilot["config_sha256"],
            self.config["passing_pilot_config_sha256"],
        )

    def test_group_plan_has_exact_balanced_12000_slots(self):
        groups = build_group_tasks(
            self.assignments,
            config=self.config,
            thresholds=self.thresholds,
            contract_sha256="a" * 64,
        )
        self.assertEqual(
            sum(
                int(group["end_index"]) - int(group["start_index"])
                for group in groups
            ),
            12000,
        )
        observed = {
            (str(group["split"]), int(group["size"]))
            for group in groups
        }
        for split in ("train", "validation", "test"):
            for size in range(4):
                self.assertIn((split, size), observed)
                combinations = {
                    tuple(group["combo"])
                    for group in groups
                    if group["split"] == split and int(group["size"]) == size
                }
                expected_combinations = {()} if size == 0 else set(
                    itertools.combinations(DEGRADATIONS, size)
                )
                self.assertEqual(combinations, expected_combinations)
                slots = sum(
                    int(group["end_index"]) - int(group["start_index"])
                    for group in groups
                    if group["split"] == split and int(group["size"]) == size
                )
                self.assertEqual(slots, SPLIT_COMPOSITION_COUNTS[split][size])

    def test_rescue_starts_after_original_256_attempt_contract(self):
        combo = ("blur", "haze")
        sources = ordered_source_pool(
            self.assignments["validation"],
            seed=int(self.config["seed"]),
            split="validation",
            combo=combo,
        )
        policy = self.config["visibility_policy"]
        common = {
            "split": "validation",
            "composition_size": 2,
            "combination_index": 2,
            "within_combination_index": 2,
            "combo": combo,
            "ordered_sources": sources,
            "seed": int(self.config["seed"]),
            "policy_order": policy["physical_application_order"],
            "visibility_policy_name": policy["policy_name"],
            "data_root": self.config["data_root"],
            "composition_rescue": policy["composition_rescue"],
        }
        original = build_candidate_record(attempt=255, **common)
        rescued = build_candidate_record(attempt=256, **common)
        self.assertNotIn("composition_rescue", original)
        self.assertIn("composition_rescue", rescued)
        factors = rescued["composition_rescue"]["factors"]
        self.assertTrue(0.30 <= factors["haze_beta_scale"] <= 0.85)
        self.assertTrue(
            0.45 <= factors["haze_airlight_contrast_scale"] <= 0.90
        )

    def test_complete_synthetic_manifest_requires_physical_order_and_evidence(self):
        records = build_pilot_records(self.inventory, seed=int(self.config["seed"]))
        policy = self.config["visibility_policy"]
        for record in records:
            record["application_order"] = list(
                physical_order(
                    tuple(record["degradations"]),
                    policy["physical_application_order"],
                )
            )
            record["visibility_policy"] = policy["policy_name"]
            record["screening_attempt"] = 0
            record["visibility_evidence"] = [
                {
                    "degradation": name,
                    "level": record["levels"][name],
                    "position": position,
                    "full_sha256": "a" * 64,
                    "counterfactual_sha256": "b" * 64,
                    "primary_metric": policy["primary_metric"],
                    "visibility_mode": policy["visibility_mode"],
                    "anchor_observed_value": 1.0,
                    "minimum_anchor_value": 0.5,
                    "retention_ratio": 1.0,
                    "minimum_retention_ratio": policy[
                        "minimum_rmse_survival_ratio"
                    ],
                    "pass": True,
                }
                for position, name in enumerate(record["application_order"])
            ]
        summary = validate_visibility_records(records, policy=policy)
        self.assertEqual(summary["records"], 12000)
        self.assertEqual(summary["visible_active_components"], 21600)
        self.assertTrue(summary["combination_and_severity_balanced"])

        broken = copy.deepcopy(records)
        triple = next(record for record in broken if record["composition_size"] == 3)
        triple["visibility_evidence"][0]["pass"] = False
        with self.assertRaisesRegex(RuntimeError, "invisible active component"):
            validate_visibility_records(broken, policy=policy)

    def test_immutable_writer_refuses_different_existing_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "artifact.json"
            first_sha = write_new_or_verify(path, b"first\n")
            self.assertEqual(first_sha, write_new_or_verify(path, b"first\n"))
            with self.assertRaises(FileExistsError):
                write_new_or_verify(path, b"different\n")


if __name__ == "__main__":
    unittest.main()
