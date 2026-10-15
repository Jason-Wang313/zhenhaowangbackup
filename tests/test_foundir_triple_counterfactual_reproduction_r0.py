from __future__ import annotations

import copy
import unittest

from smoke.build_foundir_triple_counterfactual_reproduction_r0_v1 import (
    build_lattice,
    ordered_subsets,
    validate_lattices,
)


def fixture_parent(role: str, names: tuple[str, ...], source_id: str) -> dict:
    return {
        "record_id": f"record-{source_id}",
        "protocol_role": role,
        "source_id": source_id,
        "source_path": "unused.png",
        "source_sha256": "0" * 64,
        "crop": {"x": 0, "y": 0, "width": 16, "height": 16},
        "non_target_nuisance": {"family": "none", "seed": 1},
        "composition_size": len(names),
        "degradations": list(names),
        "levels": {name: index + 2 for index, name in enumerate(names)},
        "application_order": list(names),
        "operators": {name: {"operator": name, "seed": index + 10} for index, name in enumerate(names)},
        "agenticir_revision": "fixture",
    }


class CounterfactualReproductionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "data": {
                "verify_source_sha256": False,
                "required_heldout_triple_combinations": ["blur+noise+low_light"],
            },
            "protocol": {
                "gradient_eligible_roles": ["train"],
                "checkpoint_selection_roles": ["development"],
                "maximum_gradient_composition_size": 2,
                "maximum_selection_composition_size": 2,
                "heldout_composition_size": 3,
            },
        }

    def lattices(self):
        return [
            build_lattice(fixture_parent("train", ("blur", "noise"), "train-source"), self.config),
            build_lattice(fixture_parent("development", ("noise", "low_light"), "dev-source"), self.config),
            build_lattice(
                fixture_parent("heldout_triple", ("blur", "noise", "low_light"), "test-source"),
                self.config,
            ),
        ]

    def test_ordered_subsets_preserve_parent_order(self):
        self.assertEqual(
            ordered_subsets(("blur", "rain", "noise")),
            [(), ("blur",), ("rain",), ("noise",), ("blur", "rain"),
             ("blur", "noise"), ("rain", "noise"), ("blur", "rain", "noise")],
        )

    def test_valid_protocol_passes_and_triple_is_fail_closed(self):
        lattices = self.lattices()
        audit = validate_lattices(self.config, lattices)
        self.assertEqual(audit["triple_gradient_leakage"], 0)
        self.assertEqual(audit["triple_selection_leakage"], 0)
        triple = lattices[-1]
        self.assertFalse(triple["gradient_eligible"])
        self.assertFalse(triple["checkpoint_selection_eligible"])
        self.assertEqual(len(triple["nodes"]), 8)
        self.assertEqual(len(triple["counterfactual_pairs"]), 27)

    def test_triple_gradient_leak_is_rejected(self):
        lattices = self.lattices()
        lattices[-1]["gradient_eligible"] = True
        with self.assertRaisesRegex(ValueError, "triple leaked"):
            validate_lattices(self.config, lattices)

    def test_cross_role_source_overlap_is_rejected(self):
        lattices = self.lattices()
        lattices[-1]["source_id"] = lattices[0]["source_id"]
        with self.assertRaisesRegex(ValueError, "source overlap"):
            validate_lattices(self.config, lattices)

    def test_operator_parameter_mutation_is_rejected(self):
        lattices = self.lattices()
        corrupted = copy.deepcopy(lattices)
        node = next(node for node in corrupted[0]["nodes"] if node["node_id"] == "blur")
        node["operators"]["blur"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "changed operator parameters"):
            validate_lattices(self.config, corrupted)

    def test_counterfactual_target_error_is_rejected(self):
        lattices = self.lattices()
        lattices[0]["counterfactual_pairs"][0]["target_node"] = "blur"
        with self.assertRaisesRegex(ValueError, "pair algebra"):
            validate_lattices(self.config, lattices)


if __name__ == "__main__":
    unittest.main()
