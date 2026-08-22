import csv
import hashlib
import json
import subprocess
import sys
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FoundIRLowlightGate1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        completed = subprocess.run(
            [sys.executable, "-m", "smoke.freeze_foundir_lowlight_gate1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise AssertionError(completed.stderr + completed.stdout)

    def test_result_passes_without_protected_access(self):
        result = json.loads((ROOT / "reports/foundir_lowlight_gate1_result_v1.json").read_text())
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["protected_foundir_status"], "NOT_ACCESSED_DURING_GATE_1")
        self.assertEqual(result["summary"]["records"], 674)
        self.assertEqual(result["summary"]["groups"], 232)

    def test_manifest_has_no_role_leakage(self):
        with (ROOT / "manifests/foundir_lowlight_development_contract_v1.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        role_by_group = defaultdict(set)
        role_by_content = defaultdict(set)
        for row in rows:
            role_by_group[(row["family"], row["group_id"])].add(row["role"])
            role_by_content[("input", row["input_sha256"])].add(row["role"])
            role_by_content[("target", row["target_sha256"])].add(row["role"])
            self.assertEqual(row["protected_exact_overlap"], "false")
            self.assertEqual(row["absolute_l1_l5_label_available"], "false")
        self.assertTrue(all(len(roles) == 1 for roles in role_by_group.values()))
        self.assertTrue(all(len(roles) == 1 for roles in role_by_content.values()))

    def test_sice_outer_selection_is_predeclared_hash_partition(self):
        config = json.loads((ROOT / "configs/foundir_lowlight_development_contract_v1.json").read_text())
        scenes = []
        with (ROOT / "manifests/lowlight_real_paired_development_v1.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] == "sice_reusable_validation":
                    scenes.append(row["scene_id"])
        scenes = sorted(set(scenes))
        selected = sorted(
            scenes,
            key=lambda scene: (
                hashlib.sha256(f"foundir-g1-sice-outer-v1:{scene}".encode()).hexdigest(),
                scene,
            ),
        )[:10]
        self.assertEqual(set(selected), set(config["families"]["sice"]["outer_scene_ids"]))

    def test_gate7_remains_blocked(self):
        ledger = json.loads((ROOT / "manifests/foundir_lowlight_experiment_ledger_v1.json").read_text())
        gate7 = next(entry for entry in ledger["entries"] if entry["id"] == "G7_PROTECTED_EVENT")
        self.assertEqual(gate7["status"], "blocked_pending_user_authorization")
        self.assertFalse(gate7["protected_test_access"])


if __name__ == "__main__":
    unittest.main()
