import json
import unittest
from copy import deepcopy
from pathlib import Path

from smoke.qwen_multidegradation_lora_preflight import (
    ROOT,
    load_config,
    validate_contract,
    validate_model_inventory,
)


class QwenMultiDegradationLoraPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "configs/qwen_multidegradation_lora_v1.json"
        cls.config, cls.sha = load_config(cls.path)

    def test_static_contract_and_inventory(self):
        validate_contract(self.config)
        result = validate_model_inventory(self.config, full_hashes=False)
        self.assertEqual(result["revision"], "cc594898137f460bfe9f0759e9844b3ce807cfb5")

    def test_rejects_device_or_curriculum_drift(self):
        changed = deepcopy(self.config)
        changed["runtime"]["required_visible_devices"] = "0,1"
        with self.assertRaises(ValueError):
            validate_contract(changed)
        changed = deepcopy(self.config)
        changed["data"]["curriculum"][2]["maximum_composition_size"] = 2
        with self.assertRaises(ValueError):
            validate_contract(changed)

    def test_config_is_machine_serializable(self):
        json.dumps(self.config, sort_keys=True)
        self.assertEqual(len(self.sha), 64)


if __name__ == "__main__":
    unittest.main()
