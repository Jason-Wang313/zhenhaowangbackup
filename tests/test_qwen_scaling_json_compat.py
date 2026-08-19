from __future__ import annotations

import json
import unittest

from smoke.validate_qwen_lowlight_supervised_scaling_json_compat import (
    canonical_base_preflight_json_compatible,
)


class QwenScalingJsonCompatibilityTests(unittest.TestCase):
    def test_integer_level_keys_match_json_reload(self) -> None:
        canonical = canonical_base_preflight_json_compatible(
            {
                "selected_per_level": {1: 8, 2: 8, 3: 8, 4: 8, 5: 8},
                "cuda_visible_in_preflight": True,
            }
        )
        self.assertEqual(
            canonical["selected_per_level"],
            {"1": 8, "2": 8, "3": 8, "4": 8, "5": 8},
        )
        self.assertEqual(canonical, json.loads(json.dumps(canonical)))
        self.assertNotIn("cuda_visible_in_preflight", canonical)


if __name__ == "__main__":
    unittest.main()
