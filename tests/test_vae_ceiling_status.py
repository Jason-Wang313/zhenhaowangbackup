from __future__ import annotations

import copy
import unittest

from smoke.lowlight_vae_ceiling_status import validate_core


def passing_summary() -> dict[str, object]:
    return {
        "status": "completed_pixel_frequency_bypass_required",
        "config_sha256": "a" * 64,
        "official_foundir_inference_performed": False,
        "official_foundir_images_opened": False,
        "records": 174,
        "unique_targets": 130,
        "summary": {"records": 174, "scenes": 82},
        "automatic_decision": {"pixel_frequency_bypass_required": True},
        "device_inventory": [{"rank": 0}, {"rank": 1}],
    }


class VaeCeilingStatusTests(unittest.TestCase):
    def test_valid_bypass_result_passes(self) -> None:
        validate_core(
            passing_summary(),
            config_sha256="a" * 64,
            expected_records=174,
            expected_scenes=82,
            expected_unique_targets=130,
        )

    def test_status_decision_contradiction_fails(self) -> None:
        summary = copy.deepcopy(passing_summary())
        summary["automatic_decision"]["pixel_frequency_bypass_required"] = False
        with self.assertRaisesRegex(RuntimeError, "contradicts"):
            validate_core(
                summary,
                config_sha256="a" * 64,
                expected_records=174,
                expected_scenes=82,
                expected_unique_targets=130,
            )

    def test_foundir_access_claim_fails(self) -> None:
        summary = copy.deepcopy(passing_summary())
        summary["official_foundir_images_opened"] = True
        with self.assertRaisesRegex(RuntimeError, "FoundIR"):
            validate_core(
                summary,
                config_sha256="a" * 64,
                expected_records=174,
                expected_scenes=82,
                expected_unique_targets=130,
            )


if __name__ == "__main__":
    unittest.main()
