import unittest
from pathlib import Path

from smoke.run_foundir_lowlight_gate2_historical_flux import read_config


ROOT = Path(__file__).resolve().parents[1]


class Gate2HistoricalFluxTests(unittest.TestCase):
    def test_contract_is_bounded_historical_only(self):
        config = read_config(ROOT / "configs/foundir_lowlight_gate2_historical_flux_v1.json")
        self.assertFalse(config["protected_foundir_pixels_allowed"])
        self.assertEqual(config["inference"]["num_steps"], 50)
        self.assertEqual(config["runtime"]["device_count"], 1)
        self.assertIn("historical", config["interpretation"])


if __name__ == "__main__": unittest.main()
