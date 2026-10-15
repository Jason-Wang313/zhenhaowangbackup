import unittest

from smoke.check_cure_gpu_profile_v1 import validate_profile


class CUREGPUProfileTest(unittest.TestCase):
    def test_h100_and_h200_preserve_global_batch(self) -> None:
        for model, memory in (("NVIDIA H100 80GB HBM3", 80), ("NVIDIA H200", 140)):
            profile = validate_profile([model] * 8, [memory * 1024**3] * 8)
            self.assertEqual(profile["global_batch"], {"embedder": 256, "baseline": 128, "cure": 24})

    def test_mixed_or_small_gpu_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            validate_profile(["NVIDIA H100"] * 7 + ["NVIDIA H200"], [80 * 1024**3] * 8)
        with self.assertRaises(RuntimeError):
            validate_profile(["NVIDIA H100"] * 8, [70 * 1024**3] * 8)


if __name__ == "__main__":
    unittest.main()
