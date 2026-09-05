from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_test_launcher_is_one_time_and_supports_two_or_eight_gpus() -> None:
    runner = (
        ROOT / "smoke/run_qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.sh"
    ).read_text()
    launcher = (
        ROOT / "smoke/launch_qwen_multidegradation_100k_rank64_per_type_v2_fixed_test.sh"
    ).read_text()
    verifier = (
        ROOT / "smoke/verify_qwen_multidegradation_100k_v2_fixed_test.py"
    ).read_text()
    for text in (runner, launcher):
        assert "0,1)" in text or "0,1|" in text
        assert "0,1,2,3,4,5,6,7" in text
    assert "--test-only" in runner
    assert "--split test" in runner
    assert "one_time_synthetic_test_no_foundir" in runner
    assert "fixed synthetic test was already accessed" in verifier
    assert "selected_method\") != \"per_type\"" in verifier
    assert "remain-on-exit" in launcher
    assert "nohup env CUDA_VISIBLE_DEVICES" in launcher
