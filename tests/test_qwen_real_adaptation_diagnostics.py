from __future__ import annotations

from pathlib import Path

import pytest

from smoke.diagnose_qwen_real_adaptation_checkpoints import (
    ROOT,
    validated_output_dir,
    validated_steps,
)


def test_diagnostic_steps_are_frozen_candidates_only() -> None:
    allowed = [25, 50, 100, 150, 200]
    assert validated_steps([25, 50, 25], allowed) == [25, 50]
    with pytest.raises(ValueError, match="subset"):
        validated_steps([0, 25], allowed)


def test_diagnostic_output_protects_historical_result(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="historical"):
        validated_output_dir(
            ROOT / "checkpoints/qwen_lowlight_real_adaptation_step600/diagnostic"
        )
    existing = ROOT / "checkpoints"
    with pytest.raises(RuntimeError, match="already exists"):
        validated_output_dir(existing)


def test_diagnostic_output_must_stay_in_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workspace"):
        validated_output_dir(tmp_path / "outside")
