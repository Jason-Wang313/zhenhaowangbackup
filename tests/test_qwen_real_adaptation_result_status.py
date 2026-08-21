from __future__ import annotations

import hashlib
import json
from pathlib import Path

from smoke.qwen_real_adaptation_result_status import classify


def write_result(
    tmp_path: Path, *, status: str, automatic: bool, checks: dict[str, bool]
) -> Path:
    output = tmp_path / "result"
    checkpoint = output / "checkpoints" / "step_000025"
    checkpoint.mkdir(parents=True, exist_ok=True)
    visual = checkpoint / "visual.safetensors"
    visual.write_bytes(b"visual")
    metadata = {
        "config_sha256": "c" * 64,
        "visual_sha256": hashlib.sha256(b"visual").hexdigest(),
        "step": 25,
    }
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "config_sha256": "c" * 64,
                "status": status,
                "automatic_gate_pass": automatic,
                "gate_checks": checks,
                "selected_checkpoint": "checkpoints/step_000025",
                "selected_checkpoint_step": 25,
                "selected_checkpoint_metadata": metadata,
            }
        )
    )
    return output


def test_verified_failure_maps_to_status_two(tmp_path: Path) -> None:
    result = write_result(
        tmp_path, status="failed", automatic=False, checks={"gate": False}
    )
    assert classify(result, "c" * 64) == 2


def test_verified_pass_maps_to_status_zero(tmp_path: Path) -> None:
    result = write_result(
        tmp_path, status="passed", automatic=True, checks={"gate": True}
    )
    assert classify(result, "c" * 64) == 0


def test_incomplete_or_inconsistent_output_remains_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    assert classify(missing, "c" * 64) == 1
    result = write_result(
        tmp_path, status="failed", automatic=False, checks={"gate": True}
    )
    assert classify(result, "c" * 64) == 1
    result = write_result(
        tmp_path, status="passed", automatic=True, checks={"gate": False}
    )
    assert classify(result, "c" * 64) == 1
    assert classify(result, "d" * 64) == 1
    (result / "checkpoints/step_000025/visual.safetensors").write_bytes(b"changed")
    assert classify(result, "c" * 64) == 1
