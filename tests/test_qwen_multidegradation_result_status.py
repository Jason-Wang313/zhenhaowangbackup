import hashlib
import json
from pathlib import Path

from smoke.qwen_multidegradation_result_status import classify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_result(root: Path, *, passed: bool) -> Path:
    output = root / "checkpoints/run/merger_trainable"
    checkpoint = output / "stage"
    checkpoint.mkdir(parents=True)
    adapter = checkpoint / "adapter.safetensors"
    predictions = output / "validation_predictions.json"
    adapter.write_bytes(b"adapter")
    predictions.write_text("{}\n")
    payload = {
        "status": "passed" if passed else "failed",
        "automatic_gate_pass": passed,
        "config_sha256": "c" * 64,
        "preflight_config_sha256": "c" * 64,
        "variant": "merger_trainable",
        "split": "validation",
        "gate_checks": {"one": True, "two": passed},
        "training": {"gradient_audit_pass": True},
        "adapter": adapter.relative_to(root).as_posix(),
        "adapter_sha256": _sha256(adapter),
        "predictions": predictions.relative_to(root).as_posix(),
        "predictions_sha256": _sha256(predictions),
    }
    (output / "validation_result.json").write_text(json.dumps(payload))
    return output


def test_verified_failure_maps_to_two(tmp_path: Path) -> None:
    output = _write_result(tmp_path, passed=False)
    assert classify(tmp_path, output, "c" * 64, "merger_trainable") == 2


def test_verified_pass_maps_to_zero(tmp_path: Path) -> None:
    output = _write_result(tmp_path, passed=True)
    assert classify(tmp_path, output, "c" * 64, "merger_trainable") == 0


def test_missing_or_inconsistent_results_are_true_errors(tmp_path: Path) -> None:
    output = tmp_path / "missing"
    assert classify(tmp_path, output, "c" * 64, "merger_trainable") == 1
    output = _write_result(tmp_path, passed=False)
    assert classify(tmp_path, output, "d" * 64, "merger_trainable") == 1
    (output / "stage/adapter.safetensors").write_bytes(b"changed")
    assert classify(tmp_path, output, "c" * 64, "merger_trainable") == 1


def test_fixed_test_result_is_verified_separately(tmp_path: Path) -> None:
    output = _write_result(tmp_path, passed=True)
    validation = output / "validation_result.json"
    result = json.loads(validation.read_text())
    predictions = output / "test_predictions.json"
    predictions.write_text("{}\n")
    result.update(
        split="test",
        predictions=predictions.relative_to(tmp_path).as_posix(),
        predictions_sha256=_sha256(predictions),
    )
    (output / "fixed_test_result.json").write_text(json.dumps(result))
    assert classify(tmp_path, output, "c" * 64, "merger_trainable", "test") == 0
    assert classify(tmp_path, output, "c" * 64, "merger_trainable") == 0
