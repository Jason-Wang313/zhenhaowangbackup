"""Fail-closed authorization for bounded and full Lowlight baseline runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import ROOT, TrainConfig

OVERFIT_STATUS = "approved_for_16_pair_gpu_smoke_only"
FULL_STATUS = "approved_after_overfit16_guidance1_pass"
OVERFIT_CONFIG_SHA256 = "62740e0e0f26c4001057adf9d955259b7c8d89553245a2c1565a22599e7d0804"
FULL_CONFIG_SHA256 = "c71235efe88e8f3bc98ca863350af372a78aa35f1a2cc065a1867c5bb6bfde18"
GATE_PATH = ROOT / "configs/lowlight_overfit16_gate.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _workspace_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} must be a non-empty workspace path")
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes workspace: {value}") from exc
    return path


def _require_exact(mapping: dict[str, object], key: str, expected: object, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise RuntimeError(f"{label}.{key} must equal {expected!r}")


def _validate_full_static_config(config: TrainConfig) -> None:
    if config.sha256 != FULL_CONFIG_SHA256:
        raise RuntimeError("the approved full baseline config hash has changed")
    _require_exact(config.raw, "run_name", "flux2_lowlight_baseline_full", label="config")
    data = config.section("data")
    _require_exact(data, "manifest", "manifests/low_light_pilot_10000.csv", label="data")
    _require_exact(data, "split", "train", label="data")
    _require_exact(data, "expected_records", 8000, label="data")
    _require_exact(data, "expected_unique_sources", 1600, label="data")
    inference = config.section("inference")
    _require_exact(inference, "guidance", 1.0, label="inference")
    _require_exact(inference, "num_samples", 1000, label="inference")
    runtime = config.section("runtime")
    _require_exact(runtime, "output_dir", "checkpoints/lowlight_baseline_full", label="runtime")


def _load_passing_gate(*, require_artifacts: bool) -> dict[str, object]:
    if not GATE_PATH.is_file():
        raise RuntimeError(f"missing overfit authorization gate: {GATE_PATH}")
    try:
        gate = json.loads(GATE_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read overfit authorization gate: {GATE_PATH}") from exc
    if not isinstance(gate, dict):
        raise RuntimeError("overfit authorization gate must be a JSON object")
    expected = {
        "schema_version": 1,
        "gate": "flux2_lowlight_baseline_overfit16",
        "status": "passed",
        "approved_next_scope": "prepare_full_8000_pair_baseline",
        "flux_revision": "50fe5162777813d869182b139e83b10743caef15",
        "config_sha256": OVERFIT_CONFIG_SHA256,
        "guidance": 1.0,
        "automatic_gate_pass": True,
        "visual_identity_gate_pass": True,
    }
    for key, value in expected.items():
        _require_exact(gate, key, value, label="overfit_gate")
    checkpoint = _workspace_path(gate.get("checkpoint"), label="overfit_gate.checkpoint")
    evaluation = _workspace_path(gate.get("evaluation"), label="overfit_gate.evaluation")
    for path, hash_key in (
        (checkpoint, "checkpoint_sha256"),
        (evaluation, "evaluation_sha256"),
    ):
        expected_hash = gate.get(hash_key)
        if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
            raise RuntimeError(f"overfit_gate.{hash_key} must be a SHA-256 string")
        if require_artifacts and (not path.is_file() or sha256(path) != expected_hash):
            raise RuntimeError(f"overfit gate artifact missing or hash mismatch: {path}")
    return gate


def validate_run_authorization(config: TrainConfig, *, require_artifacts: bool = True) -> str:
    """Return the authorized scope or raise without mutating any run artifact."""

    status = config.raw.get("status")
    if status == OVERFIT_STATUS:
        if config.sha256 != OVERFIT_CONFIG_SHA256:
            raise RuntimeError("the approved overfit config hash has changed")
        return "overfit16"
    if status == FULL_STATUS:
        _validate_full_static_config(config)
        _load_passing_gate(require_artifacts=require_artifacts)
        return "full_baseline"
    raise RuntimeError(f"run status is not authorized: {status!r}")
