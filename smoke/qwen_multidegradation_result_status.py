#!/usr/bin/env python3
"""Normalize a completed Qwen validation result after torchrun exits.

Exit 0 means a verified scientific pass, exit 2 means a verified automatic
gate failure, and exit 1 means the run crashed or its artifacts are incomplete
or inconsistent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(
    workspace_root: Path,
    output_dir: Path,
    relative_path: object,
    expected_sha256: object,
) -> bool:
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        return False
    try:
        path = (workspace_root / relative_path).resolve()
        path.relative_to(output_dir.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file() and _sha256(path) == expected_sha256


def classify(
    workspace_root: Path,
    output_dir: Path,
    config_sha256: str,
    variant: str,
    split: str = "validation",
) -> int:
    if split not in {"validation", "test"}:
        return 1
    result_path = output_dir / (
        "validation_result.json" if split == "validation" else "fixed_test_result.json"
    )
    if not result_path.is_file():
        return 1
    try:
        result = json.loads(result_path.read_text())
    except (OSError, ValueError, TypeError):
        return 1
    checks = result.get("gate_checks") if isinstance(result, dict) else None
    training = result.get("training") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("config_sha256") != config_sha256
        or result.get("preflight_config_sha256") != config_sha256
        or result.get("variant") != variant
        or result.get("split") != split
        or not isinstance(checks, dict)
        or not checks
        or not all(isinstance(value, bool) for value in checks.values())
        or not isinstance(training, dict)
        or training.get("gradient_audit_pass") is not True
        or not _artifact(
            workspace_root,
            output_dir,
            result.get("adapter"),
            result.get("adapter_sha256"),
        )
        or not _artifact(
            workspace_root,
            output_dir,
            result.get("predictions"),
            result.get("predictions_sha256"),
        )
    ):
        return 1
    all_checks = all(checks.values())
    if result.get("status") == "passed" and result.get("automatic_gate_pass") is True:
        return 0 if all_checks else 1
    if result.get("status") == "failed" and result.get("automatic_gate_pass") is False:
        return 2 if not all_checks else 1
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    return classify(
        args.workspace_root.resolve(),
        args.output_dir.resolve(),
        args.config_sha256,
        args.variant,
        args.split,
    )


if __name__ == "__main__":
    raise SystemExit(main())
