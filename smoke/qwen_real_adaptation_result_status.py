#!/usr/bin/env python3
"""Classify a completed real-adaptation summary for the shell launcher.

Exit codes are deliberately small and stable: 0 is a verified pass, 2 is a
verified automatic-gate failure, and 1 means that the result is missing or
inconsistent (including a crash that did not produce a valid result).
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


def classify(output_dir: Path, config_sha256: str) -> int:
    summary_path = output_dir / "training_summary.json"
    if not summary_path.is_file():
        return 1
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, TypeError):
        return 1
    if not isinstance(summary, dict) or summary.get("config_sha256") != config_sha256:
        return 1
    checks = summary.get("gate_checks")
    metadata = summary.get("selected_checkpoint_metadata")
    selected = summary.get("selected_checkpoint")
    if not isinstance(checks, dict) or not checks or not all(
        isinstance(value, bool) for value in checks.values()
    ):
        return 1
    if not isinstance(metadata, dict) or not isinstance(selected, str):
        return 1
    if metadata.get("config_sha256") != config_sha256:
        return 1
    if summary.get("selected_checkpoint_step") != metadata.get("step"):
        return 1
    checkpoint = (output_dir / selected).resolve()
    try:
        checkpoint.relative_to(output_dir.resolve())
    except ValueError:
        return 1
    metadata_path = checkpoint / "metadata.json"
    visual_path = checkpoint / "visual.safetensors"
    if not metadata_path.is_file() or not visual_path.is_file():
        return 1
    try:
        if json.loads(metadata_path.read_text()) != metadata:
            return 1
        visual_sha = metadata.get("visual_sha256")
        if not isinstance(visual_sha, str) or _sha256(visual_path) != visual_sha:
            return 1
    except (OSError, ValueError, TypeError):
        return 1

    all_checks = all(checks.values())
    if (
        summary.get("status") == "passed"
        and summary.get("automatic_gate_pass") is True
    ):
        return 0 if all_checks else 1
    if (
        summary.get("status") == "failed"
        and summary.get("automatic_gate_pass") is False
    ):
        return 2 if not all_checks else 1
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    return classify(args.output_dir, args.config_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
