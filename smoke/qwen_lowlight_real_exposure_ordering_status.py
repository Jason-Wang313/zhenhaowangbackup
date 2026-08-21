#!/usr/bin/env python3
"""Normalize a completed exposure-ordering gate to shell status 0 or 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.exposure_ordering import (  # noqa: E402
    completed_exposure_result_status,
)
from smoke.qwen_lowlight_supervised_preflight import sha256, workspace_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    summary = json.loads(args.summary.read_text())
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise RuntimeError("config and summary must be JSON objects")
    data = config.get("data")
    model = config.get("model")
    runtime = config.get("runtime")
    if not isinstance(data, dict) or not isinstance(model, dict) or not isinstance(runtime, dict):
        raise RuntimeError("exposure-ordering config sections are missing")
    expected_summary = workspace_path(runtime["output_dir"]) / "result_summary.json"
    if args.summary.resolve() != expected_summary.resolve():
        raise RuntimeError("summary path does not match the frozen output directory")
    status = completed_exposure_result_status(
        summary,
        config_sha256=sha256(args.config),
        manifest_sha256=str(data["manifest_sha256"]),
        checkpoint_sha256=str(model["selected_visual_sha256"]),
    )
    print(
        "QWEN_REAL_EXPOSURE_ORDERING_STATUS "
        f"scientific_status={summary['status']} shell_status={status}"
    )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
