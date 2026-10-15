#!/usr/bin/env python3
"""Select one FLUX checkpoint from development metrics without sealed-test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    resolve_workspace_path,
    select_checkpoint,
    worst_normalized_margin,
)


def select(config_path: Path, result_paths: list[Path]) -> dict[str, object]:
    config_sha = file_sha256(config_path)
    candidates = []
    for path in result_paths:
        payload = json.loads(path.read_text())
        if (
            payload.get("status") != "development_evaluation_complete"
            or payload.get("config_sha256") != config_sha
            or payload.get("split") != "validation"
            or payload.get("arm") != "qwen"
            or payload.get("protected_foundir_test_accessed") is not False
        ):
            raise RuntimeError(f"invalid development candidate: {path}")
        checkpoint = ROOT / payload["checkpoint"]
        if file_sha256(checkpoint) != payload["checkpoint_sha256"]:
            raise RuntimeError(f"candidate checkpoint changed: {checkpoint}")
        step = int(checkpoint.stem.split("_", 1)[1])
        candidates.append(
            {
                "step": step,
                "checkpoint": payload["checkpoint"],
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "task_metrics": payload["summary"]["task_metrics"],
                "worst_normalized_margin": worst_normalized_margin(
                    payload["summary"]["task_metrics"]
                ),
                "result": path.relative_to(ROOT).as_posix(),
                "result_sha256": file_sha256(path),
            }
        )
    winner = select_checkpoint(candidates)
    return {
        "schema_version": 2,
        "status": "frozen_development_selected_checkpoint",
        "metric_contract": "sow_isolated_categories_v1",
        "config_sha256": config_sha,
        "selection_rule": "maximize_worst_five_task_normalized_psnr_lpips_margin_then_earlier_step",
        "candidates": candidates,
        "selected_step": winner["step"],
        "checkpoint": winner["checkpoint"],
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_development_result": winner["result"],
        "selected_worst_normalized_margin": winner["worst_normalized_margin"],
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.config = resolve_workspace_path(ROOT, args.config)
    args.result = [resolve_workspace_path(ROOT, path) for path in args.result]
    args.output = resolve_workspace_path(ROOT, args.output)
    payload = select(args.config, args.result)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != content:
        raise RuntimeError("refusing to overwrite differing checkpoint selection")
    if not args.output.exists():
        with args.output.open("xb") as handle:
            handle.write(content)
    print("FOUNDIR_FLUX_CHECKPOINT_SELECTION_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
