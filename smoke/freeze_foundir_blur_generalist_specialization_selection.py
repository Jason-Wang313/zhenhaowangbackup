#!/usr/bin/env python3
"""Freeze a Blur-specialization checkpoint using development metrics only."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import resolve_workspace_path  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import write_once  # noqa: E402
from smoke.train_foundir_blur_generalist_specialization_ddp import selection_score  # noqa: E402


def verify_candidate(
    path: Path,
    *,
    config: Mapping[str, object],
    config_sha: str,
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    checkpoint = resolve_workspace_path(ROOT, payload.get("checkpoint", ""))
    step = int(payload.get("step", -1))
    expected_path = checkpoint.parent / f"development_step_{step:06d}.json"
    metrics = payload.get("blur", {}).get("specialized", {})
    if (
        path != expected_path
        or payload.get("status") != "pretrained_blur_specialization_development_complete"
        or payload.get("schema_version") != 1
        or payload.get("config_sha256") != config_sha
        or payload.get("split") != "validation"
        or payload.get("sealed_internal_test_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
        or step not in {int(value) for value in config["training"]["checkpoint_steps"]}
        or payload.get("checkpoint_sha256") != file_sha256(checkpoint)
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics.get(key), bool)
            and math.isfinite(float(metrics[key]))
            for key in ("psnr", "lpips")
        )
    ):
        raise RuntimeError(f"invalid Blur specialization development candidate: {path}")
    return payload


def freeze_selection(
    *,
    config_path: Path,
    candidate_paths: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    config_sha = file_sha256(config_path)
    candidates = [
        verify_candidate(path, config=config, config_sha=config_sha)
        for path in candidate_paths
    ]
    steps = [int(candidate["step"]) for candidate in candidates]
    if not candidates or steps != sorted(set(steps)):
        raise RuntimeError("Blur specialization candidates must be unique and step-sorted")
    winner = max(
        candidates,
        key=lambda item: (selection_score(item, config["gate"]), -int(item["step"])),
    )
    result = {
        "schema_version": 1,
        "status": "frozen_blur_specialization_development_selection",
        "selection_rule": "maximize_blur_worst_normalized_psnr_lpips_margin_then_earlier_step",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "candidate_results": [
            {
                "step": int(candidate["step"]),
                "result": candidate_paths[index].relative_to(ROOT).as_posix(),
                "result_sha256": file_sha256(candidate_paths[index]),
                "checkpoint": candidate["checkpoint"],
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "blur": candidate["blur"]["specialized"],
                "noise_diagnostic": candidate["noise"]["specialized_false_positive_diagnostic"],
                "selection_score": selection_score(candidate, config["gate"]),
            }
            for index, candidate in enumerate(candidates)
        ],
        "selected_step": int(winner["step"]),
        "checkpoint": winner["checkpoint"],
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_development_metrics": winner,
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    candidates = [resolve_workspace_path(ROOT, path) for path in args.development_result]
    output = resolve_workspace_path(ROOT, args.output)
    result = freeze_selection(
        config_path=config_path,
        candidate_paths=candidates,
        output=output,
    )
    print("FOUNDIR_BLUR_SPECIALIZATION_SELECTION_PASS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
