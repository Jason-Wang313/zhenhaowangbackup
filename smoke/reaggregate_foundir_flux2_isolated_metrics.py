#!/usr/bin/env python3
"""Reaggregate immutable FLUX sidecars under the corrected isolated SOW contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.foundir_flux2_protocol import (  # noqa: E402
    resolve_workspace_path,
    select_checkpoint,
    worst_normalized_margin,
)
from smoke.evaluate_foundir_five_type_flux2_ddp import aggregate, verify_sidecar, write_once  # noqa: E402


def reaggregate(source: Path, config: Mapping[str, object], config_sha: str) -> dict[str, object]:
    payload = json.loads(source.read_text())
    if (
        payload.get("status") not in {"development_evaluation_complete", "sealed_internal_test_complete"}
        or payload.get("config_sha256") != config_sha
        or payload.get("split") not in {"validation", "internal_test"}
        or payload.get("arm") != "qwen"
        or payload.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError(f"invalid source evaluation: {source}")
    sidecar_dir = source.parent / "per_record"
    sidecar_paths = sorted(sidecar_dir.glob("*.json"))
    expected = int(payload["summary"]["records"])
    if len(sidecar_paths) != expected:
        raise RuntimeError(f"source sidecar cardinality changed: {source}")
    rows = []
    seen = set()
    for path in sidecar_paths:
        raw = json.loads(path.read_text())
        identity = {
            "config_sha256": payload["config_sha256"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "conditions_sha256": payload["conditions_sha256"],
            "split": payload["split"],
            "arm": payload["arm"],
            "record_id": raw.get("record_id"),
        }
        row = verify_sidecar(path, identity)
        if row["record_id"] in seen:
            raise RuntimeError(f"duplicate source record: {row['record_id']}")
        seen.add(row["record_id"])
        rows.append(row)
    summary = aggregate(rows, config=config)
    return {
        "schema_version": 4,
        "status": (
            "corrected_isolated_development_reaggregation"
            if payload["split"] == "validation"
            else "corrected_isolated_sealed_diagnostic_reaggregation"
        ),
        "metric_contract": "sow_isolated_categories_v1",
        "source_result": source.relative_to(ROOT).as_posix(),
        "source_result_sha256": file_sha256(source),
        "config_sha256": config_sha,
        "checkpoint": payload["checkpoint"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "conditions_sha256": payload["conditions_sha256"],
        "split": payload["split"],
        "arm": payload["arm"],
        "summary": summary,
        "source_pixels_reused_without_inference": True,
        "checkpoint_selection_performed": payload["split"] == "validation",
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, action="append", default=[])
    parser.add_argument("--sealed-result", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config_path = resolve_workspace_path(ROOT, args.config)
    output_root = resolve_workspace_path(ROOT, args.output_root)
    config = json.loads(config_path.read_text())
    config_sha = file_sha256(config_path)
    corrected = []
    for raw in args.development_result:
        source = resolve_workspace_path(ROOT, raw)
        result = reaggregate(source, config, config_sha)
        step = int(Path(result["checkpoint"]).stem.split("_", 1)[1])
        path = output_root / "development" / f"step_{step:06d}" / "result.json"
        write_once(path, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
        corrected.append({**result, "result": path.relative_to(ROOT).as_posix()})
    if not corrected:
        raise RuntimeError("at least one development result is required")
    candidates = [
        {
            "step": int(Path(item["checkpoint"]).stem.split("_", 1)[1]),
            "checkpoint": item["checkpoint"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "task_metrics": item["summary"]["task_metrics"],
            "worst_normalized_margin": worst_normalized_margin(item["summary"]["task_metrics"]),
            "corrected_result": item["result"],
        }
        for item in corrected
    ]
    winner = select_checkpoint(candidates)
    selection = {
        "schema_version": 4,
        "status": "corrected_isolated_development_selected_checkpoint",
        "metric_contract": "sow_isolated_categories_v1",
        "config_sha256": config_sha,
        "selection_rule": "maximize_worst_five_task_normalized_psnr_lpips_margin_then_earlier_step",
        "candidates": candidates,
        "selected_step": winner["step"],
        "checkpoint": winner["checkpoint"],
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "selected_development_result": winner["corrected_result"],
        "selected_worst_normalized_margin": winner["worst_normalized_margin"],
        "sealed_internal_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(
        output_root / "corrected_checkpoint_selection.json",
        (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode(),
    )
    if args.sealed_result is not None:
        source = resolve_workspace_path(ROOT, args.sealed_result)
        sealed = reaggregate(source, config, config_sha)
        write_once(
            output_root / "sealed_existing_checkpoint_diagnostic" / "result.json",
            (json.dumps(sealed, indent=2, sort_keys=True) + "\n").encode(),
        )
    print("FOUNDIR_ISOLATED_REAGGREGATION_PASS " + json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
