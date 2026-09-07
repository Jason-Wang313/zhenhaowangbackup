#!/usr/bin/env python3
"""Build the 128-row FLUX overfit cache from frozen physical-v8 Qwen outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.evaluate_foundir_five_type_qwen_native_patches_ddp import (  # noqa: E402
    aggregated_predictions,
)
from smoke.export_foundir_five_type_qwen_conditions import (  # noqa: E402
    condition_row,
    encode_csv,
    load_manifest,
    sha256,
    write_once,
)


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def build(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    qwen = config["qwen_condition"]
    if (
        config["scope"].get("exclude_jpeg") is not True
        or config["evaluation"].get("condition_arms") != ["qwen"]
        or qwen.get("method") != "per_type_dialogue_center_crop_448"
        or qwen.get("inference_view") != "center"
    ):
        raise RuntimeError("the physical-v8 no-JPEG/no-ablation condition contract changed")

    manifest_path = workspace_path(config["runtime"]["manifest"])
    raw_path = workspace_path(qwen["foundir_training_views"])
    diagnostic_path = workspace_path(qwen["foundir_training_diagnostic"])
    if sha256(raw_path) != qwen["foundir_training_views_sha256"]:
        raise RuntimeError("frozen physical-v8 view predictions changed")
    if sha256(diagnostic_path) != qwen["foundir_training_diagnostic_sha256"]:
        raise RuntimeError("frozen physical-v8 FoundIR diagnostic changed")
    diagnostic = json.loads(diagnostic_path.read_text())
    if (
        diagnostic.get("protected_foundir_official_test_accessed") is not False
        or diagnostic.get("qwen_training_performed") is not False
        or diagnostic.get("metrics", {}).get("valid_output_rate") != 1.0
        or diagnostic.get("metrics", {}).get("minimum_type_recall", 0.0) < 0.5
        or diagnostic.get("metrics", {}).get("type_micro_f1", 0.0) < 0.8
    ):
        raise RuntimeError("physical-v8 Qwen does not meet the authorized admission floor")

    rows = load_manifest(manifest_path)
    pilot_ids = {
        row["parent_record_id"]
        for row in json.loads(raw_path.read_text())["view_predictions"]
    }
    pilot = [row for row in rows if row["record_id"] in pilot_ids]
    pilot.sort(key=lambda row: row["record_id"])
    if len(pilot) != int(config["overfit"]["records"]):
        raise RuntimeError("physical-v8 overfit record set changed")

    raw = json.loads(raw_path.read_text())
    predictions = aggregated_predictions(raw, pilot, method="center")
    by_id = {row["record_id"]: row for row in predictions}
    condition_rows = []
    for record in pilot:
        prediction = by_id[record["record_id"]]
        if prediction.get("valid") is not True:
            raise RuntimeError(f"invalid physical-v8 prediction: {record['record_id']}")
        raw_output = json.dumps(
            prediction["levels"], separators=(",", ":"), sort_keys=False
        )
        condition_rows.append(condition_row(record, raw_output))

    output_csv = workspace_path(config["overfit"]["conditions_csv"])
    output_json = workspace_path(config["overfit"]["conditions_json"])
    csv_content = encode_csv(condition_rows)
    write_once(output_csv, csv_content)
    payload = {
        "schema_version": 1,
        "status": "frozen_physical_v8_center_qwen_overfit_conditions",
        "records": len(condition_rows),
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256(config_path),
        "manifest": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "source_view_predictions": raw_path.relative_to(ROOT).as_posix(),
        "source_view_predictions_sha256": sha256(raw_path),
        "conditions_csv": output_csv.relative_to(ROOT).as_posix(),
        "conditions_csv_sha256": sha256(output_csv),
        "qwen_method": qwen["method"],
        "qwen_inference_view": "center",
        "condition_arms": ["qwen"],
        "jpeg_included": False,
        "protected_foundir_test_accessed": False,
    }
    write_once(
        output_json,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_flux2_dit_v2.json",
    )
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    payload = build(config_path)
    print("FOUNDIR_PHYSICAL_V8_OVERFIT_CONDITIONS_PASS " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
