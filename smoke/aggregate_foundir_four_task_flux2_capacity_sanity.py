#!/usr/bin/env python3
"""Validate and aggregate all four independent FLUX.2 capacity-arm results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise RuntimeError(f"refusing to overwrite differing aggregate: {path}")
    if not path.exists():
        path.write_bytes(content)


def aggregate(config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    config_sha = sha256(config_path)
    root = ROOT / config["runtime"]["output_root"]
    tasks: dict[str, object] = {}
    for category, spec in config["tasks"].items():
        result_path = root / spec["slug"] / "result.json"
        if not result_path.is_file():
            raise RuntimeError(f"missing terminal task result: {result_path}")
        result = json.loads(result_path.read_text())
        checkpoint = ROOT / result["checkpoint"]
        if (
            result.get("task") != category
            or result.get("config_sha256") != config_sha
            or result.get("qwen_degradation_conditioning") is not False
            or result.get("mixed_degradations") is not False
            or result.get("pixel_refiner") is not False
            or result.get("sealed_internal_accessed") is not False
            or result.get("protected_foundir_test_accessed") is not False
            or result.get("development_gt_optimization") is not False
            or not checkpoint.is_file()
            or sha256(checkpoint) != result.get("checkpoint_sha256")
            or result["final_metrics"].get("records") != 16
            or result["descriptive_validation_metrics"].get("records") != 16
        ):
            raise RuntimeError(f"task result identity/contract failed: {category}")
        tasks[category] = {
            "result": result_path.relative_to(ROOT).as_posix(),
            "result_sha256": sha256(result_path),
            "checkpoint": result["checkpoint"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "automatic_capacity_gate_pass": result["automatic_gate_pass"],
            "human_visual_review": result["human_visual_review"],
            "train": {
                "psnr": result["final_metrics"]["mean_output_psnr"],
                "lpips": result["final_metrics"]["mean_output_lpips"],
                "psnr_gain_db": result["final_metrics"]["mean_psnr_gain_db"],
                "lpips_improvement": result["final_metrics"]["mean_lpips_improvement"],
            },
            "train_sow_proxy": result["train_sow_proxy"],
            "descriptive_validation": {
                "psnr": result["descriptive_validation_metrics"]["mean_output_psnr"],
                "lpips": result["descriptive_validation_metrics"]["mean_output_lpips"],
                "psnr_gain_db": result["descriptive_validation_metrics"]["mean_psnr_gain_db"],
                "lpips_improvement": result["descriptive_validation_metrics"]["mean_lpips_improvement"],
            },
        }
    all_capacity = all(item["automatic_capacity_gate_pass"] for item in tasks.values())
    return {
        "schema_version": 1,
        "status": "four_task_capacity_complete_awaiting_joint_visual_audit",
        "automatic_gate_pass": all_capacity,
        "human_visual_review_required": True,
        "human_visual_review_complete": False,
        "human_visual_review": "pending",
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "tasks": tasks,
        "joint_training_started": False,
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "development_gt_optimization": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload = aggregate(config_path)
    output = ROOT / json.loads(config_path.read_text())["runtime"]["output_root"] / "result.json"
    write_once(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_FOUR_TASK_CAPACITY_AGGREGATE_COMPLETE " + json.dumps({
        "automatic_gate_pass": payload["automatic_gate_pass"],
        "tasks": {key: value["automatic_capacity_gate_pass"] for key, value in payload["tasks"].items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
