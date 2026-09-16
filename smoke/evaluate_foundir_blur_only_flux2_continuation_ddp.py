#!/usr/bin/env python3
"""Evaluate the staged FLUX.2 Blur continuation on all 2,595 development records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import smoke.evaluate_foundir_blur_only_flux2_full_development_ddp as base  # noqa: E402


def validate_config(config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    config = base.load_json(config_path)
    if (
        config.get("status") != "user_authorized_blur_only_continuation_development_evaluation"
        or config.get("authorization", {}).get("hold_id")
        != "flux2_blur_continuation_authorization-60565f4f75ef056b"
        or config.get("authorization", {}).get("hold_sha256")
        != "52535877d8b0ede5424fb41c6191f7eb9969778ca42d15830de74f17003a628b"
    ):
        raise RuntimeError("continuation evaluation authorization changed")
    if config["scope"] != {
        "category": "01Blur",
        "split": "validation",
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
        "development_gt_optimization": False,
    }:
        raise RuntimeError("continuation evaluation scope changed")
    parent_path = base.workspace_path(config["parent_training_config"])
    parent = base.load_json(parent_path)
    if (
        parent.get("status") != "user_authorized_blur_only_low_lr_continuation"
        or parent.get("model", {}).get("name") != "flux.2-klein-base-4b"
        or parent.get("scope", {}).get("qwen_degradation_conditioning") is not False
        or parent.get("scope", {}).get("mixed_degradations") is not False
        or parent.get("scope", {}).get("pixel_refiner") is not False
    ):
        raise RuntimeError("continuation parent config changed")
    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, list) or [int(item["step"]) for item in checkpoints] != [12000]:
        raise RuntimeError("step-12000 must be the sole first-gate checkpoint")
    parent_sha = base.sha256(parent_path)
    for item in checkpoints:
        checkpoint = base.workspace_path(item["path"])
        if not checkpoint.is_file():
            raise RuntimeError(f"missing continuation checkpoint: {checkpoint}")
        metadata = base.load_json(checkpoint.with_suffix(checkpoint.suffix + ".json"))
        checkpoint_sha = base.sha256(checkpoint)
        if int(metadata.get("step", -1)) != 12000 or metadata.get("config_sha256") != parent_sha:
            raise RuntimeError("continuation checkpoint metadata changed")
        item["sha256"] = checkpoint_sha
        optimizer = checkpoint.with_suffix(".optimizer.pt")
        if not optimizer.is_file():
            raise RuntimeError("continuation checkpoint lacks optimizer state")
    data, inference, metrics, distributed = (
        config["data"], config["inference"], config["metrics"], config["distributed"]
    )
    if (
        int(data.get("expected_records", -1)) != 2595
        or int(data.get("expected_source_groups", -1)) != 165
        or int(data.get("crop_size", -1)) != 512
        or data.get("crop_policy") != "one_deterministic_aligned_512_crop_per_record_shared_across_checkpoints"
        or inference.get("schedule") != "official_flux2_shifted_schedule"
        or int(inference.get("num_steps", -1)) != 50
        or float(inference.get("guidance", -1)) != 1.0
        or inference.get("quantization") != "round_clip_uint8_png_equivalent_before_metrics"
        or metrics.get("names") != ["psnr", "lpips"]
        or metrics.get("bootstrap_unit") != "source_group_equal_weight"
        or int(metrics.get("bootstrap_resamples", -1)) != 5000
        or int(distributed.get("required_world_size", -1)) != 8
        or distributed.get("allowed_gpu_name_fragments") != ["H100", "H200"]
        or distributed.get("evaluation_batch_size_per_rank") != {"H100": 2, "H200": 4}
    ):
        raise RuntimeError("continuation evaluation contract changed")
    if base.sha256(base.workspace_path(config["candidate_bundle"])) != config["candidate_bundle_sha256"]:
        raise RuntimeError("continuation authorization evidence changed")
    return config, parent


def run_preflight(config_path: Path) -> None:
    config, _ = validate_config(config_path)
    rows = base.load_development_rows(config)
    visual_root = ROOT / "checkpoints/foundir_blur_only_flux2_continuation_v1/fixed_validation/training_triptychs"
    directory = visual_root / "step_012000"
    if not (directory / "metrics.json").is_file() or len(list(directory.glob("*.png"))) != 16:
        raise RuntimeError(f"step-12000 fixed visual evidence is incomplete: {directory}")
    print("FOUNDIR_BLUR_FLUX2_CONTINUATION_DEVELOPMENT_PREFLIGHT_PASS " + json.dumps({
        "records": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "checkpoint_step": 12000,
        "checkpoint_sha256": config["checkpoints"][0]["sha256"],
        "sealed_internal_accessed": False,
        "protected_foundir_test_accessed": False,
    }, sort_keys=True), flush=True)


def render_report(result: Mapping[str, object]) -> str:
    item = result["checkpoint_curve"][0]
    return "\n".join([
        "# FLUX.2 Blur-only step-12k continuation development 结果",
        "",
        "> DEVELOPMENT ONLY；未访问 sealed internal 或 official FoundIR test。",
        "",
        f"- 全部 {result['records']} 张 source-group-disjoint Blur validation images，{result['source_groups']} 个 source groups。",
        "- 训练从 step-10000 LoRA 与 AdamW optimizer 精确恢复；global batch 32；纯 flow objective 未改变。",
        "- 推理为 official shifted 50 steps、guidance 1.0，并在 PNG-equivalent uint8 量化后计算指标。",
        "",
        "| Step | PSNR | LPIPS | PSNR >=25.9 | LPIPS <=0.18 |",
        "|---:|---:|---:|:---:|:---:|",
        f"| 12000 | {item['metrics']['psnr']:.6f} | {item['metrics']['lpips']:.6f} | {'pass' if item['sow_checks']['psnr'] else 'fail'} | {'pass' if item['sow_checks']['lpips'] else 'fail'} |",
        "",
        "只有 step-12k 相对 step-10k 的 25.734236 / 0.145406 仍有价值时，才继续到 step-14k。",
        "",
    ])


def main() -> int:
    base.validate_config = validate_config
    base.run_preflight = run_preflight
    base.render_advisor_report = render_report
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
