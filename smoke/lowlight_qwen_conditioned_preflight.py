#!/usr/bin/env python3
"""Fail-fast preflight for the fair Qwen-conditioned FLUX.2 run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.condition_adapter import LowLightLevelContext
from lowlight_restoration.conditioned_authorization import (
    validate_conditioned_authorization,
    workspace_path,
)
from lowlight_restoration.config import load_train_config
from lowlight_restoration.data import load_pair_manifest
from lowlight_restoration.qwen_predictions import (
    load_frozen_ridge,
    load_prediction_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--require-predictions", action="store_true")
    args = parser.parse_args()

    config = load_train_config(args.config)
    try:
        scope = validate_conditioned_authorization(config)
    except RuntimeError as exc:
        raise SystemExit(f"RUN_NOT_AUTHORIZED {exc}") from exc
    if scope != "full_qwen_conditioned":
        raise SystemExit("QWEN_CONDITIONED_SCOPE_REQUIRED")
    expected_prefix = ROOT / ".venv/flux2-smoke"
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise SystemExit(
            f"WRONG_PYTHON expected={expected_prefix} actual={sys.prefix}"
        )

    condition = config.section("condition")
    adapter = LowLightLevelContext(
        context_dim=int(condition["context_dim"]),
        levels=int(condition["levels"]),
        init_std=float(condition["init_std"]),
    )
    adapter_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    if adapter_parameters != int(condition["expected_trainable_parameters"]):
        raise SystemExit(f"BAD_CONDITION_ADAPTER_PARAMETERS {adapter_parameters}")
    prompt_path = workspace_path(
        condition["baseline_prompt_context"],
        label="condition.baseline_prompt_context",
    )
    prompt = load_file(str(prompt_path), device="cpu")
    if set(prompt) != {"context", "context_ids", "empty_context", "empty_context_ids"}:
        raise SystemExit("BAD_FROZEN_PROMPT_TENSORS")
    context = torch.cat([prompt["empty_context"], prompt["context"]], dim=0)
    context_ids = torch.cat(
        [prompt["empty_context_ids"], prompt["context_ids"]], dim=0
    )
    conditioned, conditioned_ids = adapter(
        context, context_ids, torch.tensor([1, 5], dtype=torch.int64)
    )
    if conditioned.shape != (2, 513, 7680) or conditioned_ids.shape != (2, 513, 4):
        raise SystemExit("BAD_CONDITION_ADAPTER_SHAPE")
    del adapter, context, context_ids, conditioned, conditioned_ids

    load_frozen_ridge(
        workspace_path(condition["qwen_checkpoint"], label="condition.qwen_checkpoint")
    )
    records = load_pair_manifest(config.workspace_path("data", "manifest"))
    split_counts = {
        split: sum(record.split == split for record in records)
        for split in ("train", "validation", "test")
    }
    if split_counts != condition["expected_prediction_splits"]:
        raise SystemExit(f"BAD_MANIFEST_SPLIT_COUNTS {split_counts}")

    prediction_path = workspace_path(
        condition["prediction_manifest"], label="condition.prediction_manifest"
    )
    prediction_state: dict[str, object]
    if prediction_path.exists():
        _, prediction_summary, prediction_sha = load_prediction_manifest(
            config, records, verify_sidecars=True
        )
        prediction_state = {
            "present": True,
            "records": prediction_summary["records"],
            "manifest_sha256": prediction_sha,
            "bundle_sha256": prediction_summary["prediction_bundle_sha256"],
        }
    else:
        if args.require_predictions:
            raise SystemExit("MISSING_REQUIRED_QWEN_CONDITION_PREDICTIONS")
        output_dir = workspace_path(
            condition["prediction_output_dir"], label="condition.prediction_output_dir"
        )
        existing_sidecars = (
            len(list((output_dir / "records").glob("*.json")))
            if (output_dir / "records").is_dir()
            else 0
        )
        prediction_state = {
            "present": False,
            "existing_resumable_sidecars": existing_sidecars,
        }

    output_dir = config.workspace_path("runtime", "output_dir")
    frozen_config = output_dir / "config.json"
    if frozen_config.exists() and frozen_config.read_bytes() != config.path.read_bytes():
        raise SystemExit("CONDITIONED_OUTPUT_HAS_DIFFERENT_CONFIG")
    if shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise SystemExit("LESS_THAN_20_GIB_FREE_DISK")
    if not args.cpu_only:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise SystemExit("CUDA_NOT_AVAILABLE")
        properties = torch.cuda.get_device_properties(0)
        required = int(config.section("runtime")["minimum_vram_gib"]) * 1024**3
        if properties.total_memory < required:
            raise SystemExit(f"INSUFFICIENT_VRAM {properties.total_memory}")
        print(f"OK_CUDA name={properties.name} bytes={properties.total_memory}")
    print(
        "LOWLIGHT_QWEN_CONDITIONED_PREFLIGHT_PASS "
        + json.dumps(
            {
                "scope": scope,
                "cpu_only": args.cpu_only,
                "config_sha256": config.sha256,
                "fair_baseline_sections": [
                    "model",
                    "data",
                    "lora",
                    "training",
                    "inference",
                ],
                "adapter_parameters": adapter_parameters,
                "total_trainable_parameters": condition[
                    "expected_total_trainable_parameters"
                ],
                "prediction_state": prediction_state,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
