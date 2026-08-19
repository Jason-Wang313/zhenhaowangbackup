#!/usr/bin/env python3
"""Evaluate the frozen step-600 Qwen visual checkpoint on real Lowlight proxies."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord, load_pair_manifest  # noqa: E402
from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    PRIMARY_SUBSET,
    STRESS_SUBSET,
    augment_real_metrics,
    load_real_validation_config,
    real_metric_gate_checks,
    validate_real_validation_contract,
)
from lowlight_restoration.qwen_supervised import (  # noqa: E402
    evaluate_generated_outputs,
    prompt_text,
)
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    load_inventory,
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (  # noqa: E402
    atomic_write_new_json,
    git_revision,
    load_model_and_processor,
    read_json,
    seed_everything,
)


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("real-validation manifest is empty")
    return rows


def _verified_full_preflight(
    config: dict[str, object], config_sha256: str
) -> dict[str, object]:
    runtime = config["runtime"]
    assert isinstance(runtime, dict)
    path = workspace_path(runtime["output_dir"]) / "preflight_full.json"
    if not path.is_file():
        raise RuntimeError("full GPU real-validation preflight is missing")
    payload = read_json(path)
    model = payload.get("model")
    data = payload.get("data")
    runtime_evidence = payload.get("runtime")
    if (
        payload.get("status") != "passed"
        or payload.get("config_sha256") != config_sha256
        or payload.get("contract_only") is not False
        or not isinstance(model, dict)
        or model.get("large_model_hashes_checked") is not True
        or not isinstance(data, dict)
        or not isinstance(data.get("sources"), dict)
        or data["sources"].get("hashes_checked") is not True
        or not isinstance(runtime_evidence, dict)
        or runtime_evidence.get("gpu_required") is not True
        or runtime_evidence.get("visible_cuda_devices") != 1
    ):
        raise RuntimeError("full GPU real-validation preflight is incomplete")
    return payload


def _load_frozen_model(
    config: dict[str, object], device: torch.device
) -> tuple[torch.nn.Module, Any, dict[str, object]]:
    inventory, _ = load_inventory(config)
    model, processor, parameter_groups = load_model_and_processor(
        config, inventory, device
    )
    configured_model = config["model"]
    assert isinstance(configured_model, dict)
    checkpoint = workspace_path(configured_model["selected_visual_checkpoint"])
    if sha256(checkpoint) != configured_model["selected_visual_sha256"]:
        raise RuntimeError("selected step-600 visual checkpoint changed before load")
    state = load_file(checkpoint, device="cpu")
    missing, unexpected = model.visual.load_state_dict(state, strict=True)
    del state
    if missing or unexpected:
        raise RuntimeError(
            f"selected visual checkpoint keys differ missing={missing} unexpected={unexpected}"
        )
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("frozen real-validation model retained trainable parameters")
    model.eval()
    model.config.use_cache = True
    return model, processor, {
        "model_revision": inventory["model"]["revision"],
        "selected_checkpoint_step": configured_model["selected_checkpoint_step"],
        "selected_visual_sha256": configured_model["selected_visual_sha256"],
        "parameter_groups": parameter_groups,
        "all_parameters_frozen": True,
    }


def _read_existing_batch(
    path: Path,
    *,
    config_sha256: str,
    checkpoint_sha256: str,
    subset: str,
    records: Sequence[PairRecord],
) -> list[str] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    outputs = payload.get("outputs")
    expected_ids = [record.pair_id for record in records]
    if (
        payload.get("config_sha256") != config_sha256
        or payload.get("selected_visual_sha256") != checkpoint_sha256
        or payload.get("subset") != subset
        or payload.get("pair_ids") != expected_ids
        or not isinstance(outputs, list)
        or len(outputs) != len(records)
        or any(not isinstance(value, str) for value in outputs)
    ):
        raise RuntimeError(f"existing real-validation batch differs: {path}")
    return outputs


def _generate_batch(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    *,
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    images = []
    for record in records:
        with Image.open(record.degraded_path) as image:
            images.append(image.convert("RGB"))
    fixed_prompt = prompt_text(processor)
    encoded = processor(
        text=[fixed_prompt] * len(records),
        images=images,
        padding=True,
        return_tensors="pt",
    )
    del images
    inputs = {
        key: value.to(device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    if not torch.all(inputs["attention_mask"][:, -1] == 1):
        raise RuntimeError("left-padded real generation ends in a padding token")
    prompt_width = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    if int(generated.shape[0]) != len(records):
        raise RuntimeError("real-validation generation cardinality changed")
    return [
        processor.tokenizer.decode(
            generated[row, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        for row in range(len(records))
    ]


def _generate_subset(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    *,
    output_dir: Path,
    subset: str,
    config_sha256: str,
    checkpoint_sha256: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[list[str], dict[str, int]]:
    outputs: list[str] = []
    generated_batches = reused_batches = 0
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch = list(records[start : start + batch_size])
        path = output_dir / "batches" / subset / f"batch_{batch_index:04d}.json"
        existing = _read_existing_batch(
            path,
            config_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            subset=subset,
            records=batch,
        )
        if existing is not None:
            raw = existing
            reused_batches += 1
            marker = "REUSED"
        else:
            raw = _generate_batch(
                model,
                processor,
                batch,
                device=device,
                max_new_tokens=max_new_tokens,
            )
            atomic_write_new_json(
                path,
                {
                    "schema_version": 1,
                    "config_sha256": config_sha256,
                    "selected_visual_sha256": checkpoint_sha256,
                    "subset": subset,
                    "batch_index": batch_index,
                    "pair_ids": [record.pair_id for record in batch],
                    "outputs": raw,
                },
            )
            generated_batches += 1
            marker = "GENERATED"
        outputs.extend(raw)
        print(
            f"QWEN_REAL_BATCH_{marker} subset={subset} "
            f"batch={batch_index + 1}/{total_batches} records={len(outputs)}/{len(records)}",
            flush=True,
        )
    return outputs, {
        "total_batches": total_batches,
        "generated_batches_this_attempt": generated_batches,
        "reused_batches_this_attempt": reused_batches,
    }


def _evaluate_subset(
    *,
    subset: str,
    records: Sequence[PairRecord],
    rows_by_pair: dict[str, dict[str, str]],
    raw_outputs: Sequence[str],
    gate: dict[str, object],
    output_dir: Path,
    config_sha256: str,
    checkpoint_step: int,
) -> tuple[dict[str, object], dict[str, bool], dict[str, str]]:
    prediction_rows, base_metrics = evaluate_generated_outputs(records, raw_outputs)
    metrics = augment_real_metrics(prediction_rows, base_metrics)
    checks = real_metric_gate_checks(metrics, gate, prefix=subset)
    enriched = []
    for prediction in prediction_rows:
        source = rows_by_pair[str(prediction["pair_id"])]
        enriched.append(
            {
                **prediction,
                "dataset": source["dataset"],
                "subset": source["subset"],
                "annotation_kind": source["annotation_kind"],
                "annotation_method": source["annotation_method"],
                "input_reference_luma_ratio": float(
                    source["input_reference_luma_ratio"]
                ),
                "proxy_anchor_error_ev": float(source["proxy_anchor_error_ev"]),
                "degraded_path": source["degraded_path"],
                "degraded_sha256": source["degraded_sha256"],
                "reference_path": source["reference_path"],
                "reference_sha256": source["reference_sha256"],
            }
        )
    failures = [row for row in enriched if row["severity_correct"] is not True]
    payloads: dict[str, object] = {
        "metrics": {
            "schema_version": 1,
            "config_sha256": config_sha256,
            "selected_checkpoint_step": checkpoint_step,
            "subset": subset,
            "annotation_kind": "reference_based_proxy_not_human_ground_truth",
            "metrics": metrics,
            "gate_checks": checks,
            "automatic_subset_pass": all(checks.values()),
        },
        "predictions": {
            "schema_version": 1,
            "config_sha256": config_sha256,
            "selected_checkpoint_step": checkpoint_step,
            "subset": subset,
            "annotation_kind": "reference_based_proxy_not_human_ground_truth",
            "records": enriched,
        },
        "failures": {
            "schema_version": 1,
            "config_sha256": config_sha256,
            "selected_checkpoint_step": checkpoint_step,
            "subset": subset,
            "total_failures": len(failures),
            "records": failures,
        },
        "confusion": {
            "schema_version": 1,
            "config_sha256": config_sha256,
            "selected_checkpoint_step": checkpoint_step,
            "subset": subset,
            "severity_confusion": metrics["confusion"],
            "degradation_type_f1": metrics["degradation_type_f1"],
            "severity_f1": metrics["severity_f1"],
        },
    }
    artifacts = {
        name: output_dir / "evaluation" / f"{subset}_{name}.json"
        for name in payloads
    }
    for name, payload in payloads.items():
        atomic_write_new_json(artifacts[name], payload)
    return metrics, checks, {name: sha256(path) for name, path in artifacts.items()}


def _completed_result(
    output_dir: Path, config_sha256: str, checkpoint_sha256: str
) -> int | None:
    path = output_dir / "result_summary.json"
    if not path.is_file():
        return None
    summary = read_json(path)
    checks = summary.get("gate_checks")
    artifacts = summary.get("artifact_sha256")
    if (
        summary.get("config_sha256") != config_sha256
        or summary.get("selected_visual_sha256") != checkpoint_sha256
        or not isinstance(checks, dict)
        or not checks
        or not isinstance(artifacts, dict)
        or not artifacts
    ):
        raise RuntimeError("existing real-validation result is incomplete")
    for relative, expected in artifacts.items():
        artifact = workspace_path(relative)
        if not artifact.is_file() or sha256(artifact) != expected:
            raise RuntimeError(f"existing real-validation artifact changed: {artifact}")
    status = summary.get("status")
    automatic_pass = summary.get("automatic_gate_pass")
    if status == "passed" and automatic_pass is True and all(checks.values()):
        print("QWEN_REAL_VALIDATION_PASS existing_verified_result", flush=True)
        return 0
    if status == "failed" and automatic_pass is False and not all(checks.values()):
        print("QWEN_REAL_VALIDATION_FAILED existing_verified_result", flush=True)
        return 2
    raise RuntimeError("existing real-validation terminal status is inconsistent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, _ = load_real_validation_config(args.config)
    validate_real_validation_contract(config)
    model_config = config["model"]
    data = config["data"]
    evaluation = config["evaluation"]
    gates = config["gates"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (model_config, data, evaluation, gates, runtime)
    )
    output_dir = workspace_path(runtime["output_dir"])
    existing = _completed_result(
        output_dir, config_sha, str(model_config["selected_visual_sha256"])
    )
    if existing is not None:
        return existing

    preflight = _verified_full_preflight(config, config_sha)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("real validation requires exactly one visible CUDA GPU")
    seed_everything(int(runtime["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    device = torch.device("cuda:0")

    manifest_path = workspace_path(data["manifest"])
    records = load_pair_manifest(manifest_path)
    manifest_rows = _manifest_rows(manifest_path)
    rows_by_pair = {row["pair_id"]: row for row in manifest_rows}
    subset_records = {
        subset: [record for record in records if rows_by_pair[record.pair_id]["subset"] == subset]
        for subset in evaluation["subset_order"]
    }
    started = time.monotonic()
    model, processor, frozen_model_evidence = _load_frozen_model(config, device)
    original_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    subset_results: dict[str, object] = {}
    flat_checks: dict[str, bool] = {}
    artifact_hashes: dict[str, str] = {}
    try:
        for subset in evaluation["subset_order"]:
            selected = subset_records[subset]
            print(
                f"QWEN_REAL_SUBSET_BEGIN subset={subset} records={len(selected)}",
                flush=True,
            )
            raw_outputs, batch_evidence = _generate_subset(
                model,
                processor,
                selected,
                output_dir=output_dir,
                subset=subset,
                config_sha256=config_sha,
                checkpoint_sha256=str(model_config["selected_visual_sha256"]),
                batch_size=int(evaluation["batch_size"]),
                max_new_tokens=int(evaluation["max_new_tokens"]),
                device=device,
            )
            subset_gate = gates["subsets"][subset]
            assert isinstance(subset_gate, dict)
            metrics, checks, artifacts = _evaluate_subset(
                subset=subset,
                records=selected,
                rows_by_pair=rows_by_pair,
                raw_outputs=raw_outputs,
                gate=subset_gate,
                output_dir=output_dir,
                config_sha256=config_sha,
                checkpoint_step=int(model_config["selected_checkpoint_step"]),
            )
            flat_checks.update(checks)
            for name, digest in artifacts.items():
                relative = str(
                    (output_dir / "evaluation" / f"{subset}_{name}.json").relative_to(ROOT)
                )
                artifact_hashes[relative] = digest
            subset_results[subset] = {
                "records": len(selected),
                "metrics": metrics,
                "gate_checks": checks,
                "automatic_subset_pass": all(checks.values()),
                "batch_evidence": batch_evidence,
                "artifact_sha256": artifacts,
            }
            print(
                f"QWEN_REAL_SUBSET_END subset={subset} "
                f"severity_accuracy={float(metrics['severity_accuracy']):.6f} "
                f"within_one={float(metrics['within_one_level_accuracy']):.6f} "
                f"passed={all(checks.values())}",
                flush=True,
            )
    finally:
        processor.tokenizer.padding_side = original_padding_side
        model.config.use_cache = False

    automatic_pass = all(flat_checks.values())
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if automatic_pass else "failed",
        "scope": "frozen_step600_non_FoundIR_real_Lowlight_validation",
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "annotation_kind": "reference_based_proxy_not_human_ground_truth",
        "annotation_warning": (
            "Project levels are reference-based proxy labels, not human or official "
            "SICE/LOL-v2 severity ground truth."
        ),
        "official_foundir_used": False,
        "model_predictions_consulted_during_selection": False,
        "selected_checkpoint_step": model_config["selected_checkpoint_step"],
        "selected_visual_sha256": model_config["selected_visual_sha256"],
        "model": frozen_model_evidence,
        "manifest_sha256": data["manifest_sha256"],
        "metadata_sha256": data["metadata_sha256"],
        "gpu_count": 1,
        "gpu_count_rationale": (
            "Small frozen 233-image evaluation; one H100/H200 avoids multi-process "
            "model-loading overhead while preserving the fixed inference contract."
        ),
        "preflight_sha256": sha256(output_dir / "preflight_full.json"),
        "runtime": preflight["runtime"],
        "subsets": subset_results,
        "gate_checks": flat_checks,
        "automatic_gate_pass": automatic_pass,
        "artifact_sha256": artifact_hashes,
    }
    atomic_write_new_json(output_dir / "result_summary.json", summary)
    elapsed = time.monotonic() - started
    print(
        ("QWEN_REAL_VALIDATION_PASS" if automatic_pass else "QWEN_REAL_VALIDATION_FAILED"),
        json.dumps(flat_checks, sort_keys=True),
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return 0 if automatic_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
