#!/usr/bin/env python3
"""Score five canonical JSON targets in the post-failure Qwen diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord, load_pair_manifest  # noqa: E402
from lowlight_restoration.qwen_constrained import (  # noqa: E402
    CANONICAL_LEVELS,
    mean_shifted_target_token_nll,
    select_minimum_mean_nll,
)
from lowlight_restoration.qwen_real_validation import (  # noqa: E402
    load_real_validation_config,
    validate_real_validation_contract,
)
from lowlight_restoration.qwen_supervised import (  # noqa: E402
    canonical_target,
    open_degraded_images,
    prepare_supervised_batch,
)
from smoke.evaluate_qwen_lowlight_real_validation import (  # noqa: E402
    _completed_result,
    _evaluate_subset,
    _load_frozen_model,
    _manifest_rows,
)
from smoke.qwen_lowlight_real_validation_constrained_preflight import (  # noqa: E402
    load_diagnostic_config,
    validate_diagnostic_contract,
)
from smoke.qwen_lowlight_supervised_preflight import (  # noqa: E402
    sha256,
    workspace_path,
)
from smoke.train_qwen_lowlight_supervised import (  # noqa: E402
    atomic_write_new_json,
    git_revision,
    read_json,
    seed_everything,
)


def _verified_full_preflight(
    config: dict[str, object], config_sha256: str
) -> dict[str, object]:
    runtime = config["runtime"]
    parent = config["parent_failure"]
    assert isinstance(runtime, dict) and isinstance(parent, dict)
    path = workspace_path(runtime["output_dir"]) / "preflight_full.json"
    if not path.is_file():
        raise RuntimeError("full GPU constrained-diagnostic preflight is missing")
    payload = read_json(path)
    model = payload.get("model")
    data = payload.get("data")
    runtime_evidence = payload.get("runtime")
    parent_failure = payload.get("parent_failure")
    if (
        payload.get("status") != "passed"
        or payload.get("config_sha256") != config_sha256
        or payload.get("contract_only") is not False
        or payload.get("interpretation") != config.get("interpretation")
        or not isinstance(model, dict)
        or model.get("large_model_hashes_checked") is not True
        or not isinstance(data, dict)
        or not isinstance(data.get("sources"), dict)
        or data["sources"].get("hashes_checked") is not True
        or not isinstance(runtime_evidence, dict)
        or runtime_evidence.get("gpu_required") is not True
        or runtime_evidence.get("visible_cuda_devices") != 1
        or not isinstance(parent_failure, dict)
        or parent_failure.get("result_sha256") != parent.get("result_sha256")
        or parent_failure.get("automatic_gate_pass") is not False
    ):
        raise RuntimeError("full constrained-diagnostic GPU preflight is incomplete")
    return payload


def _read_existing_score_batch(
    path: Path,
    *,
    config_sha256: str,
    checkpoint_sha256: str,
    parent_result_sha256: str,
    subset: str,
    records: Sequence[PairRecord],
) -> tuple[list[str], list[dict[str, object]]] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    expected_ids = [record.pair_id for record in records]
    outputs = payload.get("outputs")
    score_rows = payload.get("score_rows")
    if (
        payload.get("config_sha256") != config_sha256
        or payload.get("selected_visual_sha256") != checkpoint_sha256
        or payload.get("parent_result_sha256") != parent_result_sha256
        or payload.get("subset") != subset
        or payload.get("pair_ids") != expected_ids
        or not isinstance(outputs, list)
        or not isinstance(score_rows, list)
        or len(outputs) != len(records)
        or len(score_rows) != len(records)
        or any(not isinstance(value, str) for value in outputs)
    ):
        raise RuntimeError(f"existing constrained score batch differs: {path}")
    normalized: list[dict[str, object]] = []
    for record, output, row in zip(records, outputs, score_rows, strict=True):
        if not isinstance(row, dict) or row.get("pair_id") != record.pair_id:
            raise RuntimeError(f"existing constrained score row differs: {path}")
        scores = row.get("candidate_mean_nll")
        if not isinstance(scores, dict):
            raise RuntimeError(f"existing constrained scores are missing: {path}")
        selected, selected_nll, margin = select_minimum_mean_nll(scores)
        if (
            row.get("selected_level") != selected
            or float(row.get("selected_mean_nll", -1.0)) != selected_nll
            or row.get("runner_up_margin") != margin
            or output != canonical_target(selected)
        ):
            raise RuntimeError(f"existing constrained selection differs: {path}")
        normalized.append(dict(row))
    return list(outputs), normalized


def _score_batch(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    *,
    device: torch.device,
) -> tuple[list[str], list[dict[str, object]]]:
    images = open_degraded_images(records)
    scores_by_row: list[dict[str, float]] = [dict() for _ in records]
    token_counts_by_level: dict[str, int] = {}
    try:
        for level in CANONICAL_LEVELS:
            candidates = [
                replace(record, project_level=f"L{level}") for record in records
            ]
            batch, expected_counts = prepare_supervised_batch(
                processor, candidates, images
            )
            labels = batch.pop("labels").to(device)
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.inference_mode():
                output = model(**inputs, use_cache=False, return_dict=True)
                means, realized_counts = mean_shifted_target_token_nll(
                    output.logits, labels
                )
            if realized_counts != expected_counts or len(means) != len(records):
                raise RuntimeError("canonical candidate target-token count changed")
            if len(set(realized_counts)) != 1:
                raise RuntimeError("one canonical target has variable token length")
            token_counts_by_level[str(level)] = realized_counts[0]
            for row, mean_nll in enumerate(means):
                scores_by_row[row][str(level)] = mean_nll
            del output, labels, inputs, batch
    finally:
        for image in images:
            image.close()

    outputs: list[str] = []
    score_rows: list[dict[str, object]] = []
    for record, scores in zip(records, scores_by_row, strict=True):
        selected, selected_nll, margin = select_minimum_mean_nll(scores)
        outputs.append(canonical_target(selected))
        score_rows.append(
            {
                "pair_id": record.pair_id,
                "candidate_mean_nll": scores,
                "candidate_target_token_counts": token_counts_by_level,
                "selected_level": selected,
                "selected_mean_nll": selected_nll,
                "runner_up_margin": margin,
            }
        )
    return outputs, score_rows


def _score_subset(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    *,
    output_dir: Path,
    subset: str,
    config_sha256: str,
    checkpoint_sha256: str,
    parent_result_sha256: str,
    batch_size: int,
    device: torch.device,
) -> tuple[list[str], list[dict[str, object]], dict[str, object], dict[str, str]]:
    outputs: list[str] = []
    score_rows: list[dict[str, object]] = []
    generated_batches = reused_batches = 0
    batch_hashes: dict[str, str] = {}
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(records), batch_size)):
        batch_records = list(records[start : start + batch_size])
        path = output_dir / "score_batches" / subset / f"batch_{batch_index:04d}.json"
        existing = _read_existing_score_batch(
            path,
            config_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            parent_result_sha256=parent_result_sha256,
            subset=subset,
            records=batch_records,
        )
        if existing is None:
            batch_outputs, batch_scores = _score_batch(
                model, processor, batch_records, device=device
            )
            atomic_write_new_json(
                path,
                {
                    "schema_version": 1,
                    "config_sha256": config_sha256,
                    "selected_visual_sha256": checkpoint_sha256,
                    "parent_result_sha256": parent_result_sha256,
                    "subset": subset,
                    "batch_index": batch_index,
                    "pair_ids": [record.pair_id for record in batch_records],
                    "outputs": batch_outputs,
                    "score_rows": batch_scores,
                },
            )
            generated_batches += 1
            marker = "GENERATED"
        else:
            batch_outputs, batch_scores = existing
            reused_batches += 1
            marker = "REUSED"
        outputs.extend(batch_outputs)
        score_rows.extend(batch_scores)
        batch_hashes[str(path.relative_to(ROOT))] = sha256(path)
        print(
            f"QWEN_REAL_CONSTRAINED_BATCH_{marker} subset={subset} "
            f"batch={batch_index + 1}/{total_batches} "
            f"records={len(outputs)}/{len(records)}",
            flush=True,
        )
    margins = [
        float(row["runner_up_margin"])
        for row in score_rows
        if row.get("runner_up_margin") is not None
    ]
    evidence: dict[str, object] = {
        "total_batches": total_batches,
        "generated_batches_this_attempt": generated_batches,
        "reused_batches_this_attempt": reused_batches,
        "mean_runner_up_margin": statistics.fmean(margins),
        "median_runner_up_margin": statistics.median(margins),
        "minimum_runner_up_margin": min(margins),
        "maximum_runner_up_margin": max(margins),
    }
    return outputs, score_rows, evidence, batch_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha = load_diagnostic_config(args.config)
    validate_diagnostic_contract(config)
    parent = config["parent_failure"]
    model_config = config["model"]
    scoring = config["scoring"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (parent, model_config, scoring, runtime)
    )
    parent_config_path = workspace_path(parent["config"])
    parent_config, parent_config_sha, _ = load_real_validation_config(
        parent_config_path
    )
    validate_real_validation_contract(parent_config)
    if parent_config_sha != parent.get("config_sha256"):
        raise RuntimeError("parent real-validation config changed before scoring")
    parent_model = parent_config["model"]
    parent_data = parent_config["data"]
    parent_evaluation = parent_config["evaluation"]
    parent_gates = parent_config["gates"]
    assert all(
        isinstance(section, dict)
        for section in (parent_model, parent_data, parent_evaluation, parent_gates)
    )
    output_dir = workspace_path(runtime["output_dir"])
    existing = _completed_result(
        output_dir, config_sha, str(model_config["selected_visual_sha256"])
    )
    if existing is not None:
        return existing

    preflight = _verified_full_preflight(config, config_sha)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("constrained diagnostic requires exactly one visible GPU")
    seed_everything(int(runtime["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    device = torch.device("cuda:0")

    manifest_path = workspace_path(parent_data["manifest"])
    records = load_pair_manifest(manifest_path)
    manifest_rows = _manifest_rows(manifest_path)
    rows_by_pair = {row["pair_id"]: row for row in manifest_rows}
    subset_records = {
        subset: [
            record
            for record in records
            if rows_by_pair[record.pair_id]["subset"] == subset
        ]
        for subset in parent_evaluation["subset_order"]
    }
    started = time.monotonic()
    model, processor, frozen_model_evidence = _load_frozen_model(
        parent_config, device
    )
    if processor.tokenizer.padding_side != "right":
        raise RuntimeError("canonical scoring requires right padding")
    model.config.use_cache = False
    subset_results: dict[str, object] = {}
    flat_checks: dict[str, bool] = {}
    artifact_hashes: dict[str, str] = {}
    for subset in parent_evaluation["subset_order"]:
        selected = subset_records[subset]
        print(
            f"QWEN_REAL_CONSTRAINED_SUBSET_BEGIN subset={subset} "
            f"records={len(selected)}",
            flush=True,
        )
        raw_outputs, score_rows, batch_evidence, batch_hashes = _score_subset(
            model,
            processor,
            selected,
            output_dir=output_dir,
            subset=subset,
            config_sha256=config_sha,
            checkpoint_sha256=str(model_config["selected_visual_sha256"]),
            parent_result_sha256=str(parent["result_sha256"]),
            batch_size=int(scoring["batch_size"]),
            device=device,
        )
        subset_gate = parent_gates["subsets"][subset]
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
        artifact_hashes.update(batch_hashes)
        for name, digest in artifacts.items():
            relative = str(
                (
                    output_dir / "evaluation" / f"{subset}_{name}.json"
                ).relative_to(ROOT)
            )
            artifact_hashes[relative] = digest
        subset_results[subset] = {
            "records": len(selected),
            "metrics": metrics,
            "diagnostic_threshold_checks": checks,
            "clears_original_subset_thresholds": all(checks.values()),
            "batch_evidence": batch_evidence,
            "score_rows": len(score_rows),
            "artifact_sha256": {**batch_hashes, **artifacts},
        }
        print(
            f"QWEN_REAL_CONSTRAINED_SUBSET_END subset={subset} "
            f"severity_accuracy={float(metrics['severity_accuracy']):.6f} "
            f"within_one={float(metrics['within_one_level_accuracy']):.6f} "
            f"clears_original_thresholds={all(checks.values())}",
            flush=True,
        )

    diagnostic_pass = all(flat_checks.values())
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if diagnostic_pass else "failed",
        "scope": config["scope"],
        "interpretation": config["interpretation"],
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "original_gate_remains_failed": True,
        "independent_validation_claim_allowed": False,
        "official_foundir_used": False,
        "original_failed_predictions_consulted_during_method_selection": True,
        "labels_changed_after_failure": False,
        "thresholds_changed_after_failure": False,
        "model_weights_changed_after_failure": False,
        "parent_failure": preflight["parent_failure"],
        "selected_checkpoint_step": model_config["selected_checkpoint_step"],
        "selected_visual_sha256": model_config["selected_visual_sha256"],
        "model": frozen_model_evidence,
        "scoring": scoring,
        "manifest_sha256": parent_data["manifest_sha256"],
        "metadata_sha256": parent_data["metadata_sha256"],
        "annotation_kind": "reference_based_proxy_not_human_ground_truth",
        "annotation_warning": (
            "Project levels remain reference-based proxy labels, not human or "
            "official SICE/LOL-v2 severity ground truth."
        ),
        "gpu_count": 1,
        "preflight_sha256": sha256(output_dir / "preflight_full.json"),
        "runtime": preflight["runtime"],
        "subsets": subset_results,
        "gate_checks": flat_checks,
        "automatic_gate_pass": diagnostic_pass,
        "diagnostic_clears_original_thresholds": diagnostic_pass,
        "fresh_independent_real_holdout_required_before_diffusion_claims": True,
        "artifact_sha256": artifact_hashes,
    }
    atomic_write_new_json(output_dir / "result_summary.json", summary)
    elapsed = time.monotonic() - started
    print(
        (
            "QWEN_REAL_CONSTRAINED_DIAGNOSTIC_PASS"
            if diagnostic_pass
            else "QWEN_REAL_CONSTRAINED_DIAGNOSTIC_FAILED"
        ),
        json.dumps(flat_checks, sort_keys=True),
        f"elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return 0 if diagnostic_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
