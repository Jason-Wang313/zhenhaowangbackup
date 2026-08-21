#!/usr/bin/env python3
"""Evaluate frozen Qwen canonical scores on objective real exposure stacks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.data import PairRecord  # noqa: E402
from lowlight_restoration.exposure_ordering import (  # noqa: E402
    LEVELS,
    completed_exposure_result_status,
    evaluate_exposure_ordering,
    exposure_pairs,
    load_frozen_exposure_stacks,
)
from lowlight_restoration.qwen_constrained import (  # noqa: E402
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
from smoke.evaluate_qwen_lowlight_real_validation import _load_frozen_model  # noqa: E402
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


def load_config(path: Path) -> tuple[dict[str, object], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("exposure-ordering config must be a JSON object")
    return payload, sha256(path)


def _load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError("exposure-ordering manifest has no header")
        rows = list(reader)
    if not rows:
        raise RuntimeError("exposure-ordering manifest is empty")
    return rows


def validate_contract(
    config_path: Path, *, hash_visual_checkpoint: bool
) -> tuple[
    dict[str, object],
    str,
    dict[str, object],
    dict[str, tuple[object, ...]],
    list[dict[str, str]],
]:
    config, config_sha = load_config(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("status")
        != "approved_objective_relative_exposure_development_diagnostic"
        or config.get("official_foundir_used") is not False
        or config.get("model_predictions_consulted_during_data_selection") is not False
    ):
        raise RuntimeError("exposure-ordering top-level contract changed")
    model = config.get("model")
    data = config.get("data")
    scoring = config.get("scoring")
    gates = config.get("gates")
    runtime = config.get("runtime")
    if not all(
        isinstance(section, dict)
        for section in (model, data, scoring, gates, runtime)
    ):
        raise RuntimeError("exposure-ordering config sections are missing")
    assert isinstance(model, dict)
    assert isinstance(data, dict)
    assert isinstance(scoring, dict)
    assert isinstance(gates, dict)
    assert isinstance(runtime, dict)
    if (
        model.get("all_parameters_frozen") is not True
        or scoring.get("candidate_project_levels") != list(LEVELS)
        or scoring.get("free_form_generation_used") is not False
        or runtime.get("required_visible_device_set") != "0,1"
        or runtime.get("required_world_size") != 2
        or gates.get("passing_proves_only_relative_exposure_sensitivity") is not True
        or gates.get("downstream_causal_restoration_gate_still_required") is not True
        or data.get("absolute_project_l1_l5_labels_available") is not False
        or data.get("reusable_development_suite") is not True
    ):
        raise RuntimeError("exposure-ordering scientific contract changed")

    parent_path = workspace_path(model["base_contract"])
    if sha256(parent_path) != model["base_contract_sha256"]:
        raise RuntimeError("base real-validation model contract changed")
    parent_config, parent_sha, _ = load_real_validation_config(parent_path)
    validate_real_validation_contract(parent_config)
    if parent_sha != model["base_contract_sha256"]:
        raise RuntimeError("base real-validation config digest differs")
    runtime_model_config = copy.deepcopy(parent_config)
    parent_model = runtime_model_config["model"]
    assert isinstance(parent_model, dict)
    parent_model.update(
        selected_checkpoint_step=model["selected_checkpoint_step"],
        selected_visual_checkpoint=model["selected_visual_checkpoint"],
        selected_visual_sha256=model["selected_visual_sha256"],
    )
    visual_path = workspace_path(model["selected_visual_checkpoint"])
    if not visual_path.is_file():
        raise RuntimeError("selected Qwen visual checkpoint is missing")
    if hash_visual_checkpoint and sha256(visual_path) != model["selected_visual_sha256"]:
        raise RuntimeError("selected Qwen visual checkpoint identity changed")

    manifest_path = workspace_path(data["manifest"])
    metadata_path = workspace_path(data["metadata"])
    if sha256(manifest_path) != data["manifest_sha256"]:
        raise RuntimeError("exposure-ordering manifest identity changed")
    if sha256(metadata_path) != data["metadata_sha256"]:
        raise RuntimeError("exposure-ordering metadata identity changed")
    metadata = read_json(metadata_path)
    metadata_gate = metadata.get("evaluation_contract")
    threshold_keys = (
        "primary_pair_gap_ev",
        "minimum_primary_continuous_concordance",
        "minimum_primary_hard_non_inversion_rate",
        "minimum_median_stack_spearman",
        "minimum_positive_stack_spearman_fraction",
        "minimum_pooled_expected_level_span",
    )
    if (
        metadata.get("status")
        != "frozen_real_exposure_ordering_development_suite"
        or metadata.get("official_foundir_used") is not False
        or metadata.get("model_inference_performed_during_selection") is not False
        or metadata.get("absolute_project_l1_l5_labels_available") is not False
        or not isinstance(metadata_gate, dict)
        or any(metadata_gate.get(key) != gates.get(key) for key in threshold_keys)
        or metadata.get("output_csv_sha256") != data["manifest_sha256"]
    ):
        raise RuntimeError("exposure-ordering metadata contract changed")

    rows = _load_manifest_rows(manifest_path)
    stacks, image_sha_by_id = load_frozen_exposure_stacks(rows)
    if (
        len(stacks) != data["stacks"]
        or len(rows) != data["images"]
        or len(image_sha_by_id) != data["images"]
        or sum(
            len(exposure_pairs(stack, minimum_ev_gap=1.0))
            for stack in stacks.values()
        )
        != data["primary_pairs_minimum_ev_1"]
    ):
        raise RuntimeError("exposure-ordering manifest counts changed")
    for row in rows:
        path = workspace_path(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"exposure image is missing or changed: {row['image_id']}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"exposure image SHA-256 changed: {row['image_id']}")
        with Image.open(path) as image:
            if (
                image.size != (int(row["width"]), int(row["height"]))
                or image.convert("RGB").mode != row["mode"]
            ):
                raise RuntimeError(f"exposure image metadata changed: {row['image_id']}")
    return config, config_sha, runtime_model_config, stacks, rows


def _records(rows: Sequence[dict[str, str]], seed: int) -> list[PairRecord]:
    records: list[PairRecord] = []
    for row in rows:
        path = workspace_path(row["path"])
        records.append(
            PairRecord(
                pair_id=row["image_id"],
                source_id=row["stack_id"],
                clean_path=path,
                degraded_path=path,
                project_level="L1",
                seed=seed,
                split="validation",
                clean_sha256=row["sha256"],
                degraded_sha256=row["sha256"],
            )
        )
    return records


def _score_batch(
    model: torch.nn.Module,
    processor: Any,
    records: Sequence[PairRecord],
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    images = open_degraded_images(records)
    scores_by_row: list[dict[str, float]] = [dict() for _ in records]
    token_counts: dict[str, int] = {}
    try:
        for level in LEVELS:
            candidates = [
                replace(record, project_level=f"L{level}") for record in records
            ]
            batch, expected_counts = prepare_supervised_batch(
                processor, candidates, images
            )
            labels = batch.pop("labels").to(device)
            inputs = {
                name: value.to(device)
                for name, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            with torch.inference_mode():
                output = model(**inputs, use_cache=False, return_dict=True)
                means, realized_counts = mean_shifted_target_token_nll(
                    output.logits, labels
                )
            if realized_counts != expected_counts or len(means) != len(records):
                raise RuntimeError("canonical target-token scoring shape changed")
            if len(set(realized_counts)) != 1:
                raise RuntimeError("canonical target token count varies within level")
            token_counts[str(level)] = realized_counts[0]
            for index, mean_nll in enumerate(means):
                scores_by_row[index][str(level)] = mean_nll
            del output, inputs, labels, batch
    finally:
        for image in images:
            image.close()

    result: list[dict[str, object]] = []
    for record, scores in zip(records, scores_by_row, strict=True):
        selected, selected_nll, margin = select_minimum_mean_nll(scores)
        result.append(
            {
                "image_id": record.pair_id,
                "candidate_mean_nll": scores,
                "candidate_target_token_counts": token_counts,
                "selected_level": selected,
                "selected_target": canonical_target(selected),
                "selected_mean_nll": selected_nll,
                "runner_up_margin": margin,
            }
        )
    return result


def _batch_path(output_dir: Path, rank: int, batch_index: int) -> Path:
    return output_dir / "score_batches" / f"rank_{rank:02d}" / f"batch_{batch_index:04d}.json"


def _read_reusable_batch(
    path: Path,
    *,
    config_sha: str,
    manifest_sha: str,
    checkpoint_sha: str,
    rank: int,
    records: Sequence[PairRecord],
) -> list[dict[str, object]] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    rows = payload.get("score_rows")
    pair_ids = [record.pair_id for record in records]
    if (
        payload.get("config_sha256") != config_sha
        or payload.get("manifest_sha256") != manifest_sha
        or payload.get("selected_visual_sha256") != checkpoint_sha
        or payload.get("rank") != rank
        or payload.get("pair_ids") != pair_ids
        or not isinstance(rows, list)
        or len(rows) != len(records)
    ):
        raise RuntimeError(f"existing exposure score batch differs: {path}")
    for record, row in zip(records, rows, strict=True):
        if not isinstance(row, dict) or row.get("image_id") != record.pair_id:
            raise RuntimeError(f"existing exposure score row differs: {path}")
        scores = row.get("candidate_mean_nll")
        if not isinstance(scores, dict):
            raise RuntimeError(f"existing exposure scores are missing: {path}")
        selected, selected_nll, margin = select_minimum_mean_nll(scores)
        if (
            row.get("selected_level") != selected
            or row.get("selected_target") != canonical_target(selected)
            or float(row.get("selected_mean_nll", -1)) != selected_nll
            or row.get("runner_up_margin") != margin
        ):
            raise RuntimeError(f"existing exposure selection differs: {path}")
    return [dict(row) for row in rows]


def _completed_summary(
    output_dir: Path, *, config_sha: str, manifest_sha: str, checkpoint_sha: str
) -> dict[str, object] | None:
    path = output_dir / "result_summary.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    try:
        completed_exposure_result_status(
            payload,
            config_sha256=config_sha,
            manifest_sha256=manifest_sha,
            checkpoint_sha256=checkpoint_sha,
        )
    except ValueError as exc:
        raise RuntimeError("existing exposure-ordering summary differs") from exc
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--skip-visual-hash", action="store_true")
    args = parser.parse_args()
    config, config_sha, model_contract, stacks, rows = validate_contract(
        args.config.resolve(), hash_visual_checkpoint=not args.skip_visual_hash
    )
    model_config = config["model"]
    data = config["data"]
    scoring = config["scoring"]
    gates = config["gates"]
    runtime = config["runtime"]
    assert all(
        isinstance(section, dict)
        for section in (model_config, data, scoring, gates, runtime)
    )
    if args.contract_only:
        print(
            "QWEN_REAL_EXPOSURE_ORDERING_CONTRACT_PASS "
            f"stacks={len(stacks)} images={len(rows)} visual_hash_checked={not args.skip_visual_hash}"
        )
        return 0

    if os.environ.get("CUDA_VISIBLE_DEVICES") != runtime["required_visible_device_set"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be literal 0,1")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != runtime[
        "required_cublas_workspace_config"
    ]:
        raise RuntimeError("deterministic CUBLAS workspace configuration changed")
    import transformers

    if (
        platform.python_version() != runtime["required_python"]
        or torch.__version__ != runtime["required_pytorch"]
        or transformers.__version__ != runtime["required_transformers"]
    ):
        raise RuntimeError("validated Qwen runtime version changed")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != runtime["required_world_size"]:
        raise RuntimeError("exposure-ordering diagnostic requires two ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world_size:
        raise RuntimeError("exposure-ordering diagnostic requires two visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < int(runtime["minimum_gpu_memory_gib"]) * 1024**3:
        raise RuntimeError("selected GPU has insufficient memory")
    dist.init_process_group("nccl")
    seed_everything(int(runtime["seed"]) + rank)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    output_dir = workspace_path(runtime["output_dir"])
    if rank == 0:
        completed = _completed_summary(
            output_dir,
            config_sha=config_sha,
            manifest_sha=str(data["manifest_sha256"]),
            checkpoint_sha=str(model_config["selected_visual_sha256"]),
        )
    else:
        completed = None
    completed_flags: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(completed_flags, completed is not None)
    if any(completed_flags):
        if not all(completed_flags):
            raise RuntimeError("ranks disagree about completed exposure result")
        if rank == 0:
            assert completed is not None
            print(
                "QWEN_REAL_EXPOSURE_ORDERING_REUSED "
                f"status={completed['status']} pass={completed['automatic_gate_pass']}"
            )
        dist.destroy_process_group()
        return 0

    records = _records(rows, int(runtime["seed"]))
    selected_records = records[rank::world_size]
    model, processor, model_evidence = _load_frozen_model(model_contract, device)
    if processor.tokenizer.padding_side != "right":
        raise RuntimeError("canonical scoring requires right padding")
    model.config.use_cache = False
    batch_size = int(scoring["batch_size_per_rank"])
    for batch_index, start in enumerate(range(0, len(selected_records), batch_size)):
        batch_records = selected_records[start : start + batch_size]
        path = _batch_path(output_dir, rank, batch_index)
        score_rows = _read_reusable_batch(
            path,
            config_sha=config_sha,
            manifest_sha=str(data["manifest_sha256"]),
            checkpoint_sha=str(model_config["selected_visual_sha256"]),
            rank=rank,
            records=batch_records,
        )
        marker = "REUSED"
        if score_rows is None:
            score_rows = _score_batch(
                model, processor, batch_records, device=device
            )
            atomic_write_new_json(
                path,
                {
                    "schema_version": 1,
                    "config_sha256": config_sha,
                    "manifest_sha256": data["manifest_sha256"],
                    "selected_visual_sha256": model_config[
                        "selected_visual_sha256"
                    ],
                    "rank": rank,
                    "batch_index": batch_index,
                    "pair_ids": [record.pair_id for record in batch_records],
                    "score_rows": score_rows,
                },
            )
            marker = "GENERATED"
        print(
            f"QWEN_REAL_EXPOSURE_BATCH_{marker} rank={rank} "
            f"batch={batch_index} records={len(batch_records)}",
            flush=True,
        )
    environment = {
        "rank": rank,
        "local_rank": local_rank,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "maximum_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    environments: list[object] = [None for _ in range(world_size)]
    dist.all_gather_object(environments, environment)
    dist.barrier()

    if rank == 0:
        all_score_rows: dict[str, dict[str, object]] = {}
        artifact_hashes: dict[str, str] = {}
        for path in sorted((output_dir / "score_batches").glob("rank_*/*.json")):
            payload = read_json(path)
            if (
                payload.get("config_sha256") != config_sha
                or payload.get("manifest_sha256") != data["manifest_sha256"]
                or payload.get("selected_visual_sha256")
                != model_config["selected_visual_sha256"]
            ):
                raise RuntimeError(f"score artifact provenance differs: {path}")
            batch_rows = payload.get("score_rows")
            if not isinstance(batch_rows, list):
                raise RuntimeError(f"score artifact rows are missing: {path}")
            for score_row in batch_rows:
                if not isinstance(score_row, dict):
                    raise RuntimeError(f"score artifact row is malformed: {path}")
                image_id = score_row.get("image_id")
                if not isinstance(image_id, str) or image_id in all_score_rows:
                    raise RuntimeError(f"duplicate score identity: {image_id}")
                all_score_rows[image_id] = score_row
            artifact_hashes[str(path.relative_to(ROOT))] = sha256(path)
        image_sha_by_id = {row["image_id"]: row["sha256"] for row in rows}
        if set(all_score_rows) != set(image_sha_by_id):
            raise RuntimeError("distributed scoring did not cover the frozen manifest")
        nll_by_sha = {
            image_sha_by_id[image_id]: score_row["candidate_mean_nll"]
            for image_id, score_row in all_score_rows.items()
        }
        metrics = evaluate_exposure_ordering(
            stacks,
            nll_by_sha,
            minimum_ev_gaps=tuple(float(value) for value in scoring["minimum_ev_gaps"]),
            temperature=float(scoring["temperature"]),
        )
        primary_key = f"minimum_ev_{float(gates['primary_pair_gap_ev']):g}"
        primary = metrics["pairwise"][primary_key]
        checks = {
            "primary_continuous_concordance": float(
                primary["continuous_concordance_with_half_credit_for_ties"]
            )
            >= float(gates["minimum_primary_continuous_concordance"]),
            "primary_hard_non_inversion": float(primary["hard_level_non_inversion_rate"])
            >= float(gates["minimum_primary_hard_non_inversion_rate"]),
            "median_stack_spearman": float(metrics["median_stack_spearman"])
            >= float(gates["minimum_median_stack_spearman"]),
            "positive_stack_spearman_fraction": float(
                metrics["positive_stack_spearman_fraction"]
            )
            >= float(gates["minimum_positive_stack_spearman_fraction"]),
            "pooled_expected_level_span": float(metrics["pooled_expected_level_span"])
            >= float(gates["minimum_pooled_expected_level_span"]),
        }
        automatic_pass = all(checks.values())
        summary = {
            "schema_version": 1,
            "status": "passed" if automatic_pass else "failed",
            "scope": config["scope"],
            "interpretation": config["interpretation"],
            "config_sha256": config_sha,
            "git_revision": git_revision(),
            "manifest_sha256": data["manifest_sha256"],
            "metadata_sha256": data["metadata_sha256"],
            "selected_checkpoint_step": model_config["selected_checkpoint_step"],
            "selected_visual_sha256": model_config["selected_visual_sha256"],
            "official_foundir_used": False,
            "absolute_real_l1_l5_accuracy_claimed": False,
            "downstream_restoration_gain_claimed": False,
            "relative_exposure_evidence_only": True,
            "reusable_development_suite": True,
            "model": model_evidence,
            "scoring": scoring,
            "gates": gates,
            "metrics": metrics,
            "gate_checks": checks,
            "automatic_gate_pass": automatic_pass,
            "distributed_environment": environments,
            "artifact_sha256": artifact_hashes,
        }
        atomic_write_new_json(output_dir / "result_summary.json", summary)
        print(
            "QWEN_REAL_EXPOSURE_ORDERING_COMPLETE "
            f"status={summary['status']} pass={automatic_pass} "
            f"concordance={primary['continuous_concordance_with_half_credit_for_ties']:.6f} "
            f"median_spearman={metrics['median_stack_spearman']:.6f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
