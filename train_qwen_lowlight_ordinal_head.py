#!/usr/bin/env python3
"""Train and validate a small ordinal head on frozen Qwen-VL image embeddings."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.degradation_assessment import (
    AssessmentProtocol,
    assessment_gate,
    canonical_condition_json,
    evaluate_severity,
    fit_scalar_centroids,
    parse_protocol,
    predict_scalar_centroid,
    record_bundle_sha256,
    require_task_ready,
    strict_condition_payload,
    validate_source_isolation,
)
from lowlight_restoration.qwen_ordinal import (
    OrdinalSeverityHead,
    decode_ordinal_logits,
    monotonic_penalty,
    ordinal_targets,
    select_complete_source_sequences,
)
from predict_qwen_lowlight import (
    atomic_write_new,
    load_config as load_base_config,
    sha256,
    validate_model_files,
    workspace_path,
)
from predict_qwen_lowlight_fewshot import (
    checked_image,
    load_v2_config,
    manifest_rows,
    select_unique_source_records,
)

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> tuple[dict[str, object], str, dict[str, object]]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 3:
        raise ValueError("bad Qwen ordinal-head config")
    required = (
        "base_model_config",
        "frozen_validation_config",
        "head_selection_protocol",
        "final_validation_protocol",
        "data",
        "features",
        "head",
        "runtime",
    )
    for section in required:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"missing ordinal config section: {section}")
    base_reference = config["base_model_config"]
    validation_reference = config["frozen_validation_config"]
    assert isinstance(base_reference, dict) and isinstance(validation_reference, dict)
    base_path = workspace_path(base_reference.get("path"))
    validation_path = workspace_path(validation_reference.get("path"))
    if sha256(base_path) != base_reference.get("sha256"):
        raise RuntimeError("frozen base model config hash mismatch")
    if sha256(validation_path) != validation_reference.get("sha256"):
        raise RuntimeError("frozen validation config hash mismatch")
    base_config, base_hash = load_base_config(base_path)
    validation_config, validation_hash, _ = load_v2_config(validation_path)
    if base_hash != base_reference["sha256"] or validation_hash != validation_reference["sha256"]:
        raise RuntimeError("referenced config parser hash mismatch")
    selection_protocol = parse_protocol(config["head_selection_protocol"])
    final_protocol = parse_protocol(config["final_validation_protocol"])
    require_task_ready(selection_protocol.degradation)
    require_task_ready(final_protocol.degradation)
    if (
        selection_protocol.degradation != "low_light"
        or final_protocol.degradation != "low_light"
        or selection_protocol.calibration_mode != "frozen_visual_head"
        or final_protocol.calibration_mode != "frozen_visual_head"
    ):
        raise RuntimeError("ordinal runner is restricted to calibrated Lowlight")
    data = config["data"]
    features = config["features"]
    head = config["head"]
    runtime = config["runtime"]
    assert all(isinstance(value, dict) for value in (data, features, head, runtime))
    if int(data["head_train_sources"]) + int(data["head_selection_sources"]) != int(
        data["selected_train_sources"]
    ):
        raise ValueError("ordinal train/selection source counts do not add up")
    frozen_data = validation_config["data"]
    assert isinstance(frozen_data, dict)
    if (
        data.get("manifest") != frozen_data.get("manifest")
        or data.get("manifest_sha256") != frozen_data.get("manifest_sha256")
        or int(data["final_validation_samples"]) != int(frozen_data["validation_samples"])
        or int(data["final_validation_sources"]) != int(frozen_data["expected_validation_sources"])
        or int(data["final_validation_selection_seed"])
        != int(frozen_data["validation_selection_seed"])
    ):
        raise ValueError("ordinal final validation does not match the frozen v2 selection")
    if "validation" not in selection_protocol.protected_splits:
        raise ValueError("head selection protocol must protect final validation")
    if (
        final_protocol.calibration_split != "train"
        or final_protocol.validation_split != "validation"
        or set(final_protocol.protected_splits) != {"test", "foundir_test"}
    ):
        raise ValueError("bad final validation split policy")
    if int(data["records_per_source"]) != 5:
        raise ValueError("ordinal calibration requires complete L1-L5 sequences")
    if features.get("pooling") != "token_mean_and_std":
        raise ValueError("unsupported frozen feature pooling")
    if int(features["feature_dim"]) != 2 * int(features["embedding_dim"]):
        raise ValueError("feature dimension does not match mean+std pooling")
    expected_parameters = int(features["feature_dim"]) * 4 + 4
    if int(head["expected_trainable_parameters"]) != expected_parameters:
        raise ValueError("ordinal parameter count mismatch")
    if head.get("type") != "coral_linear":
        raise ValueError("unsupported ordinal head type")
    positive_integer_fields = (
        (data, "selected_train_sources"),
        (data, "head_train_sources"),
        (data, "head_selection_sources"),
        (features, "extraction_batch_size"),
        (features, "cache_schema_version"),
        (head, "epochs"),
        (head, "batch_size"),
        (head, "early_stopping_patience"),
    )
    if any(int(section[name]) <= 0 for section, name in positive_integer_fields):
        raise ValueError("ordinal counts and training durations must be positive")
    if (
        float(head["learning_rate"]) <= 0
        or float(head["weight_decay"]) < 0
        or float(head["monotonic_penalty_weight"]) < 0
        or int(head["seed"]) < 0
    ):
        raise ValueError("bad ordinal optimizer or seed configuration")
    if runtime.get("offline") is not True or runtime.get("device") != "cuda":
        raise ValueError("ordinal run must use the frozen offline CUDA path")
    return config, sha256(path), base_config


def flatten(sequences: Sequence[Sequence[PairRecord]]) -> list[PairRecord]:
    return [record for sequence in sequences for record in sequence]


def git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def feature_metadata(
    record: PairRecord,
    *,
    config_sha: str,
    model_revision: str,
    feature_dim: int,
    schema_version: int,
    extractor_sha256: str,
) -> dict[str, str]:
    return {
        "schema_version": str(schema_version),
        "pair_id": record.pair_id,
        "source_id": record.source_id,
        "level": str(record.level_index),
        "split": record.split,
        "degraded_sha256": record.degraded_sha256,
        "config_sha256": config_sha,
        "model_revision": model_revision,
        "pooling": "token_mean_and_std",
        "feature_dim": str(feature_dim),
        "extractor_sha256": extractor_sha256,
    }


def load_cached_feature(path: Path, expected: Mapping[str, str], feature_dim: int):
    import torch

    if not path.exists():
        return None
    with safe_open(path, framework="pt", device="cpu") as handle:
        if handle.metadata() != dict(expected) or set(handle.keys()) != {"feature"}:
            raise RuntimeError(f"cached feature metadata differs: {path}")
        feature = handle.get_tensor("feature")
    if feature.shape != (feature_dim,) or feature.dtype != torch.float32:
        raise RuntimeError(f"cached feature shape/dtype differs: {path}")
    if not bool(torch.isfinite(feature).all()):
        raise RuntimeError(f"cached feature is non-finite: {path}")
    return feature


def save_feature_new(path: Path, feature, metadata: Mapping[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_cached_feature(path, metadata, int(feature.numel()))
    if existing is not None:
        import torch

        if not torch.equal(existing, feature.cpu()):
            raise RuntimeError(f"existing cached feature differs: {path}")
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise RuntimeError(f"stale feature partial exists: {temporary}")
    save_file({"feature": feature.cpu().float().contiguous()}, temporary, metadata=dict(metadata))
    try:
        os.link(temporary, path)
    except FileExistsError:
        realized = load_cached_feature(path, metadata, int(feature.numel()))
        if realized is None or not bool((realized == feature.cpu()).all()):
            raise RuntimeError(f"racing cached feature differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def pooled_feature(embedding):
    import torch

    if embedding.ndim != 2 or embedding.shape[1] != 3584:
        raise RuntimeError(f"unexpected Qwen image embedding shape: {tuple(embedding.shape)}")
    pooled = torch.cat(
        (embedding.float().mean(dim=0), embedding.float().std(dim=0, unbiased=False)), dim=0
    )
    if pooled.shape != (7168,) or not bool(torch.isfinite(pooled).all()):
        raise RuntimeError("bad pooled Qwen image feature")
    return pooled.cpu()


def extract_features(
    records: Sequence[PairRecord],
    *,
    model,
    processor,
    row_luminance: Mapping[str, float],
    feature_dir: Path,
    config_sha: str,
    model_revision: str,
    feature_dim: int,
    schema_version: int,
    batch_size: int,
    extractor_sha256: str,
):
    import torch

    realized: dict[str, torch.Tensor] = {}
    missing: list[PairRecord] = []
    for record in records:
        path = feature_dir / f"{record.pair_id}.safetensors"
        metadata = feature_metadata(
            record,
            config_sha=config_sha,
            model_revision=model_revision,
            feature_dim=feature_dim,
            schema_version=schema_version,
            extractor_sha256=extractor_sha256,
        )
        cached = load_cached_feature(path, metadata, feature_dim)
        if cached is None:
            missing.append(record)
        else:
            realized[record.pair_id] = cached
    if realized:
        print(f"QWEN_ORDINAL_FEATURES_REUSED count={len(realized)}", flush=True)
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        images: list[Image.Image] = [checked_image(record, row_luminance) for record in batch]
        inputs = processor.image_processor(images=images, return_tensors="pt")
        if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
            raise RuntimeError("Qwen processor omitted visual tensors")
        if inputs.image_grid_thw.shape[0] != len(batch):
            raise RuntimeError("Qwen processor changed extraction batch image count")
        with torch.inference_mode():
            embeddings = model.get_image_features(
                inputs.pixel_values.cuda(), inputs.image_grid_thw.cuda()
            )
        if len(embeddings) != len(batch):
            raise RuntimeError("Qwen returned the wrong feature batch length")
        for record, embedding in zip(batch, embeddings, strict=True):
            feature = pooled_feature(embedding)
            path = feature_dir / f"{record.pair_id}.safetensors"
            metadata = feature_metadata(
                record,
                config_sha=config_sha,
                model_revision=model_revision,
                feature_dim=feature_dim,
                schema_version=schema_version,
                extractor_sha256=extractor_sha256,
            )
            save_feature_new(path, feature, metadata)
            realized[record.pair_id] = feature
        print(
            f"QWEN_ORDINAL_FEATURE_PROGRESS {min(start + len(batch), len(missing))}/{len(missing)}",
            flush=True,
        )
    return torch.stack([realized[record.pair_id] for record in records])


def prediction_metrics(levels, predictions) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = [
        {
            "true_level": int(truth),
            "predicted_level": int(prediction),
            "valid_structured_output": True,
        }
        for truth, prediction in zip(levels.tolist(), predictions.tolist(), strict=True)
    ]
    return rows, evaluate_severity(rows)


def scalar_baseline_metrics(
    fit_records: Sequence[PairRecord],
    evaluation_records: Sequence[PairRecord],
    row_luminance: Mapping[str, float],
) -> tuple[dict[int, float], dict[str, object]]:
    centroids = fit_scalar_centroids(
        (record.level_index, row_luminance[record.pair_id]) for record in fit_records
    )
    rows = [
        {
            "true_level": record.level_index,
            "predicted_level": predict_scalar_centroid(
                row_luminance[record.pair_id], centroids
            ),
            "valid_structured_output": True,
        }
        for record in evaluation_records
    ]
    return centroids, evaluate_severity(rows)


def train_head(
    train_features,
    train_levels,
    selection_features,
    selection_levels,
    *,
    head_config: Mapping[str, object],
    selection_protocol: AssessmentProtocol,
    selection_baseline: Mapping[str, object],
):
    import torch
    from torch.nn import functional as F

    seed = int(head_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    feature_mean = train_features.mean(dim=0)
    feature_std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_x = ((train_features - feature_mean) / feature_std).cuda()
    selection_x = ((selection_features - feature_mean) / feature_std).cuda()
    train_y = train_levels.cuda()
    selection_y = selection_levels.cuda()
    head = OrdinalSeverityHead(train_x.shape[1]).cuda()
    realized_parameters = sum(parameter.numel() for parameter in head.parameters())
    if realized_parameters != int(head_config["expected_trainable_parameters"]):
        raise RuntimeError(f"ordinal trainable parameter mismatch: {realized_parameters}")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(head_config["learning_rate"]),
        weight_decay=float(head_config["weight_decay"]),
    )
    best = None
    stale = 0
    batch_size = int(head_config["batch_size"])
    for epoch in range(1, int(head_config["epochs"]) + 1):
        head.train()
        generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
        order = torch.randperm(train_x.shape[0], generator=generator)
        loss_sum = 0.0
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size].cuda()
            logits = head(train_x[indexes])
            loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets(train_y[indexes]))
            loss = loss + float(head_config["monotonic_penalty_weight"]) * monotonic_penalty(logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indexes)
        head.eval()
        with torch.inference_mode():
            selection_logits = head(selection_x)
            selection_predictions = decode_ordinal_logits(selection_logits).cpu()
        _, metrics = prediction_metrics(selection_y.cpu(), selection_predictions)
        gate = assessment_gate(metrics, selection_baseline, selection_protocol)
        recalls = [float(metrics["per_level"][str(level)]["recall"]) for level in range(1, 6)]  # type: ignore[index]
        key = (
            int(gate["automatic_gate_pass"]),
            int(gate["all_predicted_levels_pass"]),
            int(gate["minimum_per_level_recall_pass"]),
            float(metrics["severity_accuracy"]),
            -float(metrics["severity_mean_absolute_error"]),
            min(recalls),
        )
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "train_loss": loss_sum / len(order),
                "state": {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()},
                "metrics": metrics,
                "gate": gate,
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or gate["automatic_gate_pass"]:
            print(
                "QWEN_ORDINAL_TRAIN "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "loss": loss_sum / len(order),
                        "selection_accuracy": metrics["severity_accuracy"],
                        "selection_mae": metrics["severity_mean_absolute_error"],
                        "selection_gate_pass": gate["automatic_gate_pass"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if stale >= int(head_config["early_stopping_patience"]):
            break
    if best is None:
        raise RuntimeError("ordinal training produced no checkpoint")
    head.load_state_dict(best["state"])
    return head, feature_mean, feature_std, best


def save_head_new_or_equal(path: Path, tensors: Mapping[str, object], metadata: Mapping[str, str]) -> None:
    import torch

    concrete = {name: value.detach().cpu().float().contiguous() for name, value in tensors.items()}  # type: ignore[union-attr]
    if path.exists():
        with safe_open(path, framework="pt", device="cpu") as handle:
            if handle.metadata() != dict(metadata) or set(handle.keys()) != set(concrete):
                raise RuntimeError(f"existing ordinal checkpoint metadata differs: {path}")
            for name, value in concrete.items():
                if not torch.equal(handle.get_tensor(name), value):
                    raise RuntimeError(f"existing ordinal checkpoint tensor differs: {name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(concrete, path, metadata=dict(metadata))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha, base_config = load_config(args.config)
    data_config = config["data"]
    feature_config = config["features"]
    head_config = config["head"]
    runtime_config = config["runtime"]
    assert all(isinstance(value, dict) for value in (data_config, feature_config, head_config, runtime_config))
    selection_protocol = parse_protocol(config["head_selection_protocol"])
    final_protocol = parse_protocol(config["final_validation_protocol"])
    required_cublas = str(runtime_config["required_cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != required_cublas:
        raise SystemExit(f"CUBLAS_WORKSPACE_CONFIG_MUST_EQUAL_{required_cublas}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise SystemExit("CUDA_NOT_AVAILABLE")
    random.seed(int(head_config["seed"]))
    np.random.seed(int(head_config["seed"]))
    torch.manual_seed(int(head_config["seed"]))
    torch.cuda.manual_seed_all(int(head_config["seed"]))
    torch.use_deterministic_algorithms(True)

    manifest = workspace_path(data_config["manifest"])
    if sha256(manifest) != data_config["manifest_sha256"]:
        raise RuntimeError("ordinal manifest hash mismatch")
    records = load_pair_manifest(manifest)
    sequences = select_complete_source_sequences(
        records,
        split="train",
        source_count=int(data_config["selected_train_sources"]),
        seed=int(data_config["source_selection_seed"]),
    )
    train_sequences = sequences[: int(data_config["head_train_sources"])]
    selection_sequences = sequences[int(data_config["head_train_sources"]) :]
    train_records = flatten(train_sequences)
    selection_records = flatten(selection_sequences)
    validate_source_isolation(train_records, selection_records)
    if any(record.split != "train" for record in [*train_records, *selection_records]):
        raise RuntimeError("ordinal head calibration escaped the train split")
    rows = manifest_rows(manifest)
    row_luminance = {row["pair_id"]: float(row["final_luminance"]) for row in rows}
    selection_centroids, selection_baseline = scalar_baseline_metrics(
        train_records, selection_records, row_luminance
    )
    model_config = base_config["model"]
    processor_config = base_config["processor"]
    assert isinstance(model_config, dict) and isinstance(processor_config, dict)
    model_hashes = validate_model_files(base_config)
    revision = str(model_config["revision"])
    local_model = workspace_path(model_config["local_path"])
    processor = AutoProcessor.from_pretrained(
        local_model,
        local_files_only=True,
        use_fast=bool(processor_config["use_fast"]),
        min_pixels=int(processor_config["min_pixels"]),
        max_pixels=int(processor_config["max_pixels"]),
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        local_model,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation=str(model_config["attention_implementation"]),
    ).cuda()
    model.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen must remain frozen for ordinal-head training")

    output_dir = workspace_path(runtime_config["output_dir"])
    feature_dir = output_dir / "features"
    feature_kwargs = {
        "model": model,
        "processor": processor,
        "row_luminance": row_luminance,
        "feature_dir": feature_dir,
        "config_sha": config_sha,
        "model_revision": revision,
        "feature_dim": int(feature_config["feature_dim"]),
        "schema_version": int(feature_config["cache_schema_version"]),
        "batch_size": int(feature_config["extraction_batch_size"]),
        "extractor_sha256": sha256(Path(__file__)),
    }
    print(
        "QWEN_ORDINAL_CUDA_READY "
        + json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "train_records": len(train_records),
                "selection_records": len(selection_records),
                "train_bundle_sha256": record_bundle_sha256(train_records),
                "selection_bundle_sha256": record_bundle_sha256(selection_records),
                "selection_baseline_accuracy": selection_baseline["severity_accuracy"],
                "selection_baseline_mae": selection_baseline["severity_mean_absolute_error"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    train_features = extract_features(train_records, **feature_kwargs)
    selection_features = extract_features(selection_records, **feature_kwargs)
    train_levels = torch.tensor([record.level_index for record in train_records], dtype=torch.int64)
    selection_levels = torch.tensor(
        [record.level_index for record in selection_records], dtype=torch.int64
    )
    head, feature_mean, feature_std, best = train_head(
        train_features,
        train_levels,
        selection_features,
        selection_levels,
        head_config=head_config,
        selection_protocol=selection_protocol,
        selection_baseline=selection_baseline,
    )
    checkpoint = output_dir / "ordinal_head.safetensors"
    checkpoint_metadata = {
        "schema_version": "1",
        "config_sha256": config_sha,
        "model_revision": revision,
        "train_bundle_sha256": record_bundle_sha256(train_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "best_epoch": str(best["epoch"]),
    }
    save_head_new_or_equal(
        checkpoint,
        {
            **best["state"],
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        checkpoint_metadata,
    )
    selection_summary = {
        "schema_version": 1,
        "status": "passed" if best["gate"]["automatic_gate_pass"] else "failed",
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": revision,
        "model_file_sha256": model_hashes,
        "train_bundle_sha256": record_bundle_sha256(train_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "best_epoch": best["epoch"],
        "best_train_loss": best["train_loss"],
        "metrics": best["metrics"],
        "gate": best["gate"],
        "baseline": {
            "centroids": {str(key): value for key, value in selection_centroids.items()},
            "metrics": selection_baseline,
        },
        "validation_split_touched": False,
        "test_split_used": False,
        "official_foundir_used": False,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
    }
    atomic_write_new(output_dir / "selection_metrics.json", selection_summary)
    print(
        "QWEN_ORDINAL_SELECTION_GATE " + json.dumps(best["gate"], sort_keys=True), flush=True
    )
    if not best["gate"]["automatic_gate_pass"]:
        raise SystemExit(2)

    # The fixed validation records and images are selected only after the
    # train-source selection gate has passed and the head checkpoint is frozen.
    final_validation_records = select_unique_source_records(
        records,
        split="validation",
        sample_count=int(data_config["final_validation_samples"]),
        seed=int(data_config["final_validation_selection_seed"]),
    )
    if len({record.source_id for record in final_validation_records}) != int(
        data_config["final_validation_sources"]
    ):
        raise RuntimeError("ordinal final validation sources are not unique")
    validate_source_isolation([*train_records, *selection_records], final_validation_records)
    all_train_records = [record for record in records if record.split == "train"]
    final_centroids, final_baseline = scalar_baseline_metrics(
        all_train_records, final_validation_records, row_luminance
    )
    if (
        abs(float(final_baseline["severity_accuracy"]) - 0.64) > 1e-12
        or abs(float(final_baseline["severity_mean_absolute_error"]) - 0.40) > 1e-12
    ):
        raise RuntimeError("frozen final validation baseline changed")
    validation_features = extract_features(final_validation_records, **feature_kwargs)
    head.eval()
    with torch.inference_mode():
        validation_x = ((validation_features - feature_mean) / feature_std).cuda()
        predictions = decode_ordinal_logits(head(validation_x)).cpu()
    validation_levels = torch.tensor(
        [record.level_index for record in final_validation_records], dtype=torch.int64
    )
    metric_rows, metrics = prediction_metrics(validation_levels, predictions)
    prediction_records: list[dict[str, object]] = []
    for record, prediction, metric_row in zip(
        final_validation_records, predictions.tolist(), metric_rows, strict=True
    ):
        raw_output = canonical_condition_json("low_light", int(prediction))
        parsed = strict_condition_payload(raw_output, "low_light")
        if parsed != prediction:
            raise RuntimeError("ordinal structured-condition serialization changed the prediction")
        result = {
            "schema_version": 3,
            "pair_id": record.pair_id,
            "source_id": record.source_id,
            "split": record.split,
            "degraded_sha256": record.degraded_sha256,
            **metric_row,
            "raw_output": raw_output,
            "config_sha256": config_sha,
            "model_revision": revision,
            "checkpoint_sha256": sha256(checkpoint),
        }
        atomic_write_new(output_dir / "records" / f"{record.pair_id}.json", result)
        prediction_records.append(result)
    gate = assessment_gate(metrics, final_baseline, final_protocol)
    summary = {
        "schema_version": 3,
        "status": "passed" if gate["automatic_gate_pass"] else "failed",
        "scope": "frozen_qwen_ordinal_lowlight_validation_only",
        "config": config,
        "config_sha256": config_sha,
        "git_revision": git_revision(),
        "model_revision": revision,
        "model_file_sha256": model_hashes,
        "head_trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "train_bundle_sha256": record_bundle_sha256(train_records),
        "selection_bundle_sha256": record_bundle_sha256(selection_records),
        "validation_bundle_sha256": record_bundle_sha256(final_validation_records),
        "selection": selection_summary,
        "baseline": {
            "centroids": {str(key): value for key, value in final_centroids.items()},
            "metrics": final_baseline,
        },
        "metrics": metrics,
        "gate": gate,
        "records": prediction_records,
        "test_split_used": False,
        "official_foundir_used": False,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        },
    }
    atomic_write_new(output_dir / "metrics.json", summary)
    marker = (
        "QWEN_LOWLIGHT_ORDINAL_VALIDATION_PASS"
        if gate["automatic_gate_pass"]
        else "QWEN_LOWLIGHT_ORDINAL_VALIDATION_FAILED"
    )
    print(marker + " " + json.dumps(gate, sort_keys=True), flush=True)
    if not gate["automatic_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
