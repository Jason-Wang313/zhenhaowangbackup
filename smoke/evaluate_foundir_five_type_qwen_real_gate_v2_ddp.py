#!/usr/bin/env python3
"""Evaluate the frozen per-type Qwen on FoundIR training images only."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.evaluate_foundir_five_type_qwen_native_patches_ddp import (  # noqa: E402
    aggregated_predictions,
    build_view_records,
    parse_levels,
)
from smoke.evaluate_foundir_five_type_qwen_training_gate import (  # noqa: E402
    calculate,
    checks,
)
from smoke.export_foundir_five_type_qwen_conditions import (  # noqa: E402
    load_manifest,
    sha256,
    write_once,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    broadcast_rank0,
    generate_outputs,
    load_model,
    setup_distributed,
    validate_runtime,
)
from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def load_contract(config_path: Path) -> tuple[dict[str, object], list[dict[str, str]]]:
    config = json.loads(config_path.read_text())
    if (
        config.get("status") != "frozen_foundir_training_only_qwen_real_gate"
        or config.get("scope") != "inference_only_no_qwen_training"
        or config.get("qwen_training_performed") is not False
        or config.get("restoration_training_performed") is not False
        or config.get("protected_foundir_official_test_accessed") is not False
        or config.get("aggregation") != "second_highest"
    ):
        raise RuntimeError("the v2 Qwen real gate contract is not frozen safely")
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("the v2 Qwen real gate artifacts are missing")
    loaded: dict[str, object] = {}
    for name, artifact in artifacts.items():
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"invalid frozen artifact: {name}")
        path = workspace_path(str(artifact["path"]))
        if not path.is_file() or sha256(path) != artifact.get("sha256"):
            raise RuntimeError(f"frozen artifact changed: {name}")
        if path.suffix == ".json":
            loaded[str(name)] = json.loads(path.read_text())

    comparison = loaded["comparison"]
    selection = loaded["checkpoint_selection"]
    validation = loaded["synthetic_validation"]
    synthetic_test = loaded["synthetic_test"]
    adapter_sha = artifacts["adapter"]["sha256"]
    if (
        comparison.get("selected_method") != "per_type"
        or comparison.get("protected_test_accessed") is not False
        or comparison.get("foundir_used") is not False
        or selection.get("selected_adapter_sha256") != adapter_sha
        or selection.get("selection_data") != "synthetic_validation_only"
        or validation.get("status") != "passed"
        or validation.get("split") != "validation"
        or validation.get("adapter_sha256") != adapter_sha
        or synthetic_test.get("status") != "passed"
        or synthetic_test.get("split") != "test"
        or synthetic_test.get("adapter_sha256") != adapter_sha
    ):
        raise RuntimeError("the selected Qwen did not pass frozen synthetic selection")

    qwen_config = loaded["qwen_config"]
    if (
        qwen_config.get("prompt", {}).get("target_schema")
        != "per_type_dialogue_l1_absent_l2_l5_v1"
        or qwen_config.get("runtime", {}).get("output_root")
        != "checkpoints/qwen_multidegradation_100k_rank64_per_type_v2"
    ):
        raise RuntimeError("the gate is not bound to the selected per-type Qwen")

    manifest_path = workspace_path(str(artifacts["manifest"]["path"]))
    records = load_manifest(manifest_path)
    if (
        len(records) != int(config["records"])
        or len({row["record_id"] for row in records}) != len(records)
        or any(row.get("split") != "train" for row in records)
        or any("/MillionIRData/Train/" not in row.get("lq_path", "") for row in records)
        or any("/MillionIRData/Test/" in row.get("lq_path", "") for row in records)
    ):
        raise RuntimeError("the gate manifest is not the frozen FoundIR-training pilot")
    return config, records


def identity(config_path: Path, config: Mapping[str, object]) -> dict[str, object]:
    artifacts = config["artifacts"]
    return {
        "schema_version": 1,
        "config_sha256": sha256(config_path),
        "manifest_sha256": artifacts["manifest"]["sha256"],
        "qwen_config_sha256": artifacts["qwen_config"]["sha256"],
        "adapter_sha256": artifacts["adapter"]["sha256"],
        "records": int(config["records"]),
        "crop_size": int(config["crop_size"]),
        "views": list(config["views"]),
        "aggregation": config["aggregation"],
        "qwen_training_performed": False,
        "protected_foundir_official_test_accessed": False,
    }


def verify_raw(payload: Mapping[str, object], expected: Mapping[str, object]) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"v2 real-gate identity changed: {key}")
    rows = payload.get("view_predictions")
    if not isinstance(rows, list) or len(rows) != int(expected["records"]) * len(expected["views"]):
        raise RuntimeError("v2 real-gate view predictions are incomplete")


def finalize(
    config_path: Path,
    config: Mapping[str, object],
    records: list[dict[str, str]],
    raw_payload: Mapping[str, object],
) -> dict[str, object]:
    predictions = aggregated_predictions(
        raw_payload, records, method=str(config["aggregation"])
    )
    metrics = calculate(predictions)
    gate_checks = checks(metrics, config["thresholds"])
    passed = all(gate_checks.values())
    result = {
        **identity(config_path, config),
        "status": "passed" if passed else "failed",
        "automatic_gate_pass": passed,
        "scope": "authorized_foundir_training_only_type_labels",
        "metrics": metrics,
        "checks": gate_checks,
        "severity_evaluated": False,
        "severity_exclusion_reason": "FoundIR training folders provide types but no project L1-L5 truth",
        "raw_output": config["raw_output"],
        "raw_output_sha256": sha256(workspace_path(str(config["raw_output"]))),
        "restoration_training_authorized": passed,
    }
    write_once(
        workspace_path(str(config["output"])),
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    return result


def verify_result(config_path: Path) -> int:
    config, records = load_contract(config_path)
    expected = identity(config_path, config)
    raw_path = workspace_path(str(config["raw_output"]))
    result_path = workspace_path(str(config["output"]))
    if not raw_path.is_file() or not result_path.is_file():
        return 1
    raw = json.loads(raw_path.read_text())
    verify_raw(raw, expected)
    result = json.loads(result_path.read_text())
    for key, value in expected.items():
        if result.get(key) != value:
            return 1
    if result.get("raw_output_sha256") != sha256(raw_path):
        return 1
    recomputed = finalize(config_path, config, records, raw)
    if result != recomputed:
        return 1
    return 0 if result.get("automatic_gate_pass") is True else 2


def run(config_path: Path) -> int:
    config, records = load_contract(config_path)
    expected = identity(config_path, config)
    raw_path = workspace_path(str(config["raw_output"]))
    result_path = workspace_path(str(config["output"]))
    if result_path.exists():
        return verify_result(config_path)
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        verify_raw(raw, expected)
        return 0 if finalize(config_path, config, records, raw)["automatic_gate_pass"] else 2

    initialized = False
    try:
        qwen_config_path = workspace_path(str(config["artifacts"]["qwen_config"]["path"]))
        qwen_config = json.loads(qwen_config_path.read_text())
        rank, world, local_rank, device = setup_distributed(qwen_config)
        initialized = True
        if world != 8:
            raise RuntimeError("the v2 real gate requires exactly eight GPUs")
        view_records = broadcast_rank0(
            rank,
            lambda: build_view_records(
                records, crop_size=int(config["crop_size"]), views=list(config["views"])
            ),
        )
        validate_runtime(qwen_config, local_rank)
        model, processor, _ = load_model(qwen_config, variant="merger_trainable", device=device)
        load_qwen_adapter(model, workspace_path(str(config["artifacts"]["adapter"]["path"])))
        raw_outputs = generate_outputs(
            model, processor, view_records, qwen_config, rank=rank, world=world, device=device
        )

        def publish() -> dict[str, object]:
            rows = []
            for record, raw in zip(view_records, raw_outputs, strict=True):
                valid, levels = parse_levels(raw, "per_type")
                rows.append(
                    {
                        "parent_record_id": record["parent_record_id"],
                        "category": record["category"],
                        "composition_size": int(record["composition_size"]),
                        "view": record["view"],
                        "crop_box": record["crop_box"],
                        "raw_output": raw,
                        "valid": valid,
                        "levels": levels,
                    }
                )
            payload = {**expected, "status": "complete", "view_predictions": rows}
            write_once(raw_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            verify_raw(payload, expected)
            return payload

        raw_payload = broadcast_rank0(rank, publish)
        result = broadcast_rank0(rank, lambda: finalize(config_path, config, records, raw_payload))
        if rank == 0:
            print("FOUNDIR_QWEN_REAL_GATE_V2 " + json.dumps(result, sort_keys=True), flush=True)
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        return 0 if result["automatic_gate_pass"] else 2
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_qwen_real_gate_v2.json",
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    if args.check_only and args.verify_existing:
        raise ValueError("choose at most one verification mode")
    if args.check_only:
        config, records = load_contract(config_path)
        print(
            "FOUNDIR_QWEN_REAL_GATE_V2_CHECK_PASS "
            + json.dumps({**identity(config_path, config), "loaded_records": len(records)}, sort_keys=True),
            flush=True,
        )
        return 0
    if args.verify_existing:
        return verify_result(config_path)
    return run(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
