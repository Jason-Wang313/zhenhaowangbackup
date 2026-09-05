#!/usr/bin/env python3
"""Test frozen Qwen candidates on native-resolution FoundIR training patches."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import DEGRADATION_ORDER  # noqa: E402
from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from lowlight_restoration.qwen_multidegradation import (  # noqa: E402
    FIXED_FIVE_SLOT_SCHEMA,
    parse_fixed_five_slot_target,
    parse_output,
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


def workspace_path(value: str | Path) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def verify_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"frozen diagnostic artifact changed: {path}: {actual}")


def crop_boxes(width: int, height: int, size: int) -> dict[str, tuple[int, int, int, int]]:
    if min(width, height) < size:
        raise RuntimeError(f"native patch requires at least {size} pixels: {width}x{height}")
    right = width - size
    bottom = height - size
    center_x = right // 2
    center_y = bottom // 2
    return {
        "top_left": (0, 0, size, size),
        "top_right": (right, 0, width, size),
        "center": (center_x, center_y, center_x + size, center_y + size),
        "bottom_left": (0, bottom, size, height),
        "bottom_right": (right, bottom, width, height),
    }


def load_contract(config_path: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    config = json.loads(config_path.read_text())
    if (
        config.get("status") != "frozen_training_only_scale_mismatch_diagnostic"
        or config.get("protected_foundir_test_accessed") is not False
        or config.get("restoration_training_authorized_by_this_diagnostic") is not False
    ):
        raise RuntimeError("native-patch diagnostic contract is not frozen and training-only")
    prelaunch_path = workspace_path(config["prelaunch_config"])
    verify_hash(prelaunch_path, str(config["prelaunch_config_sha256"]))
    prelaunch = json.loads(prelaunch_path.read_text())
    failed_path = workspace_path(config["failed_full_image_result"])
    verify_hash(failed_path, str(config["failed_full_image_result_sha256"]))
    failed = json.loads(failed_path.read_text())
    if (
        failed.get("status") != "failed"
        or failed.get("selected_candidate") is not None
        or failed.get("protected_foundir_test_accessed") is not False
        or failed.get("prelaunch_config_sha256") != config["prelaunch_config_sha256"]
    ):
        raise RuntimeError("full-image failure evidence changed")
    joint = prelaunch["candidates"]["joint"]
    pilot_path = workspace_path(joint["predictions"])
    verify_hash(pilot_path, str(joint["predictions_sha256"]))
    records = load_manifest(pilot_path)
    if len(records) != int(config["records"]):
        raise RuntimeError("native-patch pilot cardinality changed")
    return config, prelaunch, records


def build_view_records(
    records: Sequence[Mapping[str, str]], *, crop_size: int, views: Sequence[str]
) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for parent_index, record in enumerate(records):
        image_path = workspace_path(record["lq_path"])
        verify_hash(image_path, record["lq_sha256"])
        with Image.open(image_path) as image:
            boxes = crop_boxes(image.width, image.height, crop_size)
        for view in views:
            if view not in boxes:
                raise RuntimeError(f"unsupported native-patch view: {view}")
            expanded.append(
                {
                    **record,
                    "record_id": f"{record['record_id']}::{view}",
                    "parent_record_id": record["record_id"],
                    "parent_index": parent_index,
                    "view": view,
                    "output_path": record["lq_path"],
                    "crop_box": list(boxes[view]),
                }
            )
    return expanded


def parse_levels(raw: str, candidate: str) -> tuple[bool, dict[str, int] | None]:
    try:
        label = (
            parse_output(raw, FIXED_FIVE_SLOT_SCHEMA)
            if candidate == "joint"
            else parse_fixed_five_slot_target(raw)
        )
        return True, {name: label.level_map.get(name, 1) for name in DEGRADATION_ORDER}
    except Exception:
        return False, None


def aggregate(values: Sequence[int], method: str) -> int:
    ordered = sorted(int(value) for value in values)
    if len(ordered) != 5:
        raise RuntimeError("native-patch aggregation requires exactly five views")
    if method == "maximum":
        return ordered[-1]
    if method == "second_highest":
        return ordered[-2]
    if method == "median":
        return ordered[2]
    raise ValueError(f"unsupported aggregation: {method}")


def aggregated_predictions(
    payload: Mapping[str, object],
    records: Sequence[Mapping[str, str]],
    *,
    method: str,
) -> list[dict[str, object]]:
    rows = payload["view_predictions"]
    if not isinstance(rows, list):
        raise RuntimeError("native-patch view predictions are malformed")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("native-patch prediction row is malformed")
        grouped.setdefault(str(row["parent_record_id"]), []).append(row)
    output: list[dict[str, object]] = []
    for record in records:
        views = grouped.get(record["record_id"], [])
        valid = len(views) == 5 and all(row.get("valid") is True for row in views)
        levels: dict[str, int] | None = None
        if valid:
            if method == "center":
                center = [row for row in views if row["view"] == "center"]
                if len(center) != 1:
                    raise RuntimeError("native-patch center view is missing or duplicated")
                levels = {name: int(center[0]["levels"][name]) for name in DEGRADATION_ORDER}
            else:
                levels = {
                    name: aggregate([int(row["levels"][name]) for row in views], method)
                    for name in DEGRADATION_ORDER
                }
        output.append(
            {
                "record_id": record["record_id"],
                "category": record["category"],
                "composition_size": int(record["composition_size"]),
                "valid": valid,
                "levels": levels,
            }
        )
    return output


def candidate_identity(
    config_path: Path,
    config: Mapping[str, object],
    prelaunch: Mapping[str, object],
    candidate: str,
) -> dict[str, object]:
    source = prelaunch["candidates"][candidate]
    return {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_config_sha256": sha256(config_path),
        "prelaunch_config_sha256": config["prelaunch_config_sha256"],
        "failed_full_image_result_sha256": config["failed_full_image_result_sha256"],
        "pilot_sha256": prelaunch["candidates"]["joint"]["predictions_sha256"],
        "candidate": candidate,
        "qwen_config_sha256": source["qwen_config_sha256"],
        "qwen_adapter_sha256": source["adapter_sha256"],
        "records": int(config["records"]),
        "crop_size": int(config["crop_size"]),
        "views": list(config["views"]),
        "protected_foundir_test_accessed": False,
    }


def verify_candidate_payload(
    payload: Mapping[str, object], identity: Mapping[str, object]
) -> None:
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"native-patch candidate identity changed: {key}")
    rows = payload.get("view_predictions")
    expected_rows = int(identity["records"]) * len(identity["views"])
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise RuntimeError("native-patch candidate row count changed")


def run_candidate(config_path: Path, candidate: str, *, verify_existing: bool) -> None:
    config, prelaunch, records = load_contract(config_path)
    identity = candidate_identity(config_path, config, prelaunch, candidate)
    output = workspace_path(config["candidates"][candidate]["output"])
    if verify_existing:
        payload = json.loads(output.read_text())
        verify_candidate_payload(payload, identity)
        print(f"FOUNDIR_NATIVE_PATCH_{candidate.upper()}_VERIFY_PASS", flush=True)
        return

    initialized = False
    try:
        qwen_config_path = workspace_path(prelaunch["candidates"][candidate]["qwen_config"])
        qwen_config = json.loads(qwen_config_path.read_text())
        rank, world, local_rank, device = setup_distributed(qwen_config)
        initialized = True
        if world != 8:
            raise RuntimeError("native-patch diagnostic requires exactly eight GPUs")

        def prepare() -> list[dict[str, object]]:
            source = prelaunch["candidates"][candidate]
            verify_hash(qwen_config_path, str(source["qwen_config_sha256"]))
            verify_hash(workspace_path(source["adapter"]), str(source["adapter_sha256"]))
            return build_view_records(
                records,
                crop_size=int(config["crop_size"]),
                views=list(config["views"]),
            )

        view_records = broadcast_rank0(rank, prepare)
        validate_runtime(qwen_config, local_rank)
        model, processor, _ = load_model(
            qwen_config, variant="merger_trainable", device=device
        )
        load_qwen_adapter(
            model, workspace_path(prelaunch["candidates"][candidate]["adapter"])
        )
        raw_outputs = generate_outputs(
            model,
            processor,
            view_records,
            qwen_config,
            rank=rank,
            world=world,
            device=device,
        )

        def publish() -> dict[str, object]:
            predictions = []
            for record, raw in zip(view_records, raw_outputs, strict=True):
                valid, levels = parse_levels(raw, candidate)
                predictions.append(
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
            payload = {**identity, "view_predictions": predictions}
            write_once(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            verify_candidate_payload(payload, identity)
            return {"candidate": candidate, "records": len(predictions)}

        summary = broadcast_rank0(rank, publish)
        if rank == 0:
            print(
                "FOUNDIR_NATIVE_PATCH_CANDIDATE_PASS "
                + json.dumps(summary, sort_keys=True),
                flush=True,
            )
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


def finalize(config_path: Path) -> None:
    config, prelaunch, records = load_contract(config_path)
    failed = json.loads(workspace_path(config["failed_full_image_result"]).read_text())
    arms: dict[str, object] = {}
    passing: list[tuple[str, str]] = []
    for candidate in ("joint", "per_type"):
        output = workspace_path(config["candidates"][candidate]["output"])
        payload = json.loads(output.read_text())
        verify_candidate_payload(
            payload, candidate_identity(config_path, config, prelaunch, candidate)
        )
        aggregations: dict[str, object] = {}
        for method in config["aggregations"]:
            predictions = aggregated_predictions(payload, records, method=str(method))
            metrics = calculate(predictions)
            gate_checks = checks(metrics, config["thresholds"])
            passed = all(gate_checks.values())
            aggregations[str(method)] = {
                "metrics": metrics,
                "checks": gate_checks,
                "automatic_gate_pass": passed,
            }
            if passed:
                passing.append((candidate, str(method)))
        arms[candidate] = {"aggregations": aggregations}

    rule = list(config["selection_rule"])
    selected = None
    if passing:
        selected = max(
            passing,
            key=lambda value: tuple(
                float(arms[value[0]]["aggregations"][value[1]]["metrics"][metric])
                for metric in rule
            ),
        )
    prior_best_micro = max(
        float(value["metrics"]["type_micro_f1"])
        for value in failed["candidates"].values()
    )
    native_best_micro = max(
        float(value["metrics"]["type_micro_f1"])
        for arm in arms.values()
        for value in arm["aggregations"].values()
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_config_sha256": sha256(config_path),
        "failed_full_image_result_sha256": config["failed_full_image_result_sha256"],
        "records": len(records),
        "arms": arms,
        "selected_passing_arm": (
            {"candidate": selected[0], "aggregation": selected[1]}
            if selected is not None
            else None
        ),
        "scale_mismatch_hypothesis_supported": native_best_micro > prior_best_micro,
        "best_native_patch_type_micro_f1": native_best_micro,
        "best_full_image_type_micro_f1": prior_best_micro,
        "restoration_training_authorized": False,
        "protected_foundir_test_accessed": False,
    }
    output = workspace_path(config["output"])
    write_once(output, (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    print("FOUNDIR_NATIVE_PATCH_DIAGNOSTIC_COMPLETE " + json.dumps(result, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_native_patch_diagnostic_v1.json",
    )
    parser.add_argument("--candidate", choices=("joint", "per_type"))
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    if args.finalize == (args.candidate is not None):
        raise ValueError("choose exactly one of --candidate or --finalize")
    if args.finalize:
        if args.verify_existing:
            raise ValueError("--verify-existing is only valid with --candidate")
        finalize(config_path)
    else:
        run_candidate(config_path, str(args.candidate), verify_existing=args.verify_existing)


if __name__ == "__main__":
    main()
