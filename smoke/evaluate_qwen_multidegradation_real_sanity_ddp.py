#!/usr/bin/env python3
"""Run the frozen synthetic winner on the predeclared 25-image real pool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256
from lowlight_restoration.qwen_lora import load_qwen_adapter
from lowlight_restoration.qwen_multidegradation import canonical_target, parse_target
from smoke.qwen_multidegradation_lora_preflight import (
    load_config,
    run_preflight,
    validate_contract,
    workspace_path,
)
from smoke.train_qwen_multidegradation_lora_ddp import (
    atomic_json,
    broadcast_rank0,
    generate_outputs,
    load_model,
    seed_everything,
    setup_distributed,
    validate_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha = load_config(args.config)
    validate_contract(config)
    runtime = config["runtime"]
    training = config["training"]
    data = config["data"]
    assert isinstance(runtime, dict) and isinstance(training, dict) and isinstance(data, dict)
    root = workspace_path(runtime["output_root"])
    output = root / "real_sanity"
    result_path = output / "predictions.json"
    rank = world = local_rank = -1
    ready = False
    try:
        rank, world, local_rank, device = setup_distributed(config)
        ready = True
        existing = broadcast_rank0(
            rank, lambda: json.loads(result_path.read_text()) if result_path.is_file() else None
        )
        if existing is not None:
            if existing.get("config_sha256") != config_sha:
                raise RuntimeError("real sanity result belongs to another config")
            return 0
        preflight = broadcast_rank0(rank, lambda: run_preflight(args.config, full_hashes=False))
        runtime_evidence = validate_runtime(config, local_rank)
        selection_path = root / "checkpoint_selection.json"
        selection = broadcast_rank0(rank, lambda: json.loads(selection_path.read_text()))
        variant = selection.get("selected_variant")
        if variant not in ("merger_frozen", "merger_trainable"):
            raise RuntimeError("no frozen validation winner for real sanity evaluation")
        fixed_result_path = root / variant / "fixed_test_result.json"
        fixed_result = broadcast_rank0(rank, lambda: json.loads(fixed_result_path.read_text()))
        if fixed_result.get("automatic_gate_pass") is not True:
            raise RuntimeError("fixed synthetic test did not pass")
        manifest_path = workspace_path(data["real_sanity_manifest"])
        manifest = json.loads(manifest_path.read_text())
        if (
            sha256(manifest_path) != data["real_sanity_manifest_sha256"]
            or manifest.get("status")
            != "frozen_before_qwen_inference_official_sidd_bytes"
            or manifest["validation"].get("foundir_used") is not False
            or manifest["validation"].get("qwen_inference_before_freeze") is not False
        ):
            raise RuntimeError("real sanity manifest is not the frozen pre-inference non-FoundIR v2 pool")
        records = [{**record, "output_path": record["path"]} for record in manifest["records"]]
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(int(training["seed"]))
        model, processor, scope = load_model(config, variant=variant, device=device)
        adapter = workspace_path(selection["selected_adapter"])
        if sha256(adapter) != selection["selected_adapter_sha256"]:
            raise RuntimeError("selected adapter hash changed before real sanity inference")
        load_qwen_adapter(model, adapter)
        raw_outputs = generate_outputs(
            model, processor, records, config, rank=rank, world=world, device=device
        )
        rows: list[dict[str, object]] = []
        valid = primary_present = 0
        for record, raw in zip(records, raw_outputs, strict=True):
            predicted = None
            error = None
            try:
                label = parse_target(raw)
                predicted = canonical_target(label)
                valid += 1
                primary_present += int(record["expected_primary_type"] in label.degradations)
            except ValueError as exc:
                error = str(exc)
            rows.append(
                {
                    "record_id": record["record_id"],
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "expected_primary_type": record["expected_primary_type"],
                    "severity_ground_truth": None,
                    "raw_output": raw,
                    "predicted": predicted,
                    "valid_structured_output": predicted is not None,
                    "primary_type_present": (
                        record["expected_primary_type"] in label.degradations
                        if predicted is not None
                        else False
                    ),
                    "parse_error": error,
                }
            )
        if rank == 0:
            payload = {
                "schema_version": 1,
                "status": "awaiting_manual_qualitative_review",
                "config_sha256": config_sha,
                "selected_variant": variant,
                "selected_adapter": selection["selected_adapter"],
                "selected_adapter_sha256": selection["selected_adapter_sha256"],
                "real_manifest": manifest_path.relative_to(ROOT).as_posix(),
                "real_manifest_sha256": sha256(manifest_path),
                "foundir_used": False,
                "records": rows,
                "diagnostic_valid_structure_rate": valid / len(rows),
                "diagnostic_primary_type_recall": primary_present / len(rows),
                "manual_review_required": True,
                "runtime": runtime_evidence,
                "scope": scope,
                "preflight_config_sha256": preflight["config_sha256"],
            }
            atomic_json(result_path, payload)
            print(
                "QWEN_MULTIDEGRADATION_REAL_SANITY_INFERENCE_PASS "
                f"valid={valid}/{len(rows)} primary_present={primary_present}/{len(rows)}",
                flush=True,
            )
        return 0
    finally:
        if ready:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
