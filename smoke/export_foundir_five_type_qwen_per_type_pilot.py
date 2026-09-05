#!/usr/bin/env python3
"""Export a training-only FoundIR pilot from the frozen per-type Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping

import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import FIVE_TYPE_CATEGORIES  # noqa: E402
from lowlight_restoration.qwen_lora import load_qwen_adapter  # noqa: E402
from lowlight_restoration.qwen_multidegradation import (  # noqa: E402
    PER_TYPE_DIALOGUE_SCHEMA,
    parse_fixed_five_slot_target,
)
from smoke.export_foundir_five_type_qwen_conditions import (  # noqa: E402
    load_manifest,
    sha256,
    write_once,
)
from smoke.train_qwen_multidegradation_lora_ddp import (  # noqa: E402
    broadcast_rank0,
    generate_per_type_outputs,
    load_model,
    seed_everything,
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
        raise RuntimeError(f"frozen artifact changed: {path}: {actual}")


def load_and_verify_pilot_records(gate: Mapping[str, object]) -> list[dict[str, str]]:
    """Load the frozen 128-row pilot and verify the exact images being evaluated."""

    joint = gate["candidates"]["joint"]
    pilot = workspace_path(joint["predictions"])
    verify_hash(pilot, str(joint["predictions_sha256"]))
    records = load_manifest(pilot)
    expected = int(gate["records"])
    if (
        len(records) != expected
        or len({record["record_id"] for record in records}) != expected
        or any(record["split"] != "train" for record in records)
    ):
        raise RuntimeError("frozen Joint-Qwen pilot record set changed")
    if set(record["category"] for record in records) != set(FIVE_TYPE_CATEGORIES):
        raise RuntimeError("frozen Joint-Qwen pilot no longer covers every category")
    for record in records:
        verify_hash(workspace_path(record["lq_path"]), record["lq_sha256"])
    return records


def expected_identity(
    gate: Mapping[str, object], gate_path: Path
) -> dict[str, object]:
    candidate = gate["candidates"]["per_type"]
    return {
        "schema_version": 1,
        "status": "frozen_per_type_qwen_foundir_training_predictions",
        "prelaunch_config_sha256": sha256(gate_path),
        "manifest_sha256": str(gate["manifest_sha256"]),
        "qwen_config_sha256": str(candidate["qwen_config_sha256"]),
        "qwen_adapter_sha256": str(candidate["adapter_sha256"]),
        "records": int(gate["records"]),
        "decoding_version": "left_padding_v2",
        "padding_side": "left",
        "protected_foundir_test_accessed": False,
    }


def verify_existing(
    path: Path,
    identity: Mapping[str, object],
    expected_records: list[Mapping[str, str]] | None = None,
) -> dict[str, object]:
    payload = json.loads(path.read_text())
    for key, value in identity.items():
        if payload.get(key) != value:
            raise RuntimeError(f"per-type FoundIR pilot identity changed: {key}")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != int(identity["records"]):
        raise RuntimeError("per-type FoundIR pilot cardinality changed")
    if payload.get("valid_records") != sum(
        bool(row.get("valid")) for row in predictions if isinstance(row, Mapping)
    ):
        raise RuntimeError("per-type FoundIR pilot valid-record count changed")
    if expected_records is not None:
        if len(expected_records) != len(predictions):
            raise RuntimeError("per-type FoundIR pilot expected-record count changed")
        for prediction, record in zip(predictions, expected_records, strict=True):
            if not isinstance(prediction, Mapping) or (
                prediction.get("record_id") != record["record_id"]
                or prediction.get("category") != record["category"]
                or int(prediction.get("composition_size", -1))
                != int(record["composition_size"])
                or prediction.get("lq_sha256") != record["lq_sha256"]
            ):
                raise RuntimeError("per-type FoundIR pilot record mapping changed")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prelaunch-config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_prelaunch_gate_v1.json",
    )
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    gate_path = workspace_path(args.prelaunch_config)
    gate = json.loads(gate_path.read_text())
    if (
        gate.get("status") != "frozen_training_only_prelaunch_gate"
        or gate.get("protected_foundir_test_accessed") is not False
    ):
        raise RuntimeError("prelaunch gate is not frozen or permits protected data")
    candidate = gate["candidates"]["per_type"]
    qwen_config_path = workspace_path(candidate["qwen_config"])
    adapter = workspace_path(candidate["adapter"])
    verify_hash(qwen_config_path, str(candidate["qwen_config_sha256"]))
    verify_hash(adapter, str(candidate["adapter_sha256"]))
    qwen_config = json.loads(qwen_config_path.read_text())
    if qwen_config["prompt"]["target_schema"] != PER_TYPE_DIALOGUE_SCHEMA:
        raise RuntimeError("per-type Qwen schema changed")
    records = load_and_verify_pilot_records(gate)
    output = workspace_path(candidate["predictions"])
    identity = expected_identity(gate, gate_path)
    if args.verify_existing:
        payload = verify_existing(output, identity, records)
        print("FOUNDIR_FIVE_TYPE_QWEN_PER_TYPE_VERIFY_PASS " + json.dumps(payload, sort_keys=True))
        return

    generation_records = [{**row, "output_path": row["lq_path"]} for row in records]
    initialized = False
    try:
        rank, world, local_rank, device = setup_distributed(qwen_config)
        initialized = True
        if world != 8:
            raise RuntimeError("per-type FoundIR pilot requires exactly eight GPUs")
        validate_runtime(qwen_config, local_rank)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
        )
        seed_everything(int(gate["selection_seed"]))
        model, processor, _ = load_model(
            qwen_config, variant="merger_trainable", device=device
        )
        load_qwen_adapter(model, adapter)
        outputs, details = generate_per_type_outputs(
            model,
            processor,
            generation_records,
            qwen_config,
            rank=rank,
            world=world,
            device=device,
        )

        def publish() -> dict[str, object]:
            predictions = []
            for record, raw, per_type in zip(records, outputs, details, strict=True):
                try:
                    label = parse_fixed_five_slot_target(raw)
                    levels = {name: label.level_map.get(name, 1) for name in ("blur", "noise", "rain", "haze", "low_light")}
                    valid = True
                except Exception:
                    levels = None
                    valid = False
                predictions.append(
                    {
                        "record_id": record["record_id"],
                        "category": record["category"],
                        "composition_size": int(record["composition_size"]),
                        "lq_sha256": record["lq_sha256"],
                        "raw_output": raw,
                        "valid": valid,
                        "levels": levels,
                        "per_type_responses": per_type,
                    }
                )
            payload = {
                **identity,
                "valid_records": sum(bool(row["valid"]) for row in predictions),
                "predictions": predictions,
            }
            content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
            write_once(output, content)
            return verify_existing(output, identity, records)

        payload = broadcast_rank0(rank, publish)
        if rank == 0:
            print(
                "FOUNDIR_FIVE_TYPE_QWEN_PER_TYPE_PASS "
                + json.dumps({key: value for key, value in payload.items() if key != "predictions"}, sort_keys=True),
                flush=True,
            )
    finally:
        if initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
