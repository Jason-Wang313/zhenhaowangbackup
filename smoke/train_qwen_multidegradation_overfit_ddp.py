#!/usr/bin/env python3
"""Bounded two/four-GPU memorization gate before the full Qwen LoRA ablation."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import (
    read_jsonl,
    sha256,
    stable_digest,
)
from lowlight_restoration.qwen_lora import save_qwen_adapter
from lowlight_restoration.qwen_multidegradation import (
    DEGRADATIONS,
    FIXED_FIVE_SLOT_SCHEMA,
    PER_TYPE_DIALOGUE_SCHEMA,
    evaluate_outputs,
    prepare_per_type_supervised_batch,
)
from smoke.qwen_multidegradation_lora_preflight import (
    accumulation_steps_for_world,
    load_config,
    run_preflight,
    validate_contract,
    workspace_path,
)
from smoke.train_qwen_multidegradation_lora_ddp import (
    atomic_json,
    broadcast_rank0,
    create_optimizer,
    generate_per_type_outputs,
    generate_outputs,
    label_for,
    load_model,
    move_batch,
    open_images,
    rank_microbatch,
    seed_everything,
    setup_distributed,
    validate_runtime,
)
from lowlight_restoration.qwen_multidegradation import prepare_supervised_batch


def overfit_accumulation_steps(overfit: Mapping[str, object], world: int) -> int:
    mapping = overfit.get("gradient_accumulation_steps_by_world_size")
    if not isinstance(mapping, Mapping) or str(world) not in mapping:
        raise ValueError(f"no overfit accumulation contract for world size {world}")
    steps = int(mapping[str(world)])
    if steps <= 0:
        raise ValueError("overfit gradient accumulation must be positive")
    return steps


def expand_per_type_records(
    records: Sequence[Mapping[str, object]], *, seed: int
) -> list[Mapping[str, object]]:
    dialogues = [
        {**record, "query_degradation": degradation}
        for record in records
        for degradation in DEGRADATIONS
    ]
    dialogues.sort(
        key=lambda record: stable_digest(
            seed,
            "overfit_per_type",
            record["record_id"],
            record["query_degradation"],
        )
    )
    return dialogues


def select_overfit_records(
    records: Sequence[Mapping[str, object]], *, seed: int
) -> list[Mapping[str, object]]:
    """Select eight records per composition while covering all 20 type-levels."""

    selected: list[Mapping[str, object]] = []
    covered: set[tuple[str, int]] = set()
    used_sources: set[str] = set()
    for size in range(4):
        candidates = [
            record
            for record in records
            if record["split"] == "train" and int(record["composition_size"]) == size
        ]
        candidates.sort(key=lambda record: stable_digest(seed, size, record["record_id"]))
        chosen: list[Mapping[str, object]] = []
        while len(chosen) < 8:
            available = [record for record in candidates if record not in chosen]
            if not available:
                raise RuntimeError(f"cannot select overfit composition {size}")
            def value(record: Mapping[str, object]) -> tuple[int, int, str]:
                levels = record["levels"]
                assert isinstance(levels, Mapping)
                pairs = {(str(name), int(level)) for name, level in levels.items()}
                return (
                    len(pairs - covered),
                    int(str(record["source_id"]) not in used_sources),
                    stable_digest(seed, "tie", record["record_id"]),
                )
            winner = max(available, key=value)
            chosen.append(winner)
            levels = winner["levels"]
            assert isinstance(levels, Mapping)
            covered.update((str(name), int(level)) for name, level in levels.items())
            used_sources.add(str(winner["source_id"]))
        selected.extend(chosen)
    expected = {(name, level) for name in ("blur", "noise", "rain", "haze", "low_light") for level in range(2, 6)}
    if covered != expected:
        raise RuntimeError(f"overfit selection misses type-level pairs: {sorted(expected-covered)}")
    if len(selected) != 32 or Counter(int(record["composition_size"]) for record in selected) != Counter({0: 8, 1: 8, 2: 8, 3: 8}):
        raise RuntimeError("overfit selection composition changed")
    return selected


def probe_loss(
    model: torch.nn.Module,
    processor: object,
    records: Sequence[Mapping[str, object]],
    device: torch.device,
    *,
    rank: int,
    world: int,
    micro: int,
    accumulation: int,
    target_schema: str,
) -> float:
    model.eval()
    loss = torch.zeros((), dtype=torch.float32, device=device)
    for accumulation_index in range(accumulation):
        local_records = rank_microbatch(
            records,
            rank=rank,
            world=world,
            micro=micro,
            accumulation_index=accumulation_index,
        )
        images = open_images(local_records)
        labels = [label_for(record) for record in local_records]
        if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
            batch, _ = prepare_per_type_supervised_batch(
                processor,
                labels,
                images,
                [str(record["query_degradation"]) for record in local_records],
            )
        else:
            batch, _ = prepare_supervised_batch(processor, labels, images)
        with torch.no_grad():
            loss += (
                model(**move_batch(batch, device), use_cache=False).loss.detach().float()
                / accumulation
            )
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= dist.get_world_size()
    model.train()
    return float(loss.item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config, config_sha = load_config(args.config)
    validate_contract(config)
    training = config["training"]
    overfit = config["overfit"]
    runtime = config["runtime"]
    data = config["data"]
    assert all(isinstance(item, dict) for item in (training, overfit, runtime, data))
    output = workspace_path(
        overfit.get(
            "output_root",
            workspace_path(runtime["output_root"])
            .with_name("qwen_multidegradation_lora_overfit_v1")
            .relative_to(ROOT)
            .as_posix(),
        )
    )
    result_path = output / "result.json"
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
                raise RuntimeError("existing overfit result belongs to another config")
            return 0 if existing.get("automatic_gate_pass") is True else 2
        preflight = broadcast_rank0(rank, lambda: run_preflight(args.config, full_hashes=False))
        runtime_evidence = validate_runtime(config, local_rank)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        seed_everything(int(training["seed"]))
        records = read_jsonl(workspace_path(data["manifest"]))
        selected = select_overfit_records(records, seed=int(overfit["selection_seed"]))
        prompt = config["prompt"]
        assert isinstance(prompt, Mapping)
        target_schema = str(prompt["target_schema"])
        training_records = (
            expand_per_type_records(selected, seed=int(overfit["selection_seed"]))
            if target_schema == PER_TYPE_DIALOGUE_SCHEMA
            else list(selected)
        )
        bundle_sha = __import__("hashlib").sha256(
            json.dumps([record["record_id"] for record in selected], separators=(",", ":")).encode()
        ).hexdigest()
        model, processor, scope = load_model(
            config, variant=str(overfit["variant"]), device=device
        )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        ddp = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        optimizer = create_optimizer(model, config)
        for group in optimizer.param_groups:
            rate_key = {
                "lora": "lora_learning_rate",
                "merger": "merger_learning_rate",
                "language_lora": "language_lora_learning_rate",
            }[group["kind"]]
            group["lr"] = float(overfit.get(rate_key, overfit["lora_learning_rate"]))
        micro = int(overfit.get("micro_batch_size_per_rank", training["micro_batch_size_per_rank"]))
        accumulation = (
            overfit_accumulation_steps(overfit, world)
            if "gradient_accumulation_steps_by_world_size" in overfit
            else accumulation_steps_for_world(config, world)
        )
        overfit_global_batch = micro * world * accumulation
        if len(training_records) < overfit_global_batch:
            raise RuntimeError("overfit dialogue bundle is smaller than its global batch")
        probe_records = training_records[:overfit_global_batch]
        initial_loss = probe_loss(
            model,
            processor,
            probe_records,
            device,
            rank=rank,
            world=world,
            micro=micro,
            accumulation=accumulation,
            target_schema=target_schema,
        )
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        gradient_seen: set[str] = set()
        frozen_gradients: set[str] = set()
        losses: list[dict[str, object]] = []
        started = time.monotonic()
        steps = int(overfit["total_optimizer_steps"])
        for step in range(1, steps + 1):
            shift = (step * overfit_global_batch) % len(training_records)
            ordered = training_records[shift:] + training_records[:shift]
            global_records = ordered[:overfit_global_batch]
            optimizer.zero_grad(set_to_none=True)
            reduced = torch.zeros((), dtype=torch.float32, device=device)
            for accumulation_index in range(accumulation):
                local_records = rank_microbatch(
                    global_records,
                    rank=rank,
                    world=world,
                    micro=micro,
                    accumulation_index=accumulation_index,
                )
                images = open_images(local_records)
                labels = [label_for(record) for record in local_records]
                if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
                    batch, _ = prepare_per_type_supervised_batch(
                        processor,
                        labels,
                        images,
                        [str(record["query_degradation"]) for record in local_records],
                    )
                else:
                    batch, _ = prepare_supervised_batch(processor, labels, images)
                synchronization = (
                    ddp.no_sync()
                    if accumulation_index + 1 < accumulation
                    else contextlib.nullcontext()
                )
                with synchronization:
                    loss = ddp(**move_batch(batch, device), use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite overfit loss at step {step}")
                    (loss / accumulation).backward()
                reduced += loss.detach().float() / accumulation
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    gradient_seen.add(name)
                elif not parameter.requires_grad and parameter.grad is not None:
                    frozen_gradients.add(name)
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(training["max_grad_norm"]),
            )
            optimizer.step()
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            reduced /= world
            if rank == 0 and (step <= 5 or step % 10 == 0 or step == steps):
                row = {"step": step, "loss": float(reduced.item()), "grad_norm": float(norm.detach().float().item())}
                losses.append(row)
                print("QWEN_MULTIDEGRADATION_OVERFIT_TRAIN " + json.dumps(row, sort_keys=True), flush=True)
        final_loss = probe_loss(
            model,
            processor,
            probe_records,
            device,
            rank=rank,
            world=world,
            micro=micro,
            accumulation=accumulation,
            target_schema=target_schema,
        )
        if target_schema == PER_TYPE_DIALOGUE_SCHEMA:
            raw, response_details = generate_per_type_outputs(
                model,
                processor,
                selected,
                config,
                rank=rank,
                world=world,
                device=device,
            )
            evaluation_schema = FIXED_FIVE_SLOT_SCHEMA
        else:
            raw = generate_outputs(model, processor, selected, config, rank=rank, world=world, device=device)
            response_details = None
            evaluation_schema = target_schema
        rows, metrics = evaluate_outputs(
            [label_for(record) for record in selected],
            raw,
            target_schema=evaluation_schema,
        )
        for row, record in zip(rows, selected, strict=True):
            row["record_id"] = record["record_id"]
        if response_details is not None:
            for row, details in zip(rows, response_details, strict=True):
                row["per_type_responses"] = details
        local_audit = not frozen_gradients and gradient_seen == trainable
        audit_tensor = torch.tensor(int(local_audit), dtype=torch.int32, device=device)
        dist.all_reduce(audit_tensor, op=dist.ReduceOp.MIN)
        checks = {
            "loss_ratio": final_loss / initial_loss <= float(overfit["maximum_final_to_initial_loss_ratio"]),
            "valid_structure": float(metrics["valid_structured_output_rate"]) >= float(overfit["minimum_valid_structured_output_rate"]),
            "degradation_set_exact": float(metrics["degradation_set_exact_match_rate"]) >= float(overfit["minimum_degradation_set_exact_match_rate"]),
            "combined_label_exact": float(metrics["combined_label_exact_match_rate"]) >= float(overfit["minimum_combined_label_exact_match_rate"]),
            "severity_accuracy": float(metrics["severity_typed_accuracy"]) >= float(overfit["minimum_severity_typed_accuracy"]),
            "gradient_scope": bool(audit_tensor.item()),
        }
        automatic_pass = all(checks.values())
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            adapter = output / "adapter.safetensors"
            adapter_summary = save_qwen_adapter(
                model,
                adapter,
                {"config_sha256": config_sha, "kind": "overfit_gate_only", "steps": steps},
            )
            predictions = output / "predictions.json"
            atomic_json(predictions, {"schema_version": 1, "selection_bundle_sha256": bundle_sha, "records": rows})
            result = {
                "schema_version": 1,
                "status": "passed" if automatic_pass else "failed",
                "automatic_gate_pass": automatic_pass,
                "config_sha256": config_sha,
                "selection_bundle_sha256": bundle_sha,
                "selected_records": [record["record_id"] for record in selected],
                "type_level_coverage": 20,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "final_to_initial_loss_ratio": final_loss / initial_loss,
                "metrics": metrics,
                "gate_checks": checks,
                "scope": scope,
                "training_metrics": losses,
                "elapsed_seconds": time.monotonic() - started,
                "dialogue_records": len(training_records),
                "effective_global_batch_size": overfit_global_batch,
                "adapter_sha256": adapter_summary["adapter_sha256"],
                "predictions_sha256": sha256(predictions),
                "runtime": runtime_evidence,
                "preflight_config_sha256": preflight["config_sha256"],
            }
            atomic_json(result_path, result)
            print(
                ("QWEN_MULTIDEGRADATION_OVERFIT_PASS " if automatic_pass else "QWEN_MULTIDEGRADATION_OVERFIT_FAILED ")
                + json.dumps({"loss_ratio": result["final_to_initial_loss_ratio"], "metrics": metrics}, sort_keys=True),
                flush=True,
            )
        passed = torch.tensor(int(automatic_pass), dtype=torch.int32, device=device)
        dist.broadcast(passed, src=0)
        return 0 if passed.item() else 2
    finally:
        if ready:
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
