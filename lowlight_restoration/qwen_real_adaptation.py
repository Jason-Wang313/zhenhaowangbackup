"""Contracts and deterministic schedules for real-domain Qwen adaptation.

The adaptation stage continues from the frozen synthetic-pilot step-600 visual
checkpoint.  It mixes balanced LOL-v2 real-captured training examples with
balanced synthetic replay while keeping the tokenizer and language decoder
frozen.  All schedule construction is independent of DDP world size so a
2-GPU H100/H200 run and an 8-GPU H200 run preserve the same global batches.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from .data import PairRecord
from .qwen_distributed import complete_source_groups
from .qwen_supervised import record_bundle_sha256

APPROVED_STATUS = "approved_for_qwen_real_lowlight_adaptation_gate_only"
REAL_SPLITS = ("train", "validation", "test")
ALLOWED_WORLD_SIZES = (2, 8)


def load_adaptation_config(
    path: Path,
) -> tuple[dict[str, object], str, bytes]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("bad Qwen real-adaptation config schema")
    if config.get("status") != APPROVED_STATUS:
        raise ValueError("config is not approved for real-domain adaptation")
    return config, hashlib.sha256(raw).hexdigest(), raw


def _positive_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def validate_adaptation_contract(config: Mapping[str, object]) -> None:
    required_sections = (
        "model",
        "processor",
        "prompt",
        "data",
        "training",
        "distributed",
        "generation",
        "gate",
        "runtime",
        "telemetry",
    )
    sections = {name: config.get(name) for name in required_sections}
    if any(not isinstance(section, dict) for section in sections.values()):
        raise ValueError("real-adaptation config is missing a required section")
    model = sections["model"]
    processor = sections["processor"]
    data = sections["data"]
    training = sections["training"]
    distributed = sections["distributed"]
    generation = sections["generation"]
    gate = sections["gate"]
    runtime = sections["runtime"]
    telemetry = sections["telemetry"]
    assert all(
        isinstance(section, dict)
        for section in (
            model,
            processor,
            data,
            training,
            distributed,
            generation,
            gate,
            runtime,
            telemetry,
        )
    )

    if (
        config.get("scope")
        != "source_disjoint_LOLv2_train_real_adaptation_with_synthetic_replay"
        or config.get("official_foundir_used") is not False
        or config.get("previously_evaluated_real_records_used_for_training")
        is not False
        or config.get("holdout_predictions_consulted_before_freeze") is not False
    ):
        raise ValueError("real-adaptation scope or holdout protection changed")
    failures = config.get("preserved_failures")
    if (
        not isinstance(failures, dict)
        or failures.get("both_status_failed") is not True
        or failures.get("neither_is_retroactively_reclassified") is not True
    ):
        raise ValueError("real-domain failure preservation contract changed")
    for name in (
        "original_free_generation_result",
        "original_free_generation_result_sha256",
        "post_failure_constrained_result",
        "post_failure_constrained_result_sha256",
    ):
        if not isinstance(failures.get(name), str) or not failures[name]:
            raise ValueError(f"preserved real-domain failure field is missing: {name}")
    if (
        model.get("architecture") != "Qwen2_5_VLForConditionalGeneration"
        or model.get("trainable_module") != "model.visual"
        or model.get("frozen_modules") != ["model.language_model", "lm_head"]
        or model.get("dtype") != "bfloat16"
        or model.get("attention_implementation") != "sdpa"
    ):
        raise ValueError("Qwen adaptation model scope changed")
    groups = model.get("expected_parameter_groups")
    if not isinstance(groups, dict):
        raise ValueError("Qwen parameter-group contract is missing")
    visual = sum(
        int(groups.get(name, 0))
        for name in ("visual.patch_embed", "visual.blocks", "visual.merger")
    )
    frozen = sum(int(groups.get(name, 0)) for name in ("language_model", "lm_head"))
    if (
        visual != model.get("expected_trainable_parameters")
        or frozen != model.get("expected_frozen_parameters")
        or visual + frozen != model.get("expected_total_parameters")
    ):
        raise ValueError("Qwen adaptation parameter counts changed")
    initialization = model.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("adaptation initialization evidence is missing")
    for name in (
        "visual_checkpoint",
        "visual_sha256",
        "checkpoint_metadata",
        "checkpoint_metadata_sha256",
        "pilot_summary",
        "pilot_summary_sha256",
    ):
        if not isinstance(initialization.get(name), str) or not initialization[name]:
            raise ValueError(f"adaptation initialization field is missing: {name}")
    if (
        initialization.get("selected_checkpoint_step") != 600
        or initialization.get("optimizer_reinitialized") is not True
    ):
        raise ValueError("adaptation must initialize visual weights from step 600")
    if (
        processor.get("use_fast") is not False
        or processor.get("padding_side") != "right"
        or processor.get("min_pixels") != processor.get("max_pixels")
    ):
        raise ValueError("Qwen adaptation processor contract changed")

    real = data.get("real")
    synthetic = data.get("synthetic_replay")
    if not isinstance(real, dict) or not isinstance(synthetic, dict):
        raise ValueError("adaptation data contracts are incomplete")
    if (
        real.get("annotation_kind") != "reference_based_proxy_not_human_ground_truth"
        or real.get("model_predictions_consulted_during_split") is not False
        or real.get("source_disjoint_by_perceptual_scene_component") is not True
        or real.get("test_never_used_for_training_or_checkpoint_selection") is not True
    ):
        raise ValueError("real proxy-label or split disclosure changed")
    splits = real.get("splits")
    if not isinstance(splits, dict) or tuple(splits) != REAL_SPLITS:
        raise ValueError("real adaptation splits must be train/validation/test")
    for split in REAL_SPLITS:
        contract = splits.get(split)
        if not isinstance(contract, dict):
            raise ValueError(f"missing real adaptation split: {split}")
        if any(
            isinstance(contract.get(name), bool)
            or not isinstance(contract.get(name), int)
            or int(contract[name]) <= 0
            for name in ("records", "sources")
        ):
            raise ValueError(f"bad real split cardinality: {split}")
        per_level = contract.get("records_per_level")
        if (
            not isinstance(per_level, dict)
            or set(per_level) != {str(level) for level in range(1, 6)}
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in per_level.values()
            )
            or sum(per_level.values()) != contract["records"]
            or not isinstance(contract.get("bundle_sha256"), str)
        ):
            raise ValueError(f"bad real per-level contract: {split}")
    if (
        synthetic.get("manifest") is None
        or synthetic.get("training_split") != "train"
        or synthetic.get("training_records") != 8_000
        or synthetic.get("training_sources") != 1_600
        or synthetic.get("training_records_per_level") != 1_600
        or not isinstance(synthetic.get("training_bundle_sha256"), str)
        or synthetic.get("validation_split") != "validation"
        or synthetic.get("validation_selection_seed") != 2026082002
        or synthetic.get("validation_source_count") != 40
        or synthetic.get("validation_records") != 200
        or synthetic.get("validation_records_per_level") != 40
        or not isinstance(synthetic.get("validation_bundle_sha256"), str)
    ):
        raise ValueError("synthetic replay selection changed")

    allowed = distributed.get("allowed_world_sizes")
    accumulation = distributed.get("gradient_accumulation_by_world_size")
    if allowed != list(ALLOWED_WORLD_SIZES) or accumulation != {"2": 4, "8": 1}:
        raise ValueError("adaptation world-size compatibility changed")
    expected_distributed = {
        "backend": "nccl",
        "allowed_world_sizes": [2, 8],
        "broadcast_buffers": False,
        "init_sync": False,
        "gradient_as_bucket_view": True,
        "find_unused_parameters": False,
        "timeout_minutes": 30,
        "gradient_accumulation_by_world_size": {"2": 4, "8": 1},
    }
    if distributed != expected_distributed:
        raise ValueError("adaptation distributed mechanics changed")
    micro_batch = int(training.get("micro_batch_size_per_rank", 0))
    global_groups = int(training.get("global_microbatches_per_step", 0))
    global_batch = int(training.get("global_batch_size", 0))
    real_groups = int(training.get("real_microbatches_per_step", 0))
    synthetic_groups = int(training.get("synthetic_microbatches_per_step", 0))
    if (
        micro_batch != 5
        or global_groups != 8
        or global_batch != micro_batch * global_groups
        or real_groups != 4
        or synthetic_groups != 4
        or real_groups + synthetic_groups != global_groups
    ):
        raise ValueError("adaptation global batch composition changed")
    for world_size in ALLOWED_WORLD_SIZES:
        if (
            micro_batch * int(accumulation[str(world_size)]) * world_size
            != global_batch
        ):
            raise ValueError("adaptation effective batch differs by world size")
    total_steps = int(training.get("total_steps", 0))
    checkpoints = training.get("checkpoint_steps")
    validations = training.get("validation_loss_steps")
    if (
        total_steps != 200
        or checkpoints != [25, 50, 100, 150, 200]
        or validations != [0, 25, 50, 100, 150, 200]
        or training.get("training_schedule_world_size_invariant") is not True
        or training.get("gradient_checkpointing") is not True
        or training.get("fused_adamw") is not True
        or training.get("learning_rate_schedule") != "linear_warmup_cosine_decay"
        or training.get("checkpoint_selection_metric")
        != "minimum_real_validation_macro_level_teacher_forced_loss"
        or training.get("checkpoint_selection_tiebreaker") != "earlier_step"
    ):
        raise ValueError("adaptation training schedule changed")
    if not 0 < int(training.get("warmup_steps", 0)) < total_steps:
        raise ValueError("adaptation warmup is invalid")
    _positive_number(training.get("learning_rate"), name="learning_rate")
    _positive_number(
        training.get("minimum_learning_rate"), name="minimum_learning_rate"
    )

    if (
        generation.get("do_sample") is not False
        or generation.get("batch_size") != 5
        or generation.get("evaluate_real_validation_after_selection") is not True
        or generation.get("evaluate_real_test_once_after_selection") is not True
        or generation.get("evaluate_synthetic_replay_after_selection") is not True
    ):
        raise ValueError("adaptation generation/evaluation contract changed")
    if gate.get("thresholds_frozen_before_model_inference") is not True:
        raise ValueError("adaptation thresholds were not predeclared")
    for name in (
        "real_validation",
        "real_test",
        "synthetic_replay",
    ):
        if not isinstance(gate.get(name), dict):
            raise ValueError(f"adaptation gate is missing: {name}")
    if (
        not 0.0
        < float(gate.get("maximum_selected_to_initial_real_loss_ratio", 0))
        < 1.0
    ):
        raise ValueError("real validation loss-decrease threshold is invalid")
    for name in (
        "require_gradient_for_every_trainable_parameter",
        "require_no_frozen_gradients",
        "require_no_frozen_updates",
        "require_all_checkpoint_freeze_audits",
        "require_real_validation_and_test_thresholds",
        "require_synthetic_replay_thresholds",
    ):
        if gate.get(name) is not True:
            raise ValueError(f"mandatory adaptation gate was disabled: {name}")

    if (
        runtime.get("offline") is not True
        or runtime.get("device") != "cuda"
        or runtime.get("allowed_visible_device_counts") != [2, 8]
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("sdpa_backend") != "math"
    ):
        raise ValueError("adaptation runtime contract changed")
    if (
        telemetry.get("sample_interval_seconds") != 2
        or telemetry.get("monitor_all_selected_gpus") is not True
        or not isinstance(telemetry.get("csv"), str)
    ):
        raise ValueError("adaptation telemetry contract changed")


def records_by_real_split(
    records: Sequence[PairRecord], config: Mapping[str, object]
) -> dict[str, list[PairRecord]]:
    data = config["data"]
    assert isinstance(data, dict)
    real = data["real"]
    assert isinstance(real, dict)
    contracts = real["splits"]
    assert isinstance(contracts, dict)
    result: dict[str, list[PairRecord]] = {}
    source_sets: dict[str, set[str]] = {}
    for split in REAL_SPLITS:
        selected = [record for record in records if record.split == split]
        contract = contracts[split]
        assert isinstance(contract, dict)
        levels = Counter(record.level_index for record in selected)
        expected_levels = {
            int(level): int(count)
            for level, count in contract["records_per_level"].items()
        }
        if (
            len(selected) != int(contract["records"])
            or len({record.source_id for record in selected})
            != int(contract["sources"])
            or levels != Counter(expected_levels)
            or record_bundle_sha256(selected) != contract["bundle_sha256"]
        ):
            raise RuntimeError(f"real adaptation split evidence changed: {split}")
        result[split] = selected
        source_sets[split] = {record.source_id for record in selected}
    if any(
        source_sets[left] & source_sets[right]
        for index, left in enumerate(REAL_SPLITS)
        for right in REAL_SPLITS[index + 1 :]
    ):
        raise RuntimeError("real adaptation source split leakage detected")
    return result


def select_synthetic_replay(
    records: Sequence[PairRecord], *, source_count: int, seed: int
) -> list[PairRecord]:
    groups = complete_source_groups(
        [record for record in records if record.split == "validation"]
    )
    groups.sort(
        key=lambda group: hashlib.sha256(
            f"{seed}:synthetic-replay:{group[0].source_id}".encode()
        ).hexdigest()
    )
    if len(groups) < source_count or source_count <= 0:
        raise ValueError("not enough complete synthetic replay sources")
    return [record for group in groups[:source_count] for record in group]


def _cycle_order(
    records: Sequence[PairRecord], *, seed: int, namespace: str, cycle: int
) -> list[PairRecord]:
    if not records or cycle < 0:
        raise ValueError("bad deterministic cycle request")
    return sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                f"{seed}:{namespace}:{cycle}:{record.pair_id}".encode()
            ).hexdigest(),
            record.pair_id,
        ),
    )


def _cycle_pick(
    records: Sequence[PairRecord],
    *,
    index: int,
    seed: int,
    namespace: str,
    excluded_sources: set[str] | None = None,
) -> PairRecord:
    if index < 0 or not records:
        raise ValueError("bad deterministic cycle index")
    cycle, position = divmod(index, len(records))
    order = _cycle_order(records, seed=seed, namespace=namespace, cycle=cycle)
    excluded = excluded_sources or set()
    for offset in range(len(order)):
        candidate = order[(position + offset) % len(order)]
        if candidate.source_id not in excluded:
            return candidate
    raise ValueError(f"cannot build source-distinct balanced group: {namespace}")


def balanced_real_group(
    records_by_level: Mapping[int, Sequence[PairRecord]],
    *,
    group_index: int,
    seed: int,
) -> list[PairRecord]:
    if set(records_by_level) != set(range(1, 6)):
        raise ValueError("real training pools must contain L1--L5")
    chosen: list[PairRecord] = []
    used_sources: set[str] = set()
    for level in range(1, 6):
        record = _cycle_pick(
            records_by_level[level],
            index=group_index,
            seed=seed,
            namespace=f"real-L{level}",
            excluded_sources=used_sources,
        )
        chosen.append(record)
        used_sources.add(record.source_id)
    return chosen


def synthetic_group(
    groups: Sequence[Sequence[PairRecord]], *, group_index: int, seed: int
) -> list[PairRecord]:
    if group_index < 0 or not groups:
        raise ValueError("bad synthetic group index")
    proxies = [group[0] for group in groups]
    selected = _cycle_pick(
        proxies,
        index=group_index,
        seed=seed,
        namespace="synthetic-train-source",
    )
    by_source = {group[0].source_id: list(group) for group in groups}
    return by_source[selected.source_id]


def global_training_groups(
    real_records: Sequence[PairRecord],
    synthetic_records: Sequence[PairRecord],
    *,
    step: int,
    seed: int,
    real_groups_per_step: int = 4,
    synthetic_groups_per_step: int = 4,
) -> list[tuple[str, list[PairRecord]]]:
    """Return one world-size-invariant 8x5 global adaptation batch."""

    if step <= 0 or real_groups_per_step != 4 or synthetic_groups_per_step != 4:
        raise ValueError(
            "adaptation schedule requires four real and four replay groups"
        )
    real_by_level = {
        level: sorted(
            (record for record in real_records if record.level_index == level),
            key=lambda record: record.pair_id,
        )
        for level in range(1, 6)
    }
    if any(not values for values in real_by_level.values()):
        raise ValueError("real training split omits a severity level")
    replay_groups = complete_source_groups(synthetic_records)
    replay_groups.sort(key=lambda group: group[0].source_id)
    base = (step - 1) * 4
    real = [
        ("real", balanced_real_group(real_by_level, group_index=base + slot, seed=seed))
        for slot in range(4)
    ]
    replay = [
        (
            "synthetic_replay",
            synthetic_group(replay_groups, group_index=base + slot, seed=seed),
        )
        for slot in range(4)
    ]
    # This ordering gives each rank two real and two replay microbatches at
    # world size two; strided rank partitioning preserves all eight groups.
    return [
        real[0],
        real[1],
        replay[0],
        replay[1],
        real[2],
        real[3],
        replay[2],
        replay[3],
    ]


def partition_training_groups(
    groups: Sequence[tuple[str, list[PairRecord]]], *, world_size: int, rank: int
) -> list[tuple[str, list[PairRecord]]]:
    if world_size not in ALLOWED_WORLD_SIZES or not 0 <= rank < world_size:
        raise ValueError("unsupported adaptation rank partition")
    if len(groups) != 8:
        raise ValueError("adaptation global step must contain eight microbatches")
    local = list(groups[rank::world_size])
    if len(local) != 8 // world_size:
        raise RuntimeError("adaptation rank received the wrong accumulation count")
    return local


def training_schedule_evidence(
    real_records: Sequence[PairRecord],
    synthetic_records: Sequence[PairRecord],
    *,
    total_steps: int,
    seed: int,
) -> dict[str, object]:
    digest = hashlib.sha256()
    domains: Counter[str] = Counter()
    levels: Counter[tuple[str, int]] = Counter()
    unique_pairs: dict[str, set[str]] = {
        "real": set(),
        "synthetic_replay": set(),
    }
    first: list[dict[str, object]] | None = None
    last: list[dict[str, object]] | None = None
    for step in range(1, total_steps + 1):
        groups = global_training_groups(
            real_records, synthetic_records, step=step, seed=seed
        )
        rows: list[dict[str, object]] = []
        for slot, (domain, records) in enumerate(groups):
            if len(records) != 5 or {record.level_index for record in records} != set(
                range(1, 6)
            ):
                raise RuntimeError("adaptation microbatch lost L1--L5 balance")
            domains[domain] += len(records)
            for record in records:
                levels[(domain, record.level_index)] += 1
                unique_pairs[domain].add(record.pair_id)
            pair_ids = [record.pair_id for record in records]
            digest.update(f"{step}\0{slot}\0{domain}\0{'|'.join(pair_ids)}\n".encode())
            rows.append({"slot": slot, "domain": domain, "pair_ids": pair_ids})
        if first is None:
            first = rows
        last = rows
    return {
        "total_steps": total_steps,
        "global_batch_size": 40,
        "total_sample_presentations": total_steps * 40,
        "schedule_sha256": digest.hexdigest(),
        "domain_presentations": dict(sorted(domains.items())),
        "domain_level_presentations": {
            f"{domain}_L{level}": levels[(domain, level)]
            for domain in ("real", "synthetic_replay")
            for level in range(1, 6)
        },
        "unique_pair_ids": {
            domain: len(values) for domain, values in sorted(unique_pairs.items())
        },
        "first_step": first,
        "last_step": last,
        "world_size_invariant": True,
    }


def validate_adaptation_validation_entry(
    entry: Mapping[str, object],
    *,
    config_sha256: str,
    step: int,
    checkpoint_visual_sha256: str | None,
    real_bundle_sha256: str,
    synthetic_bundle_sha256: str,
    world_size: int,
) -> None:
    """Validate every field in one immutable teacher-forced-loss artifact."""

    expected_keys = {
        "schema_version",
        "config_sha256",
        "step",
        "checkpoint_visual_sha256",
        "real_validation_bundle_sha256",
        "synthetic_validation_bundle_sha256",
        "real_validation",
        "synthetic_validation",
        "real_macro_level_teacher_forced_loss",
        "synthetic_macro_level_teacher_forced_loss",
        "world_size",
    }
    if set(entry) != expected_keys:
        raise ValueError("adaptation validation artifact fields changed")
    if (
        entry.get("schema_version") != 1
        or entry.get("config_sha256") != config_sha256
        or entry.get("step") != step
        or entry.get("checkpoint_visual_sha256") != checkpoint_visual_sha256
        or entry.get("real_validation_bundle_sha256") != real_bundle_sha256
        or entry.get("synthetic_validation_bundle_sha256") != synthetic_bundle_sha256
        or entry.get("world_size") != world_size
        or world_size not in ALLOWED_WORLD_SIZES
    ):
        raise ValueError("adaptation validation artifact identity changed")

    def validate_probe(value: object, *, domain: str) -> float:
        if not isinstance(value, dict) or set(value) != {
            "per_level_teacher_forced_loss",
            "macro_level_teacher_forced_loss",
            "overall_teacher_forced_loss",
            "supervised_tokens_per_level",
        }:
            raise ValueError(f"{domain} validation probe fields changed")
        per_level = value["per_level_teacher_forced_loss"]
        tokens = value["supervised_tokens_per_level"]
        expected_levels = {str(level) for level in range(1, 6)}
        if (
            not isinstance(per_level, dict)
            or set(per_level) != expected_levels
            or not isinstance(tokens, dict)
            or set(tokens) != expected_levels
        ):
            raise ValueError(f"{domain} validation severity levels changed")
        losses: list[float] = []
        counts: list[int] = []
        for level in range(1, 6):
            raw_loss = per_level[str(level)]
            raw_tokens = tokens[str(level)]
            if (
                isinstance(raw_loss, bool)
                or not isinstance(raw_loss, (int, float))
                or not math.isfinite(float(raw_loss))
                or float(raw_loss) < 0.0
            ):
                raise ValueError(f"{domain} validation loss is invalid at L{level}")
            if (
                isinstance(raw_tokens, bool)
                or not isinstance(raw_tokens, int)
                or raw_tokens <= 0
            ):
                raise ValueError(
                    f"{domain} validation token count is invalid at L{level}"
                )
            losses.append(float(raw_loss))
            counts.append(raw_tokens)
        macro = value["macro_level_teacher_forced_loss"]
        overall = value["overall_teacher_forced_loss"]
        for name, raw in (("macro", macro), ("overall", overall)):
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0.0
            ):
                raise ValueError(f"{domain} validation {name} loss is invalid")
        expected_macro = sum(losses) / len(losses)
        expected_overall = sum(
            loss * count for loss, count in zip(losses, counts, strict=True)
        ) / sum(counts)
        if not math.isclose(float(macro), expected_macro, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{domain} validation macro loss is inconsistent")
        if not math.isclose(
            float(overall), expected_overall, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{domain} validation overall loss is inconsistent")
        return float(macro)

    real_macro = validate_probe(entry["real_validation"], domain="real")
    synthetic_macro = validate_probe(entry["synthetic_validation"], domain="synthetic")
    for name, raw, expected in (
        (
            "real",
            entry["real_macro_level_teacher_forced_loss"],
            real_macro,
        ),
        (
            "synthetic",
            entry["synthetic_macro_level_teacher_forced_loss"],
            synthetic_macro,
        ),
    ):
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or not math.isclose(float(raw), expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"top-level {name} validation loss is inconsistent")


def select_best_adaptation_checkpoint(
    entries: Sequence[Mapping[str, object]], *, allowed_steps: Sequence[int]
) -> dict[str, object]:
    allowed = set(allowed_steps)
    valid: list[dict[str, object]] = []
    seen: set[int] = set()
    for source in entries:
        entry = dict(source)
        step = entry.get("step")
        loss = entry.get("real_macro_level_teacher_forced_loss")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step not in allowed
            or step in seen
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or float(loss) < 0.0
        ):
            raise ValueError("invalid adaptation validation checkpoint evidence")
        seen.add(step)
        valid.append(entry)
    if seen != allowed:
        raise ValueError("adaptation validation evidence is incomplete")
    return min(
        valid,
        key=lambda entry: (
            float(entry["real_macro_level_teacher_forced_loss"]),
            int(entry["step"]),
        ),
    )
