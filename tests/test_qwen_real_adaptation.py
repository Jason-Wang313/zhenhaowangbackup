from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lowlight_restoration.data import PairRecord, load_pair_manifest
from lowlight_restoration.qwen_real_adaptation import (
    balanced_real_group,
    global_training_groups,
    load_adaptation_config,
    partition_training_groups,
    records_by_real_split,
    select_best_adaptation_checkpoint,
    select_synthetic_replay,
    training_schedule_evidence,
    validate_adaptation_contract,
    validate_adaptation_validation_entry,
)
from lowlight_restoration.qwen_supervised import record_bundle_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/qwen_lowlight_real_adaptation_step600.json"


def record(
    pair_id: str, source_id: str, level: int, split: str = "train"
) -> PairRecord:
    digest = f"{level}" * 64
    return PairRecord(
        pair_id=pair_id,
        source_id=source_id,
        clean_path=Path("/tmp/clean.png"),
        degraded_path=Path("/tmp/degraded.png"),
        project_level=f"L{level}",
        seed=7,
        split=split,
        clean_sha256=digest,
        degraded_sha256=digest,
    )


def fixture_records() -> tuple[list[PairRecord], list[PairRecord]]:
    real = [
        record(f"real_L{level}_{index}", f"scene_L{level}_{index}", level)
        for level in range(1, 6)
        for index in range(3)
    ]
    synthetic = [
        record(f"synthetic_{source}_L{level}", f"synthetic_{source}", level)
        for source in range(6)
        for level in range(1, 6)
    ]
    return real, synthetic


def test_checked_config_and_real_manifest_contract() -> None:
    config, _, _ = load_adaptation_config(CONFIG)
    validate_adaptation_contract(config)
    real_config = config["data"]["real"]
    records = load_pair_manifest(ROOT / real_config["manifest"])
    splits = records_by_real_split(records, config)
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 673,
        "validation": 143,
        "test": 145,
    }
    assert record_bundle_sha256(records) == real_config["record_bundle_sha256"]


def test_contract_rejects_world_size_semantic_drift() -> None:
    config, _, _ = load_adaptation_config(CONFIG)
    changed = copy.deepcopy(config)
    changed["distributed"]["gradient_accumulation_by_world_size"]["2"] = 1
    with pytest.raises(ValueError, match="world-size compatibility"):
        validate_adaptation_contract(changed)


def test_real_groups_are_balanced_and_source_distinct() -> None:
    real, _ = fixture_records()
    by_level = {
        level: [value for value in real if value.level_index == level]
        for level in range(1, 6)
    }
    group = balanced_real_group(by_level, group_index=5, seed=123)
    assert [value.level_index for value in group] == [1, 2, 3, 4, 5]
    assert len({value.source_id for value in group}) == 5


def test_global_schedule_is_identical_for_two_and_eight_ranks() -> None:
    real, synthetic = fixture_records()
    groups = global_training_groups(real, synthetic, step=2, seed=123)
    expected = {
        (domain, tuple(value.pair_id for value in values)) for domain, values in groups
    }
    assert len(groups) == 8
    assert [domain for domain, _ in groups].count("real") == 4
    assert [domain for domain, _ in groups].count("synthetic_replay") == 4
    for world_size, accumulation in ((2, 4), (8, 1)):
        realized = {
            (domain, tuple(value.pair_id for value in values))
            for rank in range(world_size)
            for domain, values in partition_training_groups(
                groups, world_size=world_size, rank=rank
            )
        }
        assert realized == expected
        assert all(
            len(partition_training_groups(groups, world_size=world_size, rank=rank))
            == accumulation
            for rank in range(world_size)
        )


def test_schedule_evidence_is_balanced_and_stable() -> None:
    real, synthetic = fixture_records()
    first = training_schedule_evidence(real, synthetic, total_steps=3, seed=123)
    second = training_schedule_evidence(real, synthetic, total_steps=3, seed=123)
    assert first == second
    assert first["total_sample_presentations"] == 120
    assert first["domain_presentations"] == {
        "real": 60,
        "synthetic_replay": 60,
    }
    assert set(first["domain_level_presentations"].values()) == {12}


def test_replay_selection_and_checkpoint_selection() -> None:
    _, synthetic = fixture_records()
    validation = [
        PairRecord(**{**value.__dict__, "split": "validation"}) for value in synthetic
    ]
    replay = select_synthetic_replay(validation, source_count=3, seed=17)
    assert len(replay) == 15
    assert {value.level_index for value in replay} == set(range(1, 6))
    selected = select_best_adaptation_checkpoint(
        [
            {"step": 25, "real_macro_level_teacher_forced_loss": 0.4},
            {"step": 50, "real_macro_level_teacher_forced_loss": 0.2},
            {"step": 100, "real_macro_level_teacher_forced_loss": 0.2},
        ],
        allowed_steps=[25, 50, 100],
    )
    assert selected["step"] == 50


def validation_entry() -> dict[str, object]:
    def probe(offset: float) -> dict[str, object]:
        losses = {str(level): offset + level / 10 for level in range(1, 6)}
        tokens = {str(level): 10 + level for level in range(1, 6)}
        return {
            "per_level_teacher_forced_loss": losses,
            "macro_level_teacher_forced_loss": sum(losses.values()) / 5,
            "overall_teacher_forced_loss": sum(
                losses[level] * tokens[level] for level in losses
            )
            / sum(tokens.values()),
            "supervised_tokens_per_level": tokens,
        }

    real = probe(0.0)
    synthetic = probe(0.5)
    return {
        "schema_version": 1,
        "config_sha256": "a" * 64,
        "step": 25,
        "checkpoint_visual_sha256": "b" * 64,
        "real_validation_bundle_sha256": "c" * 64,
        "synthetic_validation_bundle_sha256": "d" * 64,
        "real_validation": real,
        "synthetic_validation": synthetic,
        "real_macro_level_teacher_forced_loss": real["macro_level_teacher_forced_loss"],
        "synthetic_macro_level_teacher_forced_loss": synthetic[
            "macro_level_teacher_forced_loss"
        ],
        "world_size": 2,
    }


def validate_entry(entry: dict[str, object]) -> None:
    validate_adaptation_validation_entry(
        entry,
        config_sha256="a" * 64,
        step=25,
        checkpoint_visual_sha256="b" * 64,
        real_bundle_sha256="c" * 64,
        synthetic_bundle_sha256="d" * 64,
        world_size=2,
    )


def test_validation_artifact_checks_every_numeric_field() -> None:
    entry = validation_entry()
    validate_entry(entry)

    corruptions = []
    changed = copy.deepcopy(entry)
    changed["real_validation"]["per_level_teacher_forced_loss"]["3"] = float("nan")
    corruptions.append(changed)
    changed = copy.deepcopy(entry)
    changed["synthetic_validation"]["supervised_tokens_per_level"]["4"] = 0
    corruptions.append(changed)
    changed = copy.deepcopy(entry)
    changed["real_validation"]["macro_level_teacher_forced_loss"] = 99.0
    corruptions.append(changed)
    changed = copy.deepcopy(entry)
    changed["synthetic_macro_level_teacher_forced_loss"] = 99.0
    corruptions.append(changed)

    for changed in corruptions:
        with pytest.raises(ValueError):
            validate_entry(changed)
