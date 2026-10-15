from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

import smoke.train_foundir_task_native_restormer_adaptation_ddp as trainer
from lowlight_restoration.flux_prompt_cache import file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _staged_config(tmp_path: Path, monkeypatch, task: str = "noise") -> tuple[Path, dict]:
    config = json.loads(
        (REPOSITORY_ROOT / f"configs/foundir_{task}_restormer_adaptation_v1.json").read_text()
    )
    checkpoint = tmp_path / config["initial_checkpoint"]
    manifest = tmp_path / config["manifest"]
    capacity = tmp_path / config["capacity_result"]
    for path in (checkpoint, manifest, capacity):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    candidate = f"{task}:{config['initial_checkpoint_name']}"
    capacity.write_text(
        json.dumps(
            {
                "status": "task_native_capacity_audit_complete",
                "candidate_capacity": {candidate: {"strong_gate_pass": False}},
                "sealed_internal_test_accessed": False,
                "protected_foundir_test_accessed": False,
            }
        )
    )
    expected_hashes = {
        checkpoint: config["initial_checkpoint_sha256"],
        manifest: config["manifest_sha256"],
        capacity: config["capacity_result_sha256"],
    }
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    monkeypatch.setattr(trainer, "file_sha256", lambda path: expected_hashes[Path(path)])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, config


def test_config_validation_locks_predeclared_training_and_gate_contract(
    tmp_path, monkeypatch
) -> None:
    config_path, config = _staged_config(tmp_path, monkeypatch)
    assert trainer.validate_config(config_path)["task"] == "noise"

    config["strong_gate"]["mean_psnr_minimum"] = 38.9
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="contract changed"):
        trainer.validate_config(config_path)


def test_task_rows_require_exact_split_record_group_and_id_coverage(
    tmp_path, monkeypatch
) -> None:
    rows = [
        {"record_id": "t1", "category": "05Noise", "split": "train", "source_group": "ta"},
        {"record_id": "t2", "category": "05Noise", "split": "train", "source_group": "tb"},
        {"record_id": "v1", "category": "05Noise", "split": "validation", "source_group": "va"},
        {
            "record_id": "i1",
            "category": "05Noise",
            "split": "internal_test",
            "source_group": "ia",
        },
        {"record_id": "other", "category": "08Haze", "split": "train", "source_group": "x"},
    ]
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contracts = copy.deepcopy(trainer.TASK_CONTRACTS)
    contracts["noise"]["records"] = {"train": 2, "validation": 1, "internal_test": 1}
    contracts["noise"]["groups"] = {"train": 2, "validation": 1, "internal_test": 1}
    monkeypatch.setattr(trainer, "TASK_CONTRACTS", contracts)
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    selected = trainer.load_task_rows(
        {
            "task": "noise",
            "category": "05Noise",
            "manifest": "manifest.csv",
            "manifest_expected_records": len(rows),
        }
    )
    assert {split: len(items) for split, items in selected.items()} == {
        "train": 2,
        "validation": 1,
        "internal_test": 1,
    }

    rows[3]["record_id"] = "v1"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="record IDs are not unique"):
        trainer.load_task_rows(
            {
                "task": "noise",
                "category": "05Noise",
                "manifest": "manifest.csv",
                "manifest_expected_records": len(rows),
            }
        )


def test_source_group_sampling_is_deterministic_and_group_uniform() -> None:
    grouped = {
        "small": [{"record_id": "s"}],
        "large": [{"record_id": f"l{index}"} for index in range(20)],
    }
    first = trainer.sample_global_batch(
        grouped, task="noise", step=7, seed=1234, count=4000
    )
    second = trainer.sample_global_batch(
        grouped, task="noise", step=7, seed=1234, count=4000
    )
    assert first == second
    counts = {name: 0 for name in grouped}
    for row, _ in first:
        counts["small" if row["record_id"] == "s" else "large"] += 1
    assert abs(counts["small"] - counts["large"]) < 200
    assert first != trainer.sample_global_batch(
        grouped, task="noise", step=8, seed=1234, count=4000
    )


def test_crop_and_learning_rate_schedules_have_exact_boundaries() -> None:
    config = {
        "optimization": {
            "progressive_crops": [[0, 128], [1000, 192], [2500, 256]],
            "learning_rate": 5e-5,
            "minimum_learning_rate": 1e-6,
            "warmup_steps": 100,
            "total_steps": 3000,
        }
    }
    assert trainer.crop_size_for_step(config, 999) == 128
    assert trainer.crop_size_for_step(config, 1000) == 192
    assert trainer.crop_size_for_step(config, 2500) == 256
    assert trainer.learning_rate(config, 1) == pytest.approx(5e-7)
    assert trainer.learning_rate(config, 100) == pytest.approx(5e-5)
    assert trainer.learning_rate(config, 3000) == pytest.approx(1e-6)
    with pytest.raises(ValueError, match="outside"):
        trainer.learning_rate(config, 0)


def test_selection_uses_worst_margin_then_earlier_step_and_verifies_checkpoint(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(trainer, "ROOT", tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    config = {
        "task": "noise",
        "selection_rule": "maximize_worst_normalized_strong_gate_margin_then_earlier_step",
        "optimization": {"milestones": [10, 20]},
    }
    for step in (10, 20):
        checkpoint = output / f"step_{step:06d}.pt"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        payload = {
            "status": "foundir_task_native_adaptation_development_milestone",
            "config_sha256": "config",
            "task": "noise",
            "split": "validation",
            "step": step,
            "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
            "checkpoint_sha256": file_sha256(checkpoint),
            "metrics": {"psnr": 40.0, "lpips": 0.05},
            "source_group_bootstrap": {},
            "strong_gate_checks": {},
            "strong_gate_pass": True,
            "selection_score": 0.25,
            "capacity_only_not_final_deployable_routing": True,
            "sealed_internal_test_accessed": False,
            "protected_foundir_test_accessed": False,
        }
        (output / f"development_step_{step:06d}.json").write_text(json.dumps(payload))
    selection = trainer.finalize_selection(config, "config", output)
    assert selection["selected_step"] == 10

    (output / "step_000010.pt").write_bytes(b"changed")
    (output / "development_selection.json").unlink()
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        trainer.finalize_selection(config, "config", output)
