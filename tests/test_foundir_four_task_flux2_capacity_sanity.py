from __future__ import annotations

import json
from pathlib import Path

import pytest

from smoke.aggregate_foundir_four_task_flux2_capacity_sanity import aggregate
from smoke.freeze_foundir_four_task_flux2_capacity_sanity import select_rows
from smoke.train_foundir_blur_only_flux2_overfit_ddp import materialize_task_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_four_task_flux2_capacity_sanity_v1.json"


def loaded() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


@pytest.mark.parametrize("task", ["05Noise", "08Haze", "10Rain", "14Lowlight"])
def test_task_materialization_locks_scope_and_batch(task: str) -> None:
    config = materialize_task_config(loaded(), task)
    config["_config_path"] = str(CONFIG)
    validate_config(config)
    assert config["scope"]["category"] == task
    assert config["scope"]["qwen_degradation_conditioning"] is False
    assert config["scope"]["mixed_degradations"] is False
    assert config["distributed"]["required_world_size"] == 2
    assert config["training"]["global_batch_size"] == 8
    assert 2 * config["training"]["micro_batch_size_per_rank"] * config["training"]["gradient_accumulation_steps"] == 8


def test_frozen_selection_is_balanced_and_source_disjoint() -> None:
    rows = select_rows(loaded())
    assert len(rows) == 128
    for task in loaded()["scope"]["categories"]:
        train = [row for row in rows if row["category"] == task and row["role"] == "tiny_overfit_train"]
        validation = [row for row in rows if row["category"] == task and row["role"] == "descriptive_validation"]
        assert len(train) == len(validation) == 16
        assert len({row["source_group"] for row in train}) == 16
        assert len({row["source_group"] for row in validation}) == 16
        assert not ({row["source_group"] for row in train} & {row["source_group"] for row in validation})
        assert {row["composition_size"] for row in [*train, *validation]} == {"1"}


@pytest.mark.parametrize("key", ["qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner"])
def test_task_config_rejects_scope_drift(key: str) -> None:
    config = materialize_task_config(loaded(), "05Noise")
    config["scope"][key] = True
    with pytest.raises(RuntimeError, match="scope contract changed"):
        validate_config(config)


def test_runner_parallelizes_four_isolated_arms_and_stops() -> None:
    runner = (ROOT / "smoke/run_foundir_four_task_flux2_capacity_sanity_v1_8gpu.sh").read_text()
    launcher = (ROOT / "smoke/launch_foundir_four_task_flux2_capacity_sanity_v1_8gpu.sh").read_text()
    assert "tasks=(05Noise 08Haze 10Rain 14Lowlight)" in runner
    assert "devices=(0,1 2,3 4,5 6,7)" in runner
    assert "--nproc_per_node=2" in runner
    assert "--require-decision" in runner
    assert "no_Qwen_degradation no_mixed" in runner
    assert "nohup setsid" in launcher
    assert "automatic_joint_launch" not in runner


def test_aggregate_rejects_missing_runtime_results(tmp_path: Path, monkeypatch) -> None:
    import smoke.aggregate_foundir_four_task_flux2_capacity_sanity as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    config = loaded()
    config["runtime"]["output_root"] = "out"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="missing terminal task result"):
        aggregate(path)
