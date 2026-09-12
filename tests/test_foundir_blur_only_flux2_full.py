from __future__ import annotations

import json
from pathlib import Path

import pytest

from smoke.train_foundir_blur_only_flux2_full_ddp import (
    BlurPresentationDataset,
    RankPositionSampler,
    load_manifests,
    snapshot_complete,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/foundir_blur_only_flux2_full_v1.json"


def loaded_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text())


def test_config_locks_advisor_authorized_scope() -> None:
    config = loaded_config()
    validate_config(config)
    assert config["scope"] == {
        "category": "01Blur",
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "official_foundir_test_accessed": False,
        "sealed_internal_accessed": False,
    }
    assert config["lora"]["initialization"] == "fresh_zero_lora"
    assert config["training"]["global_batch_size"] == 32
    assert config["distributed"]["allowed_gpu_name_fragments"] == ["H100", "H200"]
    assert config["distributed"]["hardware_profiles"] == {
        "H100": {"micro_batch_size_per_rank": 2, "gradient_accumulation_steps": 2},
        "H200": {"micro_batch_size_per_rank": 4, "gradient_accumulation_steps": 1},
    }
    assert config["inference"]["automatic_checkpoint_selection"] is False


@pytest.mark.parametrize("key", ["qwen_degradation_conditioning", "mixed_degradations", "pixel_refiner"])
def test_config_rejects_goal_drift(key: str) -> None:
    config = loaded_config()
    config["scope"][key] = True
    with pytest.raises(RuntimeError, match="scope changed"):
        validate_config(config)


def test_full_and_fixed_manifests_are_source_group_disjoint() -> None:
    config = loaded_config()
    train, fixed = load_manifests(config, verify_paths=False)
    assert len(train) == 105057
    assert len(fixed) == 16
    assert not ({row["source_group"] for row in train} & {row["source_group"] for row in fixed})
    assert {row["category"] for row in [*train, *fixed]} == {"01Blur"}
    assert {row["composition_size"] for row in [*train, *fixed]} == {"1"}


def test_rank_sampler_exactly_partitions_global_presentations() -> None:
    values = []
    for rank in range(8):
        local = list(RankPositionSampler(start=64, stop=320, rank=rank, world=8))
        assert len(local) == 32
        values.extend(local)
    assert sorted(values) == list(range(64, 320))


def test_dataset_mapping_and_crop_are_deterministic() -> None:
    config = loaded_config()
    train, _ = load_manifests(config, verify_paths=False)
    dataset = BlurPresentationDataset(train[:32], crop_size=512, seed=17)
    first = dataset[19]
    second = dataset[19]
    assert first["record_id"] == second["record_id"]
    assert first["position"] == second["position"] == 19
    assert first["lq"].shape == first["gt"].shape == (3, 512, 512)
    assert first["lq"].equal(second["lq"])
    assert first["gt"].equal(second["gt"])


def test_snapshot_requires_metrics_and_every_triptych(tmp_path: Path) -> None:
    directory = tmp_path / "fixed_validation" / "training_triptychs" / "step_002000"
    directory.mkdir(parents=True)
    (directory / "metrics.json").write_text("{}\n")
    for index in range(15):
        (directory / f"{index:02d}.png").write_bytes(b"png")
    assert not snapshot_complete(tmp_path, 2000, 16)
    (directory / "15.png").write_bytes(b"png")
    assert snapshot_complete(tmp_path, 2000, 16)


def test_launchers_enforce_governance_and_stop_for_visual_review() -> None:
    runner = (ROOT / "smoke/run_foundir_blur_only_flux2_full_v1_8gpu.sh").read_text()
    launcher = (ROOT / "smoke/launch_foundir_blur_only_flux2_full_v1_8gpu.sh").read_text()
    assert "advisor_governance.py\" check" in runner
    assert "--require-decision" in runner
    assert "STOP_FOR_ADVISOR_VISUAL_REVIEW" in runner
    assert "CUDA_VISIBLE_DEVICES exactly to 0,1,2,3,4,5,6,7" in launcher
