import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_only_splits_are_source_disjoint():
    with (ROOT / "manifests/foundir_blur_noise_restormer_pilot_v1.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert Counter(row["split"] for row in rows) == Counter(
        {"train": 2048, "validation": 128, "internal_test": 128}
    )
    groups = defaultdict(set)
    for row in rows:
        assert row["original_split"] == "train"
        assert row["category"] == "02Blur_Noise"
        groups[row["split"]].add(row["source_group"])
    assert {split: len(values) for split, values in groups.items()} == {
        "train": 768,
        "validation": 83,
        "internal_test": 83,
    }
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["internal_test"])
    assert not (groups["validation"] & groups["internal_test"])


def test_pilot_is_simple_residual_paired_training():
    config = json.loads((ROOT / "configs/foundir_blur_noise_restormer_pilot_v1.json").read_text())
    assert config["task"] == "blur_noise"
    assert config["initial_checkpoint_name"] == "real_denoising"
    assert config["optimization"]["loss"] == "rgb_l1"
    assert config["optimization"]["identity_mix_probability"] == 0.25
    assert config["optimization"]["total_steps"] == 1000
    assert config["protected_foundir_test_accessed"] is False
