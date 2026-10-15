from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from lowlight_restoration.qwen_eleven_class_gate import (
    CLASSES,
    LABEL_BY_CATEGORY,
    MIXED_CLASSES,
    SINGLETON_CLASSES,
    consensus_metrics,
    consensus_route,
    eleven_class_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_eleven_observed_class_identities_are_frozen():
    assert CLASSES == (
        "blur", "noise", "haze", "rain", "low_light", "blur+noise",
        "haze+low_light", "rain+haze", "blur+low_light", "noise+low_light",
        "blur+noise+low_light",
    )
    assert SINGLETON_CLASSES == CLASSES[:5]
    assert MIXED_CLASSES == CLASSES[5:]
    assert len(LABEL_BY_CATEGORY) == 11


def test_eleven_class_metrics_count_wrong_and_refusal_separately():
    correct = [0.001] * 11
    correct[0] = 0.99
    wrong = [0.001] * 11
    wrong[1] = 0.99
    refused = [1 / 11] * 11
    metrics = eleven_class_metrics([correct, wrong, refused], [0, 0, 2], threshold=0.9)
    assert metrics["overall"]["correct"] == 1
    assert metrics["overall"]["wrong"] == 1
    assert metrics["overall"]["refusal"] == 1
    assert metrics["per_class"]["blur"]["wrong"] == 1
    assert set(metrics["per_class"]) == set(CLASSES)


def test_consensus_routes_mixed_directly_but_requires_singleton_agreement():
    five = [0.01, 0.96, 0.01, 0.01, 0.01]
    eleven_mixed = [0.001] * 11
    eleven_mixed[5] = 0.97
    assert consensus_route(five, eleven_mixed) == "blur+noise"
    eleven_noise = [0.001] * 11
    eleven_noise[1] = 0.97
    assert consensus_route(five, eleven_noise) == "noise"
    conflicting_five = [0.96, 0.01, 0.01, 0.01, 0.01]
    assert consensus_route(conflicting_five, eleven_noise) is None
    eleven_noise[1] = 0.90
    assert consensus_route(five, eleven_noise) == "noise"


def test_consensus_metrics_keep_refusal_separate_from_wrong():
    metrics = consensus_metrics(["blur", "noise", None], ["blur", "blur", "haze"])
    assert metrics["overall"]["correct"] == 1
    assert metrics["overall"]["wrong"] == 1
    assert metrics["overall"]["refusal"] == 1


def test_frozen_partition_is_balanced_and_source_disjoint():
    path = ROOT / "manifests/foundir_qwen_eleven_class_train_only_v1.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6930
    expected = {"fit": 450, "calibration": 90, "holdout": 90}
    counts = Counter((row["head_split"], row["label"]) for row in rows)
    assert all(counts[(split, label)] == quota for split, quota in expected.items() for label in CLASSES)
    locations = defaultdict(set)
    for row in rows:
        locations[row["source_group"]].add(row["head_split"])
    assert all(len(splits) == 1 for splits in locations.values())
    assert len({row["record_id"] for row in rows}) == len(rows)


def test_fresh_holdout_v2_is_balanced_new_and_source_disjoint():
    with (ROOT / "manifests/foundir_qwen_eleven_class_train_only_v1.csv").open(newline="") as handle:
        v1 = list(csv.DictReader(handle))
    with (ROOT / "manifests/foundir_qwen_eleven_class_fresh_holdout_v2.csv").open(newline="") as handle:
        v2 = list(csv.DictReader(handle))
    fresh = [row for row in v2 if row["head_split"] == "holdout"]
    assert len(fresh) == 440
    assert Counter(row["label"] for row in fresh) == Counter({label: 40 for label in CLASSES})
    assert {row["source_group"] for row in fresh}.isdisjoint({row["source_group"] for row in v1})
    locations = defaultdict(set)
    for row in v2:
        locations[row["source_group"]].add(row["head_split"])
    assert all(len(splits) == 1 for splits in locations.values())
