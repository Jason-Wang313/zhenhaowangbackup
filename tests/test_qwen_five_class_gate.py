from __future__ import annotations

from lowlight_restoration.qwen_five_class_gate import (
    CLASSES,
    build_balanced_partition,
    image_embedding_sequence,
    routing_metrics,
)

import torch


def _rows():
    rows = []
    categories = ["01Blur", "05Noise", "08Haze", "10Rain", "14Lowlight"]
    for category in categories:
        for index in range(500):
            rows.append(
                {
                    "record_id": f"{category}-{index}",
                    "split": "train",
                    "category": category,
                    "composition_size": "1",
                    "source_group": f"{category}-source-{index}",
                    "lq_path": f"data/{category}/{index}.png",
                    "lq_sha256": "a" * 64,
                    "width": "32",
                    "height": "32",
                }
            )
    return rows


def test_partition_is_balanced_deterministic_and_source_disjoint():
    quotas = {"fit": 20, "calibration": 10, "holdout": 10}
    first = build_balanced_partition(_rows(), seed=7, records_per_class=quotas)
    second = build_balanced_partition(list(reversed(_rows())), seed=7, records_per_class=quotas)
    assert first == second
    locations = {}
    for row in first:
        locations.setdefault(row["source_group"], set()).add(row["head_split"])
    assert all(len(value) == 1 for value in locations.values())


def test_routing_metrics_counts_abstention_separately():
    probabilities = [
        [0.9, 0.025, 0.025, 0.025, 0.025],
        [0.1, 0.6, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.3, 0.2, 0.2],
    ]
    metrics = routing_metrics(probabilities, [0, 0, 2], threshold=0.7)
    assert metrics["overall"]["correct"] == 1
    assert metrics["overall"]["wrong"] == 0
    assert metrics["overall"]["refusal"] == 2
    assert set(metrics["per_class"]) == set(CLASSES)


def test_transformers_v5_image_feature_output_is_unwrapped():
    class Output:
        pooler_output = (torch.ones(3, 4), torch.zeros(2, 4))

    realized = image_embedding_sequence(Output())
    assert len(realized) == 2
    assert realized[0].shape == (3, 4)
