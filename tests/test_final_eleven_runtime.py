from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from safetensors import safe_open

from lowlight_restoration.final_eleven_runtime import FrozenFinalElevenRouter


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_frozen_runtime_replays_final_development_decisions() -> None:
    singleton = _rows("manifests/foundir_qwen_five_class_development_v1.csv")
    mixed = _rows("manifests/foundir_qwen_mixed_consensus_development_v3.csv")
    by_id = {row["record_id"]: row for row in singleton + mixed}
    decisions = _rows(
        "checkpoints/foundir_final_eleven_route_cpu_gate_v1/development_decisions.csv"
    )

    # Deterministic spread across the complete, sorted development decision table.
    indices = np.linspace(0, len(decisions) - 1, num=33, dtype=int).tolist()
    selected = [decisions[index] for index in indices]
    features = []
    image_paths = []
    for decision in selected:
        record_id = decision["record_id"]
        row = by_id[record_id]
        directory = (
            "checkpoints/foundir_qwen_five_class_development_v1/features"
            if row["label"] in {"blur", "noise", "haze", "rain", "low_light"}
            else "checkpoints/foundir_qwen_mixed_consensus_development_v3/features"
        )
        with safe_open(
            ROOT / directory / f"{record_id}.safetensors",
            framework="pt",
            device="cpu",
        ) as handle:
            features.append(handle.get_tensor("feature").numpy())
        image_paths.append(ROOT / row["lq_path"])

    realized = FrozenFinalElevenRouter().route(np.stack(features), image_paths)
    assert [item["route"] or "abstain" for item in realized] == [
        item["final_route"] for item in selected
    ]
