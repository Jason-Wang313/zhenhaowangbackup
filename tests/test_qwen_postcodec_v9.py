import io
import json
from pathlib import Path

import pytest
from PIL import Image

from lowlight_restoration.multidegradation_data import read_jsonl
from lowlight_restoration.qwen_input_transform import apply_qwen_input_transform
from smoke.build_qwen_multidegradation_postcodec_v9 import (
    deterministic_balanced_take,
)
from smoke.qwen_multidegradation_lora_preflight import (
    load_config,
    validate_contract,
)
from smoke.train_qwen_multidegradation_lora_ddp import gate_checks


ROOT = Path(__file__).resolve().parents[1]


def test_postcodec_transform_is_deterministic_and_changes_pixels() -> None:
    image = Image.effect_noise((64, 64), 80).convert("RGB")
    transform = {
        "family": "jpeg",
        "quality": 75,
        "subsampling": 2,
        "order": "after_target_degradations",
    }
    first = apply_qwen_input_transform(image, transform)
    second = apply_qwen_input_transform(image, transform)
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != image.tobytes()


def test_postcodec_transform_rejects_wrong_order() -> None:
    image = Image.new("RGB", (8, 8), (1, 2, 3))
    with pytest.raises(ValueError, match="must follow"):
        apply_qwen_input_transform(
            image,
            {"family": "jpeg", "quality": 75, "order": "before_target_degradations"},
        )


def test_balanced_take_round_robins_strata() -> None:
    records = [
        {"record_id": f"a{index}", "group": "a"} for index in range(4)
    ] + [{"record_id": f"b{index}", "group": "b"} for index in range(4)]
    selected = deterministic_balanced_take(
        records, 6, seed=7, salt="unit", key=lambda record: record["group"]
    )
    assert sum(record["group"] == "a" for record in selected) == 3
    assert sum(record["group"] == "b" for record in selected) == 3


def test_v9_data_contract_never_names_foundir_or_test() -> None:
    config = json.loads(
        (ROOT / "configs/qwen_multidegradation_postcodec_v9_data.json").read_text()
    )
    assert "foundir" not in json.dumps(config).lower()
    assert "test" not in json.dumps(config).lower()


def test_frozen_v9_manifest_has_balanced_postcodec_noise_and_no_test() -> None:
    records = read_jsonl(
        ROOT / "manifests/qwen_multidegradation_postcodec_v9.jsonl"
    )
    assert len(records) == 32400
    assert {record["split"] for record in records} == {"train", "validation"}
    for split, expected in (("train", 4800), ("validation", 1200)):
        post = [
            record
            for record in records
            if record["split"] == split
            and record["evaluation_domain"] == "post_target_jpeg"
        ]
        assert sum("noise" in record["degradations"] for record in post) == expected
        assert sum("noise" not in record["degradations"] for record in post) == expected
        assert {record["input_transform"]["quality"] for record in post} == {
            75,
            85,
            94,
            99,
        }
        assert all(
            record["non_target_nuisance"]["family"] == "none"
            and record["input_transform"]["order"] == "after_target_degradations"
            and "foundir" not in record["output_path"].lower()
            for record in post
        )


def test_v9_training_contract_is_hash_bound_and_test_free() -> None:
    config, _ = load_config(ROOT / "configs/qwen_multidegradation_postcodec_v9.json")
    validate_contract(config)
    assert config["data"]["fixed_test_split"] is None
    assert config["selection"]["data"] == "synthetic_validation_only_no_test"


def test_domain_gate_fails_postcodec_noise_recall_without_hiding_overall_pass() -> None:
    config, _ = load_config(ROOT / "configs/qwen_multidegradation_postcodec_v9.json")
    base = {
        "valid_structured_output_rate": 1.0,
        "type_macro_f1": 1.0,
        "type_micro_f1": 1.0,
        "degradation_set_exact_match_rate": 1.0,
        "combined_label_exact_match_rate": 1.0,
        "severity_typed_accuracy": 1.0,
        "severity_typed_mae_with_missing_penalty": 0.0,
        "severity_typed_mse_with_missing_penalty": 0.0,
        "per_type_valid_response_rate": 1.0,
        "per_type_level_accuracy": 1.0,
        "by_composition_size": {
            "3": {
                "degradation_set_exact_match_rate": 1.0,
                "combined_label_exact_match_rate": 1.0,
            }
        },
        "per_type": {"noise": {"recall": 1.0, "precision": 1.0}},
    }
    metrics = dict(base)
    metrics["by_evaluation_domain"] = {
        "v8_original": dict(base),
        "post_target_jpeg": {
            **base,
            "per_type": {"noise": {"recall": 0.84, "precision": 1.0}},
        },
    }
    checks = gate_checks(metrics, config, True, split="validation")
    assert all(value for key, value in checks.items() if key != "domain_post_target_jpeg_noise_recall")
    assert checks["domain_post_target_jpeg_noise_recall"] is False
