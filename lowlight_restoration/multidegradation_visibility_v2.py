"""Visibility-aware deterministic v2 multi-degradation construction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from .multidegradation_data import (
    AGENTICIR_REVISION,
    CROP_SIZE,
    ROOT,
    _balanced_levels,
    _operator_parameters,
    stable_digest,
    stable_seed,
)
from .multidegradation_visibility import audit_record_components
from .multidegradation_visibility import image_delta_metrics
from .multidegradation_data import generate_record
from .qwen_multidegradation import canonical_target, make_label


def load_visibility_anchors(
    path: Path,
    *,
    metric: str,
    quantile: str,
) -> dict[tuple[str, int], float]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "pass" or payload.get("records") != 2880:
        raise ValueError("visibility anchor is not a complete training-only result")
    thresholds: dict[tuple[str, int], float] = {}
    for key, value in payload["anchors"].items():
        name, level_text = key.split(":L", maxsplit=1)
        thresholds[(name, int(level_text))] = float(value["metrics"][metric][quantile])
    if len(thresholds) != 20 or any(value <= 0 for value in thresholds.values()):
        raise ValueError("visibility anchor thresholds are incomplete")
    return thresholds


def physical_order(combo: Sequence[str], policy_order: Sequence[str]) -> tuple[str, ...]:
    combo_set = set(combo)
    if len(combo_set) != len(combo):
        raise ValueError("degradation combination contains duplicates")
    result = tuple(name for name in policy_order if name in combo_set)
    if set(result) != combo_set:
        raise ValueError("physical order does not cover the degradation combination")
    return result


def build_candidate_record(
    *,
    split: str,
    composition_size: int,
    combination_index: int,
    within_combination_index: int,
    combo: tuple[str, ...],
    ordered_sources: Sequence[Mapping[str, object]],
    attempt: int,
    seed: int,
    policy_order: Sequence[str],
    visibility_policy_name: str,
    data_root: str,
    composition_rescue: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if attempt < 0 or not ordered_sources:
        raise ValueError("candidate attempt and source pool must be valid")
    source = ordered_sources[
        (within_combination_index + attempt * 37) % len(ordered_sources)
    ]
    candidate_seed = stable_seed(
        seed,
        "visibility_v2",
        split,
        composition_size,
        combination_index,
        within_combination_index,
        attempt,
    )
    rng = random.Random(candidate_seed)
    crop_x = rng.randint(0, int(source["width"]) - CROP_SIZE)
    crop_y = rng.randint(0, int(source["height"]) - CROP_SIZE)
    levels = _balanced_levels(combo, within_combination_index)
    order = physical_order(combo, policy_order)
    operators: dict[str, object] = {}
    for operator_index, name in enumerate(order):
        operator_seed = stable_seed(candidate_seed, operator_index, name)
        parameters = _operator_parameters(
            name,
            levels[name],
            seed=operator_seed,
            source=source,
        )
        parameters["seed"] = operator_seed
        operators[name] = parameters
    rescue_evidence: dict[str, object] | None = None
    if composition_rescue is not None and attempt >= int(
        composition_rescue["start_attempt"]
    ):
        rescue_rng = random.Random(stable_seed(candidate_seed, "composition_rescue"))
        factors: dict[str, float] = {}
        if "haze" in operators and len(combo) > 1:
            beta_range = composition_rescue["haze_beta_scale_range"]
            airlight_range = composition_rescue[
                "haze_airlight_contrast_scale_range"
            ]
            beta_factor = rescue_rng.uniform(
                float(beta_range[0]), float(beta_range[1])
            )
            airlight_factor = rescue_rng.uniform(
                float(airlight_range[0]), float(airlight_range[1])
            )
            haze = operators["haze"]
            haze["beta"] = float(haze["beta"]) * beta_factor
            haze["airlight"] = 0.5 + (
                float(haze["airlight"]) - 0.5
            ) * airlight_factor
            factors["haze_beta_scale"] = beta_factor
            factors["haze_airlight_contrast_scale"] = airlight_factor
        if "noise" in operators and len(combo) > 1:
            sigma_range = composition_rescue["noise_sigma_scale_range"]
            sigma_factor = rescue_rng.uniform(
                float(sigma_range[0]), float(sigma_range[1])
            )
            noise = operators["noise"]
            noise["sigma_0_255"] = (
                float(noise["sigma_0_255"]) * sigma_factor
            )
            factors["noise_sigma_scale"] = sigma_factor
        rescue_evidence = {
            "mode": "bounded_masking_operator_attenuation_with_unchanged_visibility_gates",
            "start_attempt": int(composition_rescue["start_attempt"]),
            "factors": factors,
        }
    label = make_label(combo, levels)
    record_id = (
        f"{split[:3]}_v2_c{composition_size}_{combination_index:02d}_"
        f"{within_combination_index:04d}"
    )
    record = {
        "record_id": record_id,
        "split": split,
        "source_id": source["source_id"],
        "source_dataset": source["dataset"],
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "crop": {
            "x": crop_x,
            "y": crop_y,
            "width": CROP_SIZE,
            "height": CROP_SIZE,
        },
        "composition_size": composition_size,
        "degradations": list(label.degradations),
        "levels": label.level_map,
        "application_order": list(order),
        "operators": operators,
        "seed": candidate_seed,
        "screening_attempt": attempt,
        "target": canonical_target(label),
        "output_path": f"{data_root}/{split}/{record_id}.png",
        "agenticir_revision": AGENTICIR_REVISION,
        "visibility_policy": visibility_policy_name,
    }
    if rescue_evidence is not None:
        record["composition_rescue"] = rescue_evidence
    return record


def evaluate_candidate_visibility(
    record: Mapping[str, object],
    *,
    thresholds: Mapping[tuple[str, int], float],
    primary_metric: str,
    minimum_survival_ratio: float,
    visibility_mode: str = "final_population_anchor",
) -> tuple[bool, list[dict[str, object]]]:
    if int(record["composition_size"]) == 0:
        return True, []
    rows, full_sha = audit_record_components(
        record,
        predicted_degradations=tuple(record["degradations"]),
        require_frozen_output=False,
    )
    evidence: list[dict[str, object]] = []
    passed = True
    clean = None
    if visibility_mode == "candidate_isolated_anchor_and_final_retention":
        clean = generate_record(record, application_order=())[0]
    elif visibility_mode != "final_population_anchor":
        raise ValueError(f"unsupported v2 visibility mode: {visibility_mode}")
    for row in rows:
        key = (str(row["degradation"]), int(row["level"]))
        threshold = float(thresholds[key])
        final_value = float(row["final"][primary_metric])
        survival_field = (
            "relative_rmse_survival_ratio"
            if primary_metric == "rgb_rmse_over_full_luma"
            else "rmse_survival_ratio"
        )
        survival = float(row[survival_field])
        if visibility_mode == "candidate_isolated_anchor_and_final_retention":
            assert clean is not None
            isolated = generate_record(
                record,
                application_order=(key[0],),
            )[0]
            isolated_value = float(
                image_delta_metrics(isolated, clean)[primary_metric]
            )
            retention = final_value / isolated_value if isolated_value > 0 else 0.0
            anchor_observed = isolated_value
        else:
            retention = survival
            anchor_observed = final_value
        component_pass = (
            anchor_observed >= threshold
            and retention >= minimum_survival_ratio
        )
        passed = passed and component_pass
        evidence.append(
            {
                "degradation": key[0],
                "level": key[1],
                "position": int(row["position"]),
                "full_sha256": full_sha,
                "counterfactual_sha256": row["counterfactual_sha256"],
                "primary_metric": primary_metric,
                "final_primary_value": final_value,
                "minimum_final_primary_value": threshold,
                "survival_field": survival_field,
                "survival_ratio": survival,
                "minimum_survival_ratio": minimum_survival_ratio,
                "visibility_mode": visibility_mode,
                "anchor_observed_value": anchor_observed,
                "minimum_anchor_value": threshold,
                "retention_ratio": retention,
                "minimum_retention_ratio": minimum_survival_ratio,
                "pass": component_pass,
            }
        )
    return passed, evidence


def ordered_source_pool(
    sources: Sequence[Mapping[str, object]],
    *,
    seed: int,
    split: str,
    combo: tuple[str, ...],
) -> tuple[Mapping[str, object], ...]:
    eligible = [
        source
        for source in sources
        if "haze" not in combo or source["dataset"] == "mio100"
    ]
    return tuple(
        sorted(
            eligible,
            key=lambda source: stable_digest(
                seed, "visibility_v2_source", split, combo, source["source_id"]
            ),
        )
    )


def select_visible_candidate(
    *,
    split: str,
    composition_size: int,
    combination_index: int,
    within_combination_index: int,
    combo: tuple[str, ...],
    sources: Sequence[Mapping[str, object]],
    thresholds: Mapping[tuple[str, int], float],
    policy: Mapping[str, object],
    seed: int,
    data_root: str,
) -> dict[str, object]:
    maximum_attempts = int(policy["maximum_attempts_per_record"])
    best: dict[str, object] | None = None
    best_score = -1.0
    attempts_to_run = 1 if composition_size == 0 else maximum_attempts
    for attempt in range(attempts_to_run):
        record = build_candidate_record(
            split=split,
            composition_size=composition_size,
            combination_index=combination_index,
            within_combination_index=within_combination_index,
            combo=combo,
            ordered_sources=sources,
            attempt=attempt,
            seed=seed,
            policy_order=policy["physical_application_order"],
            visibility_policy_name=str(policy["policy_name"]),
            data_root=data_root,
            composition_rescue=(
                policy.get("composition_rescue")
                if isinstance(policy.get("composition_rescue"), Mapping)
                else None
            ),
        )
        passed, evidence = evaluate_candidate_visibility(
            record,
            thresholds=thresholds,
            primary_metric=str(policy["primary_metric"]),
            minimum_survival_ratio=float(policy["minimum_rmse_survival_ratio"]),
            visibility_mode=str(
                policy.get("visibility_mode", "final_population_anchor")
            ),
        )
        if passed:
            return {
                "accepted": True,
                "record_id": record["record_id"],
                "combination": list(combo),
                "attempts": attempt + 1,
                "record": record,
                "visibility": evidence,
            }
        margins = [
            min(
                float(component["anchor_observed_value"])
                / float(component["minimum_anchor_value"]),
                float(component["retention_ratio"])
                / float(component["minimum_retention_ratio"]),
            )
            for component in evidence
        ]
        score = min(margins)
        if score > best_score:
            best_score = score
            best = {
                "accepted": False,
                "record_id": record["record_id"],
                "combination": list(combo),
                "attempts": attempt + 1,
                "record": record,
                "visibility": evidence,
                "best_minimum_margin": score,
            }
    assert best is not None
    best["attempts"] = maximum_attempts
    return best
