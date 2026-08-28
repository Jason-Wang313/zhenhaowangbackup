"""Deterministic source splitting and AgenticIR-based pilot construction."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.io import loadmat

from .qwen_multidegradation import DEGRADATIONS, canonical_target, make_label


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
SPLIT_SOURCE_COUNTS = {
    "mio100": {"train": 80, "validation": 10, "test": 10},
    "div2k": {"train": 640, "validation": 80, "test": 80},
    "lsdir": {"train": 880, "validation": 110, "test": 110},
}
SPLIT_COMPOSITION_COUNTS = {
    "train": {0: 960, 1: 2880, 2: 2880, 3: 2880},
    "validation": {0: 120, 1: 360, 2: 360, 3: 360},
    "test": {0: 120, 1: 360, 2: 360, 3: 360},
}
CROP_SIZE = 448
CONTRACT_SEED = 2026082401
AGENTICIR_REVISION = "9640a291480dee3ba8f2974125d4ee9e3440f3d6c"

LOWLIGHT_RATIOS = {2: 0.50, 3: 0.38, 4: 0.30, 5: 0.14}
LOWLIGHT_DST_MEANS = {2: 230, 3: 190, 4: 150, 5: 120}
NOISE_SIGMA_RANGES = {
    2: (16.848, 49.9824),
    3: (37.575, 75.15),
    4: (75.0, 105.0),
    5: (159.6, 199.5),
}
BLUR_PARAMETERS = {2: (10, 12.0), 3: (20, 18.0), 4: (30, 30.0), 5: (30, 48.0)}
# Human-approved 2026-08-28 motion-blur ladder.  Keep the legacy profile above
# immutable so old manifests remain exactly reproducible.
BLUR_PARAMETERS_EXTREME_V6 = {
    2: (22, 18.0),
    3: (30, 24.0),
    4: (90, 72.0),
    5: (200, 160.0),
}
HAZE_A_MEANS = {2: 0.60, 3: 0.70, 4: 0.80, 5: 1.00}
HAZE_BETA_MEANS = {2: 2.4, 3: 3.6, 4: 7.2, 5: 10.8}
RAIN_DISTRIBUTIONS = {2: (5.0, 5.0), 3: (60.0, 5.0), 4: (150.0, 10.0), 5: (340.0, 20.0)}
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def stable_digest(*parts: object) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()


def stable_seed(*parts: object) -> int:
    return int(stable_digest(*parts)[:16], 16) % (2**32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _verified_depth_sha256(path_text: str) -> str:
    """Hash each immutable aligned depth file once per construction process."""

    return sha256(Path(path_text))


def load_source_inventory(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if payload.get("verified_unique_sources") != 2000 or not payload.get(
        "ready_for_10000_pair_pilot"
    ):
        raise ValueError("clean source inventory is not the verified 2,000-source pool")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2000:
        raise ValueError("clean source inventory cardinality changed")
    required = {"source_id", "dataset", "path", "sha256", "width", "height"}
    for record in records:
        if not isinstance(record, dict) or not required.issubset(record):
            raise ValueError("malformed clean source record")
        if int(record["width"]) < CROP_SIZE or int(record["height"]) < CROP_SIZE:
            raise ValueError(f"source is smaller than crop: {record['source_id']}")
    return records


def assign_source_splits(
    records: Sequence[Mapping[str, object]], *, seed: int = CONTRACT_SEED
) -> dict[str, tuple[dict[str, object], ...]]:
    """Create the exact deterministic 80/10/10 split within every dataset."""

    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_dataset[str(record["dataset"])].append(dict(record))
    assignments: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
    for dataset, counts in SPLIT_SOURCE_COUNTS.items():
        candidates = sorted(
            by_dataset[dataset],
            key=lambda record: stable_digest(seed, dataset, record["source_id"]),
        )
        if len(candidates) != sum(counts.values()):
            raise ValueError(f"unexpected {dataset} source count: {len(candidates)}")
        offset = 0
        for split in SPLITS:
            count = counts[split]
            assignments[split].extend(candidates[offset : offset + count])
            offset += count
    result = {
        split: tuple(
            sorted(values, key=lambda record: stable_digest(seed, split, record["source_id"]))
        )
        for split, values in assignments.items()
    }
    seen: dict[str, str] = {}
    for split, values in result.items():
        for record in values:
            source_id = str(record["source_id"])
            if source_id in seen:
                raise RuntimeError(f"source crosses splits: {source_id}")
            seen[source_id] = split
    if len(seen) != 2000:
        raise RuntimeError("source assignment did not consume the exact inventory")
    return result


def _balanced_levels(combo: tuple[str, ...], index: int) -> dict[str, int]:
    return {
        name: 2 + ((index + DEGRADATIONS.index(name) + position) % 4)
        for position, name in enumerate(combo)
    }


def _operator_parameters(
    name: str,
    level: int,
    *,
    seed: int,
    source: Mapping[str, object],
    blur_parameters: Mapping[int, tuple[int, float]] = BLUR_PARAMETERS,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    if name == "low_light":
        dst_max = int(np.clip(np.rint(rng.normal(LOWLIGHT_DST_MEANS[level], 4)), 80, 255))
        return {
            "operator": "AgenticIR.darken_linear_mapping_plus_luma_calibration",
            "agenticir_level": level - 1,
            "dst_max": dst_max,
            "target_luma_ratio": LOWLIGHT_RATIOS[level],
        }
    if name == "noise":
        low, high = NOISE_SIGMA_RANGES[level]
        return {
            "operator": "AgenticIR.color_gaussian_noise_equivalent",
            "agenticir_level": level - 1,
            "sigma_0_255": float(rng.uniform(low, high)),
        }
    if name == "blur":
        radius, sigma = blur_parameters[level]
        return {
            "operator": "AgenticIR.motion_blur_project_strength",
            "agenticir_level": level - 1,
            "radius": radius,
            "sigma": sigma,
            "angle_degrees": float(rng.uniform(-90, 90)),
        }
    if name == "haze":
        explicit_depth = source.get("depth_path")
        if explicit_depth is not None:
            depth = Path(str(explicit_depth))
            declared_depth_sha = source.get("depth_sha256")
            if not isinstance(declared_depth_sha, str) or len(declared_depth_sha) != 64:
                raise ValueError("explicit haze depth is missing its frozen SHA-256")
        else:
            if source["dataset"] != "mio100":
                raise ValueError(
                    "haze requires a source with a frozen aligned depth map"
                )
            source_stem = Path(str(source["path"])).stem
            depth = Path("data/agenticir/haze") / source_stem / "predict_depth.mat"
            declared_depth_sha = None
        absolute_depth = ROOT / depth
        if not absolute_depth.is_file():
            raise FileNotFoundError(absolute_depth)
        observed_depth_sha = _verified_depth_sha256(str(absolute_depth.resolve()))
        if declared_depth_sha is not None and observed_depth_sha != declared_depth_sha:
            raise RuntimeError("explicit haze depth SHA-256 changed")
        return {
            "operator": "AgenticIR.atmospheric_scattering_project_strength",
            "agenticir_level": level - 1,
            "airlight": float(max(rng.normal(HAZE_A_MEANS[level], 0.015), 0.4)),
            "beta": float(rng.normal(HAZE_BETA_MEANS[level], 0.30)),
            "depth_path": depth.as_posix(),
            "depth_sha256": observed_depth_sha,
        }
    if name == "rain":
        mean, std = RAIN_DISTRIBUTIONS[level]
        return {
            "operator": "AgenticIR.rain_project_strength",
            "agenticir_level": level - 1,
            "value": float(max(rng.normal(mean, std), 3.0)),
            "length": int(rng.integers(20, 40)),
            "angle_degrees": int(rng.integers(-30, 30)),
            "width": 3,
        }
    raise ValueError(name)


def _record(
    *,
    split: str,
    composition_size: int,
    combination_index: int,
    within_combination_index: int,
    combo: tuple[str, ...],
    source: Mapping[str, object],
    crop_size: int,
    seed: int,
) -> dict[str, object]:
    record_seed = stable_seed(seed, split, composition_size, combination_index, within_combination_index)
    rng = random.Random(record_seed)
    max_x = int(source["width"]) - crop_size
    max_y = int(source["height"]) - crop_size
    crop_x = rng.randint(0, max_x)
    crop_y = rng.randint(0, max_y)
    levels = _balanced_levels(combo, within_combination_index)
    application_order = list(combo)
    rng.shuffle(application_order)
    operators: dict[str, object] = {}
    for operator_index, name in enumerate(application_order):
        operator_seed = stable_seed(record_seed, operator_index, name)
        parameters = _operator_parameters(
            name, levels[name], seed=operator_seed, source=source
        )
        parameters["seed"] = operator_seed
        operators[name] = parameters
    label = make_label(combo, levels)
    record_id = f"{split[:3]}_c{composition_size}_{combination_index:02d}_{within_combination_index:04d}"
    return {
        "record_id": record_id,
        "split": split,
        "source_id": source["source_id"],
        "source_dataset": source["dataset"],
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "crop": {"x": crop_x, "y": crop_y, "width": crop_size, "height": crop_size},
        "composition_size": composition_size,
        "degradations": list(label.degradations),
        "levels": label.level_map,
        "application_order": application_order,
        "operators": operators,
        "seed": record_seed,
        "target": canonical_target(label),
        "output_path": f"data/qwen_multidegradation_pilot_v1/{split}/{record_id}.png",
        "agenticir_revision": AGENTICIR_REVISION,
    }


def build_pilot_records(
    inventory: Sequence[Mapping[str, object]],
    *,
    seed: int = CONTRACT_SEED,
    crop_size: int = CROP_SIZE,
) -> list[dict[str, object]]:
    assignments = assign_source_splits(inventory, seed=seed)
    records: list[dict[str, object]] = []
    for split in SPLITS:
        all_sources = assignments[split]
        haze_sources = tuple(source for source in all_sources if source["dataset"] == "mio100")
        for size in range(4):
            combinations = [()] if size == 0 else list(itertools.combinations(DEGRADATIONS, size))
            quota = SPLIT_COMPOSITION_COUNTS[split][size]
            if quota % len(combinations):
                raise RuntimeError("composition quota is not combination-balanced")
            per_combination = quota // len(combinations)
            if per_combination % 4 and size:
                raise RuntimeError("degraded combination quota is not severity-balanced")
            for combination_index, combo in enumerate(combinations):
                source_pool = haze_sources if "haze" in combo else all_sources
                ordered_sources = sorted(
                    source_pool,
                    key=lambda source: stable_digest(seed, split, combo, source["source_id"]),
                )
                for index in range(per_combination):
                    source = ordered_sources[index % len(ordered_sources)]
                    records.append(
                        _record(
                            split=split,
                            composition_size=size,
                            combination_index=combination_index,
                            within_combination_index=index,
                            combo=combo,
                            source=source,
                            crop_size=crop_size,
                            seed=seed,
                        )
                    )
    validate_pilot_records(records)
    return records


def validate_pilot_records(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(records) != 12000:
        raise ValueError(f"pilot must contain 12,000 records, got {len(records)}")
    ids = [str(record["record_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record IDs")
    source_roles: dict[str, str] = {}
    counts: Counter[tuple[str, int]] = Counter()
    combinations: Counter[tuple[str, tuple[str, ...]]] = Counter()
    severity: Counter[tuple[str, tuple[str, ...], str, int]] = Counter()
    for record in records:
        split = str(record["split"])
        size = int(record["composition_size"])
        names = tuple(record["degradations"])
        label = make_label(names, record["levels"])
        if names != label.degradations or size != len(names):
            raise ValueError(f"noncanonical record label: {record['record_id']}")
        if record["target"] != canonical_target(label):
            raise ValueError(f"target mismatch: {record['record_id']}")
        if set(record["application_order"]) != set(names):
            raise ValueError(f"application order mismatch: {record['record_id']}")
        source_id = str(record["source_id"])
        previous = source_roles.setdefault(source_id, split)
        if previous != split:
            raise ValueError(f"source leakage: {source_id}")
        if "haze" in names and record["source_dataset"] != "mio100":
            raise ValueError("haze record lacks objective depth source")
        counts[(split, size)] += 1
        combinations[(split, names)] += 1
        for name, level in label.levels:
            severity[(split, names, name, level)] += 1
    for split, expected in SPLIT_COMPOSITION_COUNTS.items():
        for size, count in expected.items():
            if counts[(split, size)] != count:
                raise ValueError(f"bad {split} composition {size} count")
            expected_combinations = (
                {()}
                if size == 0
                else set(itertools.combinations(DEGRADATIONS, size))
            )
            observed_combinations = {
                names
                for observed_split, names in combinations
                if observed_split == split and len(names) == size
            }
            if observed_combinations != expected_combinations:
                raise ValueError(
                    f"bad {split} composition {size} combination coverage"
                )
            expected_per_combination = count // len(expected_combinations)
            for names in expected_combinations:
                if combinations[(split, names)] != expected_per_combination:
                    raise ValueError(
                        f"unbalanced combination: {split} {names} "
                        f"{combinations[(split, names)]}"
                    )
    for (split, names), count in combinations.items():
        if not names:
            continue
        per_level = count // 4
        for name in names:
            values = [severity[(split, names, name, level)] for level in range(2, 6)]
            if values != [per_level] * 4:
                raise ValueError(f"unbalanced severity: {split} {names} {name} {values}")
    return {
        "records": len(records),
        "unique_record_ids": len(set(ids)),
        "unique_sources_used": len(source_roles),
        "source_disjoint": True,
        "split_composition_counts": {
            split: {str(size): counts[(split, size)] for size in range(4)}
            for split in SPLITS
        },
        "haze_depth_policy": "verified_mio100_depth_only",
        "combination_and_severity_balanced": True,
    }


def encode_jsonl(records: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


def write_new_or_verify(path: Path, payload: bytes) -> str:
    """Create an immutable artifact, or verify that an identical one exists."""

    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace differing artifact: {path}")
        return hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def _luma(image: np.ndarray) -> float:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return float(np.tensordot(rgb, LUMA_WEIGHTS, axes=([-1], [0])).mean())


def _apply_lowlight(image: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    vmin, vmax = int(v.min()), int(v.max())
    target = _luma(image) * float(parameters["target_luma_ratio"])
    if vmax <= vmin:
        # AgenticIR's official (V-vmin)/(vmax-vmin) mapping is undefined in
        # this limit case.  Its project-level luma calibration has a unique
        # structure-preserving solution: directly scale the uniform-V crop.
        raw = image.copy()
    else:
        dst_max = min(int(parameters["dst_max"]), vmax - 5)
        dst_max = max(dst_max, 80)
        mapped = np.rint(
            (v.astype(np.float32) - vmin) / (vmax - vmin) * dst_max
        ).astype(np.uint8)
        raw = cv2.cvtColor(cv2.merge((h, s, mapped)), cv2.COLOR_HSV2BGR)
    raw_luma = _luma(raw)
    if raw_luma <= 0:
        # A literally black input is already at the physical lower bound.
        return raw
    return np.rint(raw.astype(np.float32) * (target / raw_luma)).clip(0, 255).astype(np.uint8)


def _apply_noise(image: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
    rng = np.random.default_rng(int(parameters["seed"]))
    noise = rng.normal(0.0, float(parameters["sigma_0_255"]), image.shape)
    return np.rint(image.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


def _apply_blur(image: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
    radius = int(parameters["radius"])
    sigma = float(parameters["sigma"])
    angle = float(parameters["angle_degrees"])
    width = radius * 2 + 1
    weights = np.exp(-(np.arange(width) ** 2) / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)
    weights /= weights.sum()
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    norm = math.hypot(*point)
    blurred = np.zeros_like(image, dtype=np.float32)
    for index in range(width):
        dy = -math.ceil(((index * point[0]) / norm) - 0.5)
        dx = -math.ceil(((index * point[1]) / norm) - 0.5)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        blurred += weights[index] * shifted
    return np.rint(blurred).clip(0, 255).astype(np.uint8)


def _apply_haze(
    image: np.ndarray,
    parameters: Mapping[str, object],
    *,
    source_shape: tuple[int, int],
    crop: Mapping[str, int],
) -> np.ndarray:
    depth = loadmat(ROOT / str(parameters["depth_path"]))["data_obj"]
    depth = cv2.resize(depth, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_CUBIC)
    maximum = float(depth.max())
    if maximum <= 0:
        raise RuntimeError("haze depth has no positive values")
    depth = depth / maximum
    x, y = int(crop["x"]), int(crop["y"])
    h, w = int(crop["height"]), int(crop["width"])
    depth_crop = depth[y : y + h, x : x + w]
    transmission = np.exp(-float(parameters["beta"]) * depth_crop)[..., None]
    result = image * transmission + float(parameters["airlight"]) * 255 * (1 - transmission)
    return np.rint(result).clip(0, 255).astype(np.uint8)


def _apply_rain(image: np.ndarray, parameters: Mapping[str, object]) -> np.ndarray:
    rng = np.random.default_rng(int(parameters["seed"]))
    noise = rng.uniform(0, 256, image.shape[:2])
    noise[noise < (256 - float(parameters["value"]) * 0.01)] = 0
    initial = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, initial)
    length = int(parameters["length"])
    transform = cv2.getRotationMatrix2D(
        (length / 2, length / 2), float(parameters["angle_degrees"]) - 45, 1 - length / 100.0
    )
    kernel = cv2.warpAffine(np.diag(np.ones(length)), transform, (length, length))
    kernel = cv2.GaussianBlur(kernel, (int(parameters["width"]), int(parameters["width"])), 0)
    rain = cv2.filter2D(noise, -1, kernel)
    cv2.normalize(rain, rain, 0, 255, cv2.NORM_MINMAX)
    rain_rgb = np.repeat(rain.astype(np.uint8)[..., None], 3, axis=2)
    return np.rint(image.astype(np.float32) + rain_rgb).clip(0, 255).astype(np.uint8)


def generate_record_stages(
    record: Mapping[str, object],
    *,
    application_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, ...]:
    """Render every prefix of an ordered subset of a frozen record.

    The subset path is used for causal visibility audits. It deliberately
    reuses the frozen crop, parameters, and random seeds; the only intervention
    is omitting operators from the original application sequence.
    """
    source_path = ROOT / str(record["source_path"])
    if sha256(source_path) != record["source_sha256"]:
        raise RuntimeError(f"source hash changed: {source_path}")
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"cannot read source: {source_path}")
    crop = record["crop"]
    assert isinstance(crop, Mapping)
    x, y = int(crop["x"]), int(crop["y"])
    width, height = int(crop["width"]), int(crop["height"])
    image = source[y : y + height, x : x + width].copy()
    if image.shape[:2] != (height, width):
        raise RuntimeError(f"crop shape mismatch: {record['record_id']}")
    operators = record["operators"]
    assert isinstance(operators, Mapping)
    frozen_order = tuple(str(name) for name in record["application_order"])
    selected_order = (
        frozen_order
        if application_order is None
        else tuple(str(name) for name in application_order)
    )
    if len(selected_order) != len(set(selected_order)):
        raise ValueError("causal application order contains duplicate operators")
    if tuple(name for name in frozen_order if name in selected_order) != selected_order:
        raise ValueError("causal application order must be an ordered subset")
    stages = [image.copy()]
    for name in selected_order:
        parameters = operators[name]
        assert isinstance(parameters, Mapping)
        if name == "low_light":
            image = _apply_lowlight(image, parameters)
        elif name == "noise":
            image = _apply_noise(image, parameters)
        elif name == "blur":
            image = _apply_blur(image, parameters)
        elif name == "haze":
            image = _apply_haze(image, parameters, source_shape=source.shape[:2], crop=crop)
        elif name == "rain":
            image = _apply_rain(image, parameters)
        else:
            raise ValueError(name)
        stages.append(image.copy())
    return tuple(stages)


def generate_record(
    record: Mapping[str, object],
    *,
    application_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, str]:
    """Render a record or an ordered causal subset and return its PNG hash."""

    image = generate_record_stages(
        record,
        application_order=application_order,
    )[-1]
    return image, hashlib.sha256(cv2.imencode(".png", image)[1].tobytes()).hexdigest()


def write_generated_png(record: Mapping[str, object]) -> dict[str, object]:
    image, encoded_sha = generate_record(record)
    output_path = ROOT / str(record["output_path"])
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"failed to encode {record['record_id']}")
    payload = encoded.tobytes()
    if hashlib.sha256(payload).hexdigest() != encoded_sha:
        raise RuntimeError("PNG digest changed between deterministic encodes")
    write_new_or_verify(output_path, payload)
    return {
        "record_id": record["record_id"],
        "output_path": record["output_path"],
        "output_sha256": encoded_sha,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]
