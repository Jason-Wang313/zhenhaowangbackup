#!/usr/bin/env python3
"""Run the target-access-separated photometric portion of Gate 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from lowlight_restoration.paired_development import fit_oracle_gamma_color


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_gate2_photometric_v1.json"
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
CANDIDATES = (
    "P0_RAW",
    "P0_FIXED_TRAIN_MEAN_EXPOSURE",
    "D_INPUT_ONLY_BOUNDED_PHOTOMETRIC",
    "O1_EXPOSURE",
    "O2_MONOTONE_TONE_WB",
    "O3_TONE_WB_SMOOTH_ILLUMINATION",
    "O4_GAMMA_FULL_MATRIX",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str) -> Path:
    path = (ROOT / raw).resolve()
    path.relative_to(ROOT)
    return path


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("protected_foundir_pixels_allowed") is not False:
        raise RuntimeError("Gate 2 photometric contract changed or permits protected pixels")
    manifest = workspace_path(config["development_manifest"])
    if sha256(manifest) != config["development_manifest_sha256"]:
        raise RuntimeError("Gate 1 development manifest changed")
    proposal = workspace_path(config["proposal_contract"])
    if sha256(proposal) != config["proposal_contract_sha256"]:
        raise RuntimeError("Gate 1 proposal contract changed")
    return config


def load_manifest(config: Mapping[str, object]) -> list[dict[str, str]]:
    path = workspace_path(str(config["development_manifest"]))
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 674 or len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("Gate 1 development rows changed")
    return rows


def load_rgb(row: Mapping[str, str], key: str) -> np.ndarray:
    path = workspace_path(row[key])
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise RuntimeError(f"invalid SDSD array: {path}")
        if row["stored_color_order"] == "BGR":
            image = image[..., ::-1]
        elif row["stored_color_order"] != "RGB":
            raise RuntimeError(f"unsupported stored color order: {row['stored_color_order']}")
        return np.ascontiguousarray(image)
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def luma(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) @ LUMA_WEIGHTS


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    difference = reference.astype(np.float64) - prediction.astype(np.float64)
    mse = float(np.mean((difference / 255.0) ** 2))
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)


def ssim(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(structural_similarity(reference, prediction, channel_axis=2, data_range=255))


def quantize(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    maximum = float(np.max(image)) if image.size else 0.0
    scaled = image * 255.0 if maximum <= 1.00001 else image
    return np.ascontiguousarray(np.rint(np.clip(scaled, 0.0, 255.0)).astype(np.uint8))


def fit_exposure(source: np.ndarray, target: np.ndarray, minimum: float, maximum: float) -> float:
    x = np.asarray(source, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    denominator = float(x @ x)
    value = 1.0 if denominator <= 1e-12 else float((x @ y) / denominator)
    return float(np.clip(value, minimum, maximum))


def apply_exposure(source: np.ndarray, exposure: float) -> np.ndarray:
    return quantize(source.astype(np.float32) * float(exposure))


def isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape != weights.shape or np.any(weights <= 0):
        raise ValueError("invalid isotonic inputs")
    blocks: list[list[float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(value), float(weight), index, index + 1])
        while len(blocks) >= 2 and float(blocks[-2][0]) > float(blocks[-1][0]):
            right = blocks.pop()
            left = blocks.pop()
            total = float(left[1]) + float(right[1])
            mean = (float(left[0]) * float(left[1]) + float(right[0]) * float(right[1])) / total
            blocks.append([mean, total, int(left[2]), int(right[3])])
    output = np.empty_like(values)
    for value, _, start, end in blocks:
        output[int(start) : int(end)] = float(value)
    return np.clip(output, 0.0, 1.0)


def fit_monotone_lut(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    source_u8 = quantize(source)
    target_u8 = quantize(target)
    output = np.empty_like(source_u8)
    luts = []
    for channel in range(3):
        x = source_u8[..., channel].reshape(-1)
        y = target_u8[..., channel].reshape(-1).astype(np.float64) / 255.0
        counts = np.bincount(x, minlength=256).astype(np.float64)
        sums = np.bincount(x, weights=y, minlength=256)
        observed = np.flatnonzero(counts)
        means = sums[observed] / counts[observed]
        projected = isotonic_increasing(means, counts[observed])
        lut = np.interp(np.arange(256), observed, projected, left=projected[0], right=projected[-1])
        lut_u8 = quantize(lut)
        output[..., channel] = lut_u8[x].reshape(source_u8.shape[:2])
        luts.append(lut_u8.tolist())
    return output, {"luts": luts}


def fit_smooth_illumination(
    base: np.ndarray, target: np.ndarray, *, grid: Sequence[int], sigma: float,
    log_bound: float, strengths: Sequence[float]
) -> tuple[np.ndarray, dict]:
    base_float = base.astype(np.float32) / 255.0
    target_float = target.astype(np.float32) / 255.0
    base_luma = luma(base_float)
    target_luma = luma(target_float)
    width, height = int(grid[1]), int(grid[0])
    small_base = cv2.resize(base_luma, (width, height), interpolation=cv2.INTER_AREA)
    small_target = cv2.resize(target_luma, (width, height), interpolation=cv2.INTER_AREA)
    log_gain = np.log((small_target + 1.0 / 255.0) / (small_base + 1.0 / 255.0))
    log_gain = np.clip(log_gain, -log_bound, log_bound)
    log_gain = cv2.GaussianBlur(log_gain, (0, 0), sigmaX=sigma, sigmaY=sigma)
    full = cv2.resize(log_gain, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_CUBIC)
    best = base
    best_strength = 0.0
    best_mse = float(np.mean((base.astype(np.float64) - target.astype(np.float64)) ** 2))
    for strength in strengths:
        gain = np.exp(float(strength) * full)[..., None]
        candidate = quantize(base_float * gain)
        mse = float(np.mean((candidate.astype(np.float64) - target.astype(np.float64)) ** 2))
        if mse < best_mse:
            best, best_strength, best_mse = candidate, float(strength), mse
    return best, {"strength": best_strength, "grid_log_gain": log_gain.tolist()}


def image_feature(source: np.ndarray, config: Mapping[str, object]) -> np.ndarray:
    source_float = source.astype(np.float32) / 255.0
    light = luma(source_float)
    q = np.asarray(config["luma_quantiles"], dtype=np.float64)
    rgb_q = np.asarray(config["rgb_feature_quantiles"], dtype=np.float64)
    features: list[float] = np.quantile(light, q).tolist()
    for channel in range(3):
        features.extend(np.quantile(source_float[..., channel], rgb_q).tolist())
    means = source_float.reshape(-1, 3).mean(axis=0)
    stds = source_float.reshape(-1, 3).std(axis=0)
    features.extend(means.tolist())
    features.extend(stds.tolist())
    features.extend([math.log((means[0] + 1e-4) / (means[1] + 1e-4)), math.log((means[2] + 1e-4) / (means[1] + 1e-4))])
    features.extend([float(np.mean(light <= value / 255.0)) for value in (4, 16, 32)])
    features.extend([float(np.mean(source_float <= 1.0 / 255.0)), float(np.mean(source_float >= 254.0 / 255.0))])
    return np.asarray(features, dtype=np.float64)


def target_summary(target: np.ndarray, config: Mapping[str, object]) -> np.ndarray:
    target_float = target.astype(np.float32) / 255.0
    values = np.quantile(luma(target_float), np.asarray(config["luma_quantiles"], dtype=np.float64)).tolist()
    means = target_float.reshape(-1, 3).mean(axis=0)
    values.extend([math.log((means[0] + 1e-4) / (means[1] + 1e-4)), math.log((means[2] + 1e-4) / (means[1] + 1e-4))])
    return np.asarray(values, dtype=np.float64)


def render_predicted_summary(source: np.ndarray, predicted: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    count = len(quantiles)
    predicted = np.asarray(predicted, dtype=np.float64)
    yq = isotonic_increasing(np.clip(predicted[:count], 0.0, 1.0), np.ones(count))
    source_float = source.astype(np.float32) / 255.0
    source_luma = luma(source_float)
    xq = np.quantile(source_luma, np.asarray(quantiles, dtype=np.float64))
    unique, inverse = np.unique(xq, return_inverse=True)
    reduced = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.float64)
    np.add.at(reduced, inverse, yq)
    np.add.at(counts, inverse, 1.0)
    reduced = isotonic_increasing(reduced / counts, counts)
    xp = np.concatenate(([0.0], unique, [1.0]))
    fp = np.concatenate(([0.0], reduced, [1.0]))
    xp, unique_indices = np.unique(xp, return_index=True)
    fp = fp[unique_indices]
    fp = np.maximum.accumulate(fp)
    tone = np.interp(source_luma, xp, fp)
    scale = tone / np.maximum(source_luma, 1.0 / 255.0)
    base = source_float * scale[..., None]
    rg = float(np.clip(predicted[count], -math.log(4.0), math.log(4.0)))
    bg = float(np.clip(predicted[count + 1], -math.log(4.0), math.log(4.0)))
    gains = np.array([math.exp(rg), 1.0, math.exp(bg)], dtype=np.float32)
    gains /= float(gains @ LUMA_WEIGHTS)
    return quantize(base * gains)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = y.mean(axis=0)
    normalized = (x - x_mean) / x_scale
    coefficient = np.linalg.solve(
        normalized.T @ normalized + float(alpha) * np.eye(normalized.shape[1]),
        normalized.T @ (y - y_mean),
    )
    return {"x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "coefficient": coefficient, "alpha": float(alpha)}


def predict_ridge(model: Mapping[str, object], feature: np.ndarray) -> np.ndarray:
    return (feature - model["x_mean"]) / model["x_scale"] @ model["coefficient"] + model["y_mean"]


def summarize_training_record(arguments: tuple[dict[str, str], dict]) -> dict:
    row, predictor = arguments
    cv2.setNumThreads(1)
    source, target = load_rgb(row, "input_path"), load_rgb(row, "target_path")
    if source.shape != target.shape:
        raise RuntimeError(row["record_id"])
    exposure = fit_exposure(source, target, 0.25, 16.0)
    return {
        "record_id": row["record_id"], "family": row["family"], "group_id": row["group_id"],
        "feature": image_feature(source, predictor), "target": target_summary(target, predictor), "exposure": exposure,
    }


def fold_index(group_id: str, prefix: str, folds: int) -> int:
    digest = hashlib.sha256(f"{prefix}{group_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def evaluate_cv_task(arguments: tuple[float, int, dict, list[dict[str, str]], dict]) -> tuple[float, int, list[dict]]:
    alpha, fold, model, rows, predictor = arguments
    cv2.setNumThreads(1)
    records = []
    for row in rows:
        source, target = load_rgb(row, "input_path"), load_rgb(row, "target_path")
        prediction = render_predicted_summary(source, predict_ridge(model, image_feature(source, predictor)), predictor["luma_quantiles"])
        records.append({"family": row["family"], "group_id": row["group_id"], "psnr": psnr(target, prediction)})
    return alpha, fold, records


def group_family_score(records: Sequence[Mapping[str, object]]) -> tuple[float, dict[str, float]]:
    group_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        group_values[(str(row["family"]), str(row["group_id"]))].append(float(row["psnr"]))
    family_values: dict[str, list[float]] = defaultdict(list)
    for (family, _), values in group_values.items():
        family_values[family].append(float(np.mean(values)))
    means = {family: float(np.mean(values)) for family, values in sorted(family_values.items())}
    return min(means.values()), means


def encoded_png(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def save_once(path: Path, image: np.ndarray) -> tuple[np.ndarray, str]:
    content = encoded_png(image)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable Gate 2 output differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)
    with Image.open(io.BytesIO(content)) as opened:
        reloaded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(reloaded, image):
        raise RuntimeError("PNG round trip changed pixels")
    return reloaded, hashlib.sha256(content).hexdigest()


def evaluate_outer_record(arguments: tuple[int, dict[str, str], dict, float, dict, dict]) -> list[dict]:
    index, row, config, fixed_exposure, model, predictor = arguments
    cv2.setNumThreads(1)
    source, target = load_rgb(row, "input_path"), load_rgb(row, "target_path")
    if source.shape != target.shape:
        raise RuntimeError(row["record_id"])
    oracle = config["oracles"]
    o1_config = oracle["O1_EXPOSURE"]
    exposure = fit_exposure(source, target, o1_config["minimum"], o1_config["maximum"])
    o1 = apply_exposure(source, exposure)
    o2_candidate, o2_parameters = fit_monotone_lut(source, target)
    if psnr(target, o1) > psnr(target, o2_candidate):
        o2, o2_parameters = o1, {"retained": "O1_EXPOSURE"}
    else:
        o2 = o2_candidate
        o2_parameters["retained"] = "MONOTONE_LUT"
    o3_config = oracle["O3_TONE_WB_SMOOTH_ILLUMINATION"]
    o3, o3_parameters = fit_smooth_illumination(
        o2, target, grid=o3_config["grid"], sigma=o3_config["gaussian_sigma_grid_cells"],
        log_bound=o3_config["log_gain_bound"], strengths=o3_config["strength_candidates"],
    )
    o4_config = oracle["O4_GAMMA_FULL_MATRIX"]
    fit = fit_oracle_gamma_color(
        source.astype(np.float32) / 255.0, target.astype(np.float32) / 255.0,
        gamma_candidates=tuple(o4_config["gamma_candidates"]), maximum_fit_pixels=o4_config["maximum_fit_pixels"], ridge=o4_config["ridge"],
    )
    o4 = quantize(np.asarray(fit["prediction"], dtype=np.float32))
    predicted_summary = predict_ridge(model, image_feature(source, predictor))
    candidates = {
        "P0_RAW": (source, {}),
        "P0_FIXED_TRAIN_MEAN_EXPOSURE": (apply_exposure(source, fixed_exposure), {"exposure": fixed_exposure}),
        "D_INPUT_ONLY_BOUNDED_PHOTOMETRIC": (render_predicted_summary(source, predicted_summary, predictor["luma_quantiles"]), {"predicted_summary": np.asarray(predicted_summary).tolist()}),
        "O1_EXPOSURE": (o1, {"exposure": exposure}),
        "O2_MONOTONE_TONE_WB": (o2, o2_parameters),
        "O3_TONE_WB_SMOOTH_ILLUMINATION": (o3, o3_parameters),
        "O4_GAMMA_FULL_MATRIX": (o4, {"gamma": float(fit["gamma"]), "coefficients": np.asarray(fit["coefficients"]).tolist()}),
    }
    stem = f"{index:04d}_{row['family']}_{hashlib.sha256(row['record_id'].encode()).hexdigest()[:16]}.png"
    output_root = workspace_path(config["output"]["directory"])
    results = []
    for candidate_id in CANDIDATES:
        image, parameters = candidates[candidate_id]
        path = output_root / candidate_id / stem
        reloaded, digest = save_once(path, image)
        results.append({
            "candidate_id": candidate_id, "family": row["family"], "group_id": row["group_id"], "record_id": row["record_id"],
            "input_path": row["input_path"], "target_path": row["target_path"], "output_path": str(path.relative_to(ROOT)), "output_sha256": digest,
            "width": int(image.shape[1]), "height": int(image.shape[0]), "psnr": psnr(target, reloaded), "ssim": ssim(target, reloaded),
            "target_access": candidate_id.startswith("O"), "parameters_json": json.dumps(parameters, separators=(",", ":")),
        })
    return results


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict:
    result = {}
    for candidate in CANDIDATES:
        selected = [row for row in rows if row["candidate_id"] == candidate]
        families = {}
        for family in sorted({str(row["family"]) for row in selected}):
            family_rows = [row for row in selected if row["family"] == family]
            by_group: dict[str, list[Mapping[str, object]]] = defaultdict(list)
            for row in family_rows:
                by_group[str(row["group_id"])].append(row)
            group_psnr = [float(np.mean([float(x["psnr"]) for x in group])) for group in by_group.values()]
            group_ssim = [float(np.mean([float(x["ssim"]) for x in group])) for group in by_group.values()]
            families[family] = {"records": len(family_rows), "groups": len(by_group), "psnr": float(np.mean(group_psnr)), "ssim": float(np.mean(group_ssim))}
        result[candidate] = {
            "families": families,
            "equal_family_psnr": float(np.mean([value["psnr"] for value in families.values()])),
            "worst_family_psnr": float(min(value["psnr"] for value in families.values())),
            "equal_family_ssim": float(np.mean([value["ssim"] for value in families.values()])),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = read_config(config_path)
    rows = load_manifest(config)
    training_rows = [row for row in rows if row["role"] == config["roles"]["fit"]]
    outer_rows = [row for row in rows if row["role"] == config["roles"]["evaluate"]]
    predictor = config["input_predictor"]
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        summaries = list(executor.map(summarize_training_record, [(row, predictor) for row in training_rows], chunksize=2))
    summary_by_id = {row["record_id"]: row for row in summaries}
    x = np.stack([summary_by_id[row["record_id"]]["feature"] for row in training_rows])
    y = np.stack([summary_by_id[row["record_id"]]["target"] for row in training_rows])

    group_logs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for item in summaries:
        group_logs[(item["family"], item["group_id"])].append(math.log(item["exposure"]))
    family_logs: dict[str, list[float]] = defaultdict(list)
    for (family, _), values in group_logs.items():
        family_logs[family].append(float(np.mean(values)))
    fixed_exposure = float(np.exp(np.mean([np.mean(values) for values in family_logs.values()])))
    fixed_exposure = float(np.clip(fixed_exposure, config["fixed_control"]["minimum_exposure"], config["fixed_control"]["maximum_exposure"]))

    folds = int(predictor["inner_folds"])
    fold_values = np.array([fold_index(row["group_id"], predictor["fold_hash_prefix"], folds) for row in training_rows])
    tasks = []
    for alpha in predictor["ridge_alphas"]:
        for fold in range(folds):
            fit_mask = fold_values != fold
            model = fit_ridge(x[fit_mask], y[fit_mask], float(alpha))
            validation = [row for row, value in zip(training_rows, fold_values) if value == fold]
            tasks.append((float(alpha), fold, model, validation, predictor))
    cv_rows: dict[float, list[dict]] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for alpha, _, records in executor.map(evaluate_cv_task, tasks, chunksize=1):
            cv_rows[alpha].extend(records)
    cv_results = {}
    for alpha in predictor["ridge_alphas"]:
        score, families = group_family_score(cv_rows[float(alpha)])
        cv_results[str(alpha)] = {"worst_family_group_balanced_psnr": score, "family_psnr": families}
    selected_alpha = min(
        (float(alpha) for alpha in predictor["ridge_alphas"]),
        key=lambda alpha: (-cv_results[str(alpha)]["worst_family_group_balanced_psnr"], alpha),
    )
    model = fit_ridge(x, y, selected_alpha)

    tasks = [(index, row, config, fixed_exposure, model, predictor) for index, row in enumerate(outer_rows)]
    result_rows = []
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for index, batch in enumerate(executor.map(evaluate_outer_record, tasks, chunksize=1), start=1):
            result_rows.extend(batch)
            print(f"GATE2_PHOTOMETRIC_PROGRESS records={index}/{len(tasks)}", flush=True)
    result_rows.sort(key=lambda row: (row["candidate_id"], row["family"], row["group_id"], row["record_id"]))

    fields = list(result_rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(result_rows)
    csv_content = buffer.getvalue().encode()
    output_manifest = workspace_path(config["output"]["manifest"])
    if output_manifest.exists() and output_manifest.read_bytes() != csv_content:
        raise RuntimeError("existing Gate 2 photometric manifest differs")
    if not output_manifest.exists():
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        with output_manifest.open("xb") as handle: handle.write(csv_content)
    payload = {
        "schema_version": 1, "status": "pass_photometric_lattice_complete_lpips_and_gpu_proposals_pending", "protected_foundir_pixels_used": False,
        "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)),
        "manifest": str(output_manifest.relative_to(ROOT)), "manifest_sha256": hashlib.sha256(csv_content).hexdigest(),
        "training_records": len(training_rows), "outer_records": len(outer_rows), "candidate_outputs": len(result_rows),
        "fixed_exposure": fixed_exposure, "input_predictor": {"selected_alpha": selected_alpha, "cross_validation": cv_results},
        "aggregate": aggregate(result_rows), "oracle_outputs_are_not_deployable": True,
    }
    output_result = workspace_path(config["output"]["result"])
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if output_result.exists() and output_result.read_bytes() != content:
        raise RuntimeError("existing Gate 2 photometric result differs")
    if not output_result.exists():
        output_result.parent.mkdir(parents=True, exist_ok=True)
        with output_result.open("xb") as handle: handle.write(content)
    print(f"FOUNDIR_LOWLIGHT_GATE2_PHOTOMETRIC_PASS outer={len(outer_rows)} candidates={len(CANDIDATES)} outputs={len(result_rows)} selected_alpha={selected_alpha}")


if __name__ == "__main__":
    main()
