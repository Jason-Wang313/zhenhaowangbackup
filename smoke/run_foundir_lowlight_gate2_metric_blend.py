#!/usr/bin/env python3
"""Evaluate the complete Gate 2 proposal lattice under the frozen metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_gate2_metric_blend_v1b.json"
BLENDS = ("B_CONVEX_ORACLE", "B_FREQUENCY_ORACLE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str) -> Path:
    path = (ROOT / raw).resolve(); path.relative_to(ROOT); return path


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content: raise RuntimeError(f"immutable Gate 2 artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle: handle.write(content)


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO(); Image.fromarray(image, mode="RGB").save(buffer, format="PNG"); return buffer.getvalue()


def save_png(path: Path, image: np.ndarray) -> str:
    content = png_bytes(image); write_once(path, content)
    with Image.open(io.BytesIO(content)) as opened: reloaded = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    if not np.array_equal(reloaded, image): raise RuntimeError("PNG round trip changed pixels")
    return hashlib.sha256(content).hexdigest()


def load_png(path: Path) -> np.ndarray:
    with Image.open(path) as opened: return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def load_target(row: Mapping[str, str]) -> np.ndarray:
    path = workspace_path(row["target_path"])
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3: raise RuntimeError(f"invalid target array: {path}")
        if row["stored_color_order"] == "BGR": image = image[..., ::-1]
        return np.ascontiguousarray(image)
    return load_png(path)


def quantize(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rint(np.clip(image, 0, 255)).astype(np.uint8))


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target.astype(np.float64) - prediction.astype(np.float64)) ** 2))


def convex_blend(proposals: Mapping[str, np.ndarray], target: np.ndarray, alphas: Sequence[float]) -> tuple[np.ndarray, dict]:
    best_image = next(iter(proposals.values())); best = mse(target, best_image); best_parameters = {"first": next(iter(proposals)), "second": next(iter(proposals)), "alpha": 0.0}
    for first, second in itertools.combinations(sorted(proposals), 2):
        a = proposals[first].astype(np.float32); b = proposals[second].astype(np.float32)
        for alpha in alphas:
            candidate = quantize((1.0 - float(alpha)) * a + float(alpha) * b)
            value = mse(target, candidate)
            if value < best:
                best, best_image = value, candidate; best_parameters = {"first": first, "second": second, "alpha": float(alpha)}
    return best_image, best_parameters


def frequency_blend(
    proposals: Mapping[str, np.ndarray], target: np.ndarray, directed_pairs: Sequence[Sequence[str]],
    sigmas: Sequence[float], alphas: Sequence[float]
) -> tuple[np.ndarray, dict]:
    first_name = min(proposals, key=lambda name: mse(target, proposals[name]))
    best_image = proposals[first_name]; best = mse(target, best_image)
    best_parameters = {"low": first_name, "high": first_name, "sigma": 0.0, "alpha": 0.0}
    float_proposals = {name: image.astype(np.float32) for name, image in proposals.items()}
    blurred = {(name, float(sigma)): cv2.GaussianBlur(image, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma)) for name, image in float_proposals.items() for sigma in sigmas}
    for low_name, high_name in directed_pairs:
        for sigma in sigmas:
            low = blurred[(low_name, float(sigma))]
            detail = float_proposals[high_name] - blurred[(high_name, float(sigma))]
            for alpha in alphas:
                candidate = quantize(low + float(alpha) * detail)
                value = mse(target, candidate)
                if value < best:
                    best, best_image = value, candidate
                    best_parameters = {"low": low_name, "high": high_name, "sigma": float(sigma), "alpha": float(alpha)}
    return best_image, best_parameters


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("protected_foundir_pixels_allowed") is not False:
        raise RuntimeError("metric/blend contract changed or permits protected pixels")
    for key in ("development_manifest", "metric_contract"):
        if sha256(workspace_path(config[key])) != config[f"{key}_sha256"]: raise RuntimeError(f"frozen dependency changed: {key}")
    photometric = config["input_manifests"]["photometric"]
    if sha256(workspace_path(photometric["path"])) != photometric["sha256"]: raise RuntimeError("photometric manifest changed")
    return config


def load_development(config: Mapping[str, object]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with workspace_path(config["development_manifest"]).open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["role"] == "outer_development"]
    if len(rows) != 245: raise RuntimeError("outer development count changed")
    return rows, {row["record_id"]: row for row in rows}


def load_candidate_manifests(config: Mapping[str, object]) -> dict[str, dict[str, dict[str, str]]]:
    sources = config["input_manifests"]
    teacher_result = json.loads(workspace_path(sources["generalist"]["identity_from"]).read_text())
    historical_result = json.loads(workspace_path(sources["historical_flux"]["identity_from"]).read_text())
    expected = {
        "historical_flux": historical_result["manifest_sha256"],
        "generalist": teacher_result["stages"]["generalist"]["manifest_sha256"],
        "cascade": teacher_result["stages"]["cascade"]["manifest_sha256"],
    }
    candidates: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for source_name, entry in sources.items():
        path = workspace_path(entry["path"])
        if source_name != "photometric" and sha256(path) != expected[source_name]: raise RuntimeError(f"candidate manifest identity changed: {source_name}")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                candidate = row["candidate_id"]
                if row["record_id"] in candidates[candidate]: raise RuntimeError(f"duplicate candidate record: {candidate}")
                candidates[candidate][row["record_id"]] = row
    return dict(candidates)


def literal_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    value = mse(target, prediction) / (255.0**2)
    return 100.0 if value == 0.0 else 10.0 * math.log10(1.0 / value)


def create_blends(config: dict, development: list[dict[str, str]], candidates: dict[str, dict[str, dict[str, str]]]) -> dict[str, dict[str, dict[str, str]]]:
    deployable = config["deployable_proposals"]
    output_root = workspace_path(config["outputs"]["blend_directory"])
    target_root = workspace_path(config["outputs"]["canonical_target_directory"])
    blend_rows: dict[str, dict[str, dict[str, str]]] = {name: {} for name in BLENDS}
    for index, row in enumerate(development):
        record_id = row["record_id"]; target = load_target(row)
        target_stem = f"{index:04d}_{row['family']}_{hashlib.sha256(record_id.encode()).hexdigest()[:16]}.png"
        target_path = target_root / target_stem; target_sha = save_png(target_path, target)
        proposals = {}
        for candidate in deployable:
            entry = candidates[candidate][record_id]
            proposal_path = workspace_path(entry["output_path"])
            if sha256(proposal_path) != entry["output_sha256"]:
                raise RuntimeError(f"proposal output identity changed: {candidate}:{record_id}")
            image = load_png(proposal_path)
            if image.shape != target.shape: raise RuntimeError(f"proposal shape mismatch: {candidate}:{record_id}")
            proposals[candidate] = image
        convex, convex_parameters = convex_blend(proposals, target, config["convex_blend"]["alpha_candidates"])
        frequency, frequency_parameters = frequency_blend(proposals, target, config["frequency_blend"]["directed_pairs"], config["frequency_blend"]["sigma_candidates"], config["frequency_blend"]["alpha_candidates"])
        for candidate, image, parameters in (
            (BLENDS[0], convex, convex_parameters), (BLENDS[1], frequency, frequency_parameters)
        ):
            output_path = output_root / candidate / target_stem
            output_sha = save_png(output_path, image)
            blend_rows[candidate][record_id] = {"candidate_id": candidate, "family": row["family"], "group_id": row["group_id"], "record_id": record_id, "target_path": str(target_path.relative_to(ROOT)), "target_sha256": target_sha, "output_path": str(output_path.relative_to(ROOT)), "output_sha256": output_sha, "parameters_json": json.dumps(parameters, separators=(",", ":"))}
        row["canonical_target_path"] = str(target_path.relative_to(ROOT)); row["canonical_target_sha256"] = target_sha
        print(f"GATE2_BLEND_PROGRESS records={index + 1}/{len(development)}", flush=True)
    candidates.update(blend_rows)
    return candidates


def metric_models(config: Mapping[str, object]):
    import lpips
    import pyiqa
    import torch

    metric_contract = json.loads(workspace_path(config["metric_contract"]).read_text())
    for package, expected in metric_contract["environment"]["packages"].items():
        if package in {"torch", "torchvision"}:
            actual = importlib.metadata.version(package)
            if not actual.startswith(expected.split("+")[0]): raise RuntimeError(f"metric package changed: {package} {actual}")
        elif importlib.metadata.version(package) != expected: raise RuntimeError(f"metric package changed: {package}")
    device = torch.device(config["runtime"]["device"])
    pyiqa_psnr = pyiqa.create_metric("psnr", device=device)
    pyiqa_lpips = pyiqa.create_metric("lpips", device=device)
    direct_lpips = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    return device, pyiqa_psnr, pyiqa_lpips, direct_lpips


def tensor(image: np.ndarray, device):
    import torch
    return torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0


def evaluate_metrics(config: dict, development: list[dict[str, str]], candidates: dict[str, dict[str, dict[str, str]]]) -> list[dict]:
    import torch
    device, pyiqa_psnr, pyiqa_lpips, direct_lpips = metric_models(config)
    candidate_ids = config["deployable_proposals"] + config["oracle_proposals"] + list(BLENDS)
    rows = []
    with torch.inference_mode():
        for record_index, development_row in enumerate(development):
            record_id = development_row["record_id"]
            target = load_png(workspace_path(development_row["canonical_target_path"]))
            target_tensor = tensor(target, device)
            for candidate_id in candidate_ids:
                entry = candidates[candidate_id][record_id]
                prediction_path = workspace_path(entry["output_path"])
                if sha256(prediction_path) != entry["output_sha256"]:
                    raise RuntimeError(f"metric input identity changed: {candidate_id}:{record_id}")
                prediction = load_png(prediction_path)
                prediction_tensor = tensor(prediction, device)
                literal = literal_psnr(target, prediction)
                evaluator = float(pyiqa_psnr(prediction_tensor, target_tensor).item())
                primary_lpips = float(pyiqa_lpips(prediction_tensor, target_tensor).item())
                direct = float(direct_lpips(prediction_tensor * 2 - 1, target_tensor * 2 - 1).item())
                rows.append({
                    "candidate_id": candidate_id, "family": development_row["family"], "group_id": development_row["group_id"], "record_id": record_id,
                    "target_path": development_row["canonical_target_path"], "output_path": entry["output_path"],
                    "psnr_literal": literal, "psnr_pyiqa": evaluator, "psnr_conservative": min(literal, evaluator),
                    "lpips_pyiqa": primary_lpips, "lpips_direct": direct, "lpips_conservative": max(primary_lpips, direct),
                    "ssim": float(structural_similarity(target, prediction, channel_axis=2, data_range=255)),
                    "target_access": candidate_id in config["oracle_proposals"] or candidate_id in BLENDS,
                })
            print(f"GATE2_METRIC_PROGRESS records={record_index + 1}/{len(development)}", flush=True)
    return rows


def grouped_values(rows: Sequence[Mapping[str, object]], field: str) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows: groups[str(row["group_id"])].append(float(row[field]))
    return {group: values for group, values in groups.items()}


def summarize(config: Mapping[str, object], rows: list[dict]) -> dict:
    aggregate = {}; rng = np.random.default_rng(config["aggregation"]["bootstrap_seed"])
    fields = ("psnr_conservative", "lpips_conservative", "ssim")
    for candidate in sorted({row["candidate_id"] for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
        families = {}
        for family in sorted({row["family"] for row in candidate_rows}):
            selected = [row for row in candidate_rows if row["family"] == family]
            family_result = {"records": len(selected)}
            for field in fields:
                groups = grouped_values(selected, field); points = np.array([np.mean(values) for values in groups.values()])
                estimates = np.mean(rng.choice(points, size=(config["aggregation"]["bootstrap_resamples"], len(points)), replace=True), axis=1)
                family_result[field] = {"estimate": float(np.mean(points)), "lower": float(np.quantile(estimates, 0.025)), "upper": float(np.quantile(estimates, 0.975))}
                family_result["groups"] = len(groups)
            families[family] = family_result
        aggregate[candidate] = {
            "families": families,
            "equal_family_psnr": float(np.mean([value["psnr_conservative"]["estimate"] for value in families.values()])),
            "worst_family_psnr": float(min(value["psnr_conservative"]["estimate"] for value in families.values())),
            "equal_family_lpips": float(np.mean([value["lpips_conservative"]["estimate"] for value in families.values()])),
            "worst_family_lpips": float(max(value["lpips_conservative"]["estimate"] for value in families.values())),
        }
    return aggregate


def promotes(candidate: Mapping[str, object], baseline: Mapping[str, object], config: Mapping[str, object]) -> bool:
    policy = config["promotion"]
    for family in baseline["families"]:
        c, b = candidate["families"][family], baseline["families"][family]
        psnr_gain = c["psnr_conservative"]["estimate"] - b["psnr_conservative"]["estimate"]
        lpips_gain = b["lpips_conservative"]["estimate"] - c["lpips_conservative"]["estimate"]
        ordinary = psnr_gain >= policy["psnr_gain_each_family"] and lpips_gain >= -policy["lpips_max_regression_each_family"]
        perceptual = lpips_gain >= policy["perceptual_lpips_gain_each_family"] and psnr_gain >= -policy["perceptual_psnr_max_regression_each_family"]
        if not (ordinary or perceptual): return False
    return True


def decide(config: Mapping[str, object], aggregate: Mapping[str, object]) -> dict:
    raw = aggregate["P0_RAW"]; fixed = aggregate["P0_FIXED_TRAIN_MEAN_EXPOSURE"]
    predictor_pass = promotes(aggregate["D_INPUT_ONLY_BOUNDED_PHOTOMETRIC"], fixed, config)
    eligible = []
    for candidate in config["deployable_proposals"]:
        if candidate == "D_INPUT_ONLY_BOUNDED_PHOTOMETRIC" and not predictor_pass: continue
        noninferior = all(
            aggregate[candidate]["families"][family]["psnr_conservative"]["estimate"] >= raw["families"][family]["psnr_conservative"]["estimate"] - config["promotion"]["perceptual_psnr_max_regression_each_family"]
            and aggregate[candidate]["families"][family]["lpips_conservative"]["estimate"] <= raw["families"][family]["lpips_conservative"]["estimate"] + config["promotion"]["lpips_max_regression_each_family"]
            for family in raw["families"]
        )
        if noninferior: eligible.append(candidate)
    provisional = max(eligible, key=lambda candidate: (aggregate[candidate]["worst_family_psnr"], -aggregate[candidate]["worst_family_lpips"])) if eligible else "P0_RAW"
    teacher_value = {candidate: promotes(aggregate[candidate], fixed, config) for candidate in ("P1_FOUNDIR_GENERALIST", "P1_FOUNDIR_CASCADE")}
    fusion_gaps = {
        blend: {family: aggregate[blend]["families"][family]["psnr_conservative"]["estimate"] - aggregate[provisional]["families"][family]["psnr_conservative"]["estimate"] for family in raw["families"]}
        for blend in BLENDS
    }
    return {
        "provisional_conservative_base": provisional,
        "input_only_photometric_head": "admitted" if predictor_pass else "killed_no_cross_family_transfer",
        "teacher_proposal_value_gate": teacher_value,
        "teacher_at_inference_deployment": "still_conditional_on_license_provenance_runtime_and_future_flux_increment",
        "fusion_ceiling_psnr_gap_over_provisional_by_family": fusion_gaps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG); args = parser.parse_args()
    config_path = args.config.resolve(); config = read_config(config_path)
    os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/foundir-metrics"))
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != config["runtime"]["device_count"]: raise SystemExit(f"METRIC_BLEND_REQUIRES_EXACTLY_ONE_VISIBLE_GPU actual={torch.cuda.device_count()}")
    development, _ = load_development(config); candidates = load_candidate_manifests(config)
    candidates = create_blends(config, development, candidates)
    rows = evaluate_metrics(config, development, candidates); rows.sort(key=lambda row: (row["candidate_id"], row["family"], row["group_id"], row["record_id"]))
    fields = list(rows[0]); buffer = io.StringIO(newline=""); writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    rows_content = buffer.getvalue().encode(); rows_path = workspace_path(config["outputs"]["rows"]); write_once(rows_path, rows_content)
    aggregate = summarize(config, rows); decision = decide(config, aggregate)
    payload = {"schema_version": 1, "status": "pass_gate2_complete", "protected_foundir_pixels_used": False, "config": str(config_path.relative_to(ROOT)), "config_sha256": sha256(config_path), "runner_sha256": sha256(Path(__file__)), "rows": str(rows_path.relative_to(ROOT)), "rows_sha256": hashlib.sha256(rows_content).hexdigest(), "records": len(development), "candidate_count": len(aggregate), "metric_rows": len(rows), "aggregate": aggregate, "decision": decision}
    result_path = workspace_path(config["outputs"]["result"]); write_once(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(f"FOUNDIR_LOWLIGHT_GATE2_PASS base={decision['provisional_conservative_base']} photometric={decision['input_only_photometric_head']} candidates={len(aggregate)}")


if __name__ == "__main__": main()
