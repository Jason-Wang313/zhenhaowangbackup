#!/usr/bin/env python3
"""Evaluate Gate 3 compact outputs and their target-only cascade blend ceiling."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs/foundir_lowlight_gate3_v1.json"
COMPACT = "P3_COMPACT_RESIDUAL_V1"
BLEND = "O_GATE3_COMPACT_CASCADE_CONVEX"


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


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable Gate 3 metric artifact differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(content)


def load_rgb_path(path: Path, stored_color_order: str = "RGB") -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path, allow_pickle=False)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError(f"invalid target array: {path}")
        if stored_color_order == "BGR":
            image = image[..., ::-1]
        return np.ascontiguousarray(image)
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"), dtype=np.uint8)


def png_bytes(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def save_png(path: Path, image: np.ndarray) -> str:
    content = png_bytes(image)
    write_once(path, content)
    return hashlib.sha256(content).hexdigest()


def read_config(path: Path) -> tuple[dict, str]:
    config = json.loads(path.read_text())
    if config.get("status") != "frozen_before_gate3_training" or config.get("protected_foundir_pixels_allowed") is not False:
        raise RuntimeError("Gate 3 metric contract is not frozen or permits protected pixels")
    for key in ("development_manifest", "gate2_result", "cascade_manifest"):
        if sha256(workspace_path(config["parents"][key])) != config["parents"][f"{key}_sha256"]:
            raise RuntimeError(f"Gate 3 metric dependency changed: {key}")
    for key in ("model", "trainer", "evaluator", "test"):
        identity = config["implementation_identities"]
        if sha256(workspace_path(identity[key])) != identity[f"{key}_sha256"]:
            raise RuntimeError(f"frozen Gate 3 implementation changed: {key}")
    return config, sha256(path)


def keyed_manifest(path: Path, expected: int) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({row["record_id"] for row in rows}) != expected:
        raise RuntimeError(f"manifest coverage changed: {path}")
    return {row["record_id"]: row for row in rows}


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((target.astype(np.float64) - prediction.astype(np.float64)) ** 2))


def literal_psnr(target: np.ndarray, prediction: np.ndarray) -> float:
    value = mse(target, prediction) / (255.0**2)
    return 100.0 if value == 0 else 10.0 * math.log10(1.0 / value)


def best_blend(compact: np.ndarray, cascade: np.ndarray, target: np.ndarray, alphas: Sequence[float]) -> tuple[np.ndarray, float]:
    winner, best_alpha, best_mse = compact, 0.0, mse(target, compact)
    for alpha in alphas:
        candidate = np.rint(np.clip((1.0 - float(alpha)) * compact.astype(np.float32) + float(alpha) * cascade.astype(np.float32), 0, 255)).astype(np.uint8)
        candidate_mse = mse(target, candidate)
        if candidate_mse < best_mse:
            winner, best_alpha, best_mse = candidate, float(alpha), candidate_mse
    return winner, best_alpha


def metric_models(config: Mapping[str, object]):
    import lpips
    import pyiqa
    import torch

    metric_contract = json.loads(workspace_path("configs/foundir_lowlight_metric_contract_v2b.json").read_text())
    for package, expected in metric_contract["environment"]["packages"].items():
        actual = importlib.metadata.version(package)
        if package in {"torch", "torchvision"}:
            if not actual.startswith(expected.split("+")[0]):
                raise RuntimeError(f"metric package changed: {package} {actual}")
        elif actual != expected:
            raise RuntimeError(f"metric package changed: {package} {actual}")
    device = torch.device("cuda:0")
    return (
        device,
        pyiqa.create_metric("psnr", device=device),
        pyiqa.create_metric("lpips", device=device),
        lpips.LPIPS(net="alex", version="0.1").to(device).eval(),
    )


def tensor(image: np.ndarray, device):
    import torch

    return torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).to(device=device, dtype=torch.float32).div(255.0)


def aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result = {}
    for candidate in sorted({str(row["candidate_id"]) for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate]
        families = {}
        for family in sorted({str(row["family"]) for row in candidate_rows}):
            selected = [row for row in candidate_rows if row["family"] == family]
            grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
            for row in selected:
                grouped[str(row["group_id"])].append(row)
            family_result = {"records": len(selected), "groups": len(grouped)}
            for field in ("psnr_conservative", "lpips_conservative", "ssim"):
                values = [float(np.mean([float(item[field]) for item in group])) for group in grouped.values()]
                family_result[field] = {"estimate": float(np.mean(values))}
            families[family] = family_result
        result[candidate] = {
            "families": families,
            "equal_family_psnr": float(np.mean([item["psnr_conservative"]["estimate"] for item in families.values()])),
            "worst_family_psnr": float(min(item["psnr_conservative"]["estimate"] for item in families.values())),
            "equal_family_lpips": float(np.mean([item["lpips_conservative"]["estimate"] for item in families.values()])),
            "worst_family_lpips": float(max(item["lpips_conservative"]["estimate"] for item in families.values())),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config, config_sha = read_config(config_path)
    training_result_path = workspace_path(config["outputs"]["training_result"])
    training_result = json.loads(training_result_path.read_text())
    if training_result.get("status") != "compact_training_and_outer_inference_complete_awaiting_metrics" or training_result.get("config_sha256") != config_sha:
        raise RuntimeError("Gate 3 compact training result is absent or belongs to another contract")
    compact_manifest_path = workspace_path(config["outputs"]["prediction_manifest"])
    if sha256(compact_manifest_path) != training_result["manifest_sha256"]:
        raise RuntimeError("Gate 3 compact prediction manifest changed")
    count = int(config["data"]["outer_records"])
    compact_rows = keyed_manifest(compact_manifest_path, count)
    cascade_rows = keyed_manifest(workspace_path(config["parents"]["cascade_manifest"]), count)
    with workspace_path(config["parents"]["development_manifest"]).open(newline="") as handle:
        development = [row for row in csv.DictReader(handle) if row["role"] == config["data"]["outer_role"]]
    if len(development) != count:
        raise RuntimeError("Gate 3 outer development coverage changed")
    blend_manifest = []
    loaded: list[tuple[dict[str, str], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = []
    for index, row in enumerate(development):
        compact_entry, cascade_entry = compact_rows[row["record_id"]], cascade_rows[row["record_id"]]
        compact_path, cascade_path = workspace_path(compact_entry["output_path"]), workspace_path(cascade_entry["output_path"])
        if sha256(compact_path) != compact_entry["output_sha256"] or sha256(cascade_path) != cascade_entry["output_sha256"]:
            raise RuntimeError(f"Gate 3 proposal pixels changed: {row['record_id']}")
        target = load_rgb_path(workspace_path(row["target_path"]), row["stored_color_order"])
        compact = load_rgb_path(compact_path)
        cascade = load_rgb_path(cascade_path)
        if target.shape != compact.shape or target.shape != cascade.shape:
            raise RuntimeError(f"Gate 3 proposal dimensions changed: {row['record_id']}")
        blend, alpha = best_blend(compact, cascade, target, config["promotion"]["oracle_blend_alpha_candidates"])
        stem = f"{index:04d}_{row['family']}_{hashlib.sha256(row['record_id'].encode()).hexdigest()[:16]}.png"
        blend_path = workspace_path(config["outputs"]["oracle_blend_directory"]) / stem
        blend_sha = save_png(blend_path, blend)
        blend_entry = {
            "candidate_id": BLEND,
            "family": row["family"],
            "group_id": row["group_id"],
            "record_id": row["record_id"],
            "target_path": row["target_path"],
            "output_path": str(blend_path.relative_to(ROOT)),
            "output_sha256": blend_sha,
            "alpha_cascade": alpha,
            "target_access": True,
        }
        blend_manifest.append(blend_entry)
        loaded.append((row, target, compact, blend, blend_entry))
        print(f"GATE3_COMPACT_BLEND records={index + 1}/{count}", flush=True)
    manifest_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(manifest_buffer, fieldnames=list(blend_manifest[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(blend_manifest)
    manifest_content = manifest_buffer.getvalue().encode()
    blend_manifest_path = workspace_path(config["outputs"]["oracle_blend_manifest"])
    write_once(blend_manifest_path, manifest_content)

    os.environ.setdefault("TORCH_HOME", str(ROOT / "models/torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache/foundir-metrics"))
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit(f"GATE3_COMPACT_METRICS_REQUIRE_ONE_VISIBLE_GPU actual={torch.cuda.device_count()}")
    device, pyiqa_psnr, pyiqa_lpips, direct_lpips = metric_models(config)
    metric_rows = []
    with torch.inference_mode():
        for index, (row, target, compact, blend, _) in enumerate(loaded):
            target_tensor = tensor(target, device)
            for candidate, prediction, target_access in ((COMPACT, compact, False), (BLEND, blend, True)):
                prediction_tensor = tensor(prediction, device)
                literal = literal_psnr(target, prediction)
                evaluator = float(pyiqa_psnr(prediction_tensor, target_tensor).item())
                pyiqa_value = float(pyiqa_lpips(prediction_tensor, target_tensor).item())
                direct = float(direct_lpips(prediction_tensor * 2 - 1, target_tensor * 2 - 1).item())
                metric_rows.append(
                    {
                        "candidate_id": candidate,
                        "family": row["family"],
                        "group_id": row["group_id"],
                        "record_id": row["record_id"],
                        "psnr_literal": literal,
                        "psnr_pyiqa": evaluator,
                        "psnr_conservative": min(literal, evaluator),
                        "lpips_pyiqa": pyiqa_value,
                        "lpips_direct": direct,
                        "lpips_conservative": max(pyiqa_value, direct),
                        "ssim": float(structural_similarity(target, prediction, channel_axis=2, data_range=255)),
                        "target_access": target_access,
                    }
                )
            print(f"GATE3_COMPACT_METRICS records={index + 1}/{count}", flush=True)
    rows_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(rows_buffer, fieldnames=list(metric_rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(sorted(metric_rows, key=lambda row: (row["candidate_id"], row["record_id"])))
    rows_content = rows_buffer.getvalue().encode()
    rows_path = workspace_path(config["outputs"]["metric_rows"])
    write_once(rows_path, rows_content)
    summary = aggregate(metric_rows)
    gate2 = json.loads(workspace_path(config["parents"]["gate2_result"]).read_text())["aggregate"]
    fixed, cascade, compact, blend = gate2["P0_FIXED_TRAIN_MEAN_EXPOSURE"], gate2["P1_FOUNDIR_CASCADE"], summary[COMPACT], summary[BLEND]
    family_checks = {}
    for family in config["data"]["outer_families"]:
        compact_family, fixed_family, cascade_family, blend_family = (
            compact["families"][family], fixed["families"][family], cascade["families"][family], blend["families"][family]
        )
        family_checks[family] = {
            "compact_psnr_gain_over_fixed": compact_family["psnr_conservative"]["estimate"] - fixed_family["psnr_conservative"]["estimate"],
            "compact_lpips_change_over_fixed": compact_family["lpips_conservative"]["estimate"] - fixed_family["lpips_conservative"]["estimate"],
            "compact_psnr_gain_over_cascade": compact_family["psnr_conservative"]["estimate"] - cascade_family["psnr_conservative"]["estimate"],
            "compact_lpips_change_over_cascade": compact_family["lpips_conservative"]["estimate"] - cascade_family["lpips_conservative"]["estimate"],
            "oracle_blend_psnr_gain_over_better_single": blend_family["psnr_conservative"]["estimate"] - max(compact_family["psnr_conservative"]["estimate"], cascade_family["psnr_conservative"]["estimate"]),
        }
    promotion = config["promotion"]
    beats_fixed = all(
        checks["compact_psnr_gain_over_fixed"] >= promotion["compact_required_to_beat_fixed_psnr_each_family"]
        and checks["compact_lpips_change_over_fixed"] <= promotion["lpips_max_regression_each_family"]
        for checks in family_checks.values()
    )
    complementary = all(
        checks["oracle_blend_psnr_gain_over_better_single"] >= promotion["oracle_compact_cascade_blend_psnr_gain_each_family"]
        for checks in family_checks.values()
    )
    decision = {
        "compact_beats_fixed_every_family": beats_fixed,
        "compact_complements_cascade_every_family": complementary,
        "compact_admitted_as_gate4_control": beats_fixed or complementary,
        "family_checks": family_checks,
        "qwen_ordinal_and_causal_gate_still_required": True,
    }
    payload = {
        "schema_version": 1,
        "status": "pass_gate3_compact_evaluation_complete",
        "protected_foundir_pixels_used": False,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": config_sha,
        "runner_sha256": sha256(Path(__file__)),
        "training_result_sha256": sha256(training_result_path),
        "metric_rows": str(rows_path.relative_to(ROOT)),
        "metric_rows_sha256": hashlib.sha256(rows_content).hexdigest(),
        "oracle_blend_manifest": str(blend_manifest_path.relative_to(ROOT)),
        "oracle_blend_manifest_sha256": hashlib.sha256(manifest_content).hexdigest(),
        "records": count,
        "metric_rows_count": len(metric_rows),
        "aggregate": summary,
        "decision": decision,
    }
    result_path = workspace_path(config["outputs"]["result"])
    write_once(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(f"FOUNDIR_LOWLIGHT_GATE3_COMPACT_METRIC_PASS admitted={decision['compact_admitted_as_gate4_control']} beats_fixed={beats_fixed} complementary={complementary}")


if __name__ == "__main__":
    main()
