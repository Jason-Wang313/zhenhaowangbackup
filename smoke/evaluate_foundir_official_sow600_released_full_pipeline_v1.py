#!/usr/bin/env python3
"""Compare frozen project v8 with released FoundIR on official SOW600."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.evaluate_foundir_composition_full_pipeline_v1 import (  # noqa: E402
    run_lowlight,
    run_weather,
)
from smoke.evaluate_foundir_composition_whole_image_v1 import (  # noqa: E402
    pad_minimum,
    tile_boxes,
)
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
)
from smoke.evaluate_foundir_five_type_released_baselines_ddp import (  # noqa: E402
    released_config,
    run_generalist,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_generalist,
    build_specialist,
    build_weather_specialist,
)

CATEGORIES = ("01Blur", "05Noise", "08Haze", "10Rain", "14Lowlight")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path.resolve().relative_to(ROOT.resolve())
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"immutable artifact differs: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def save_png(path: Path, pixels: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        current = load_rgb(path)
        if current.shape != pixels.shape or not np.array_equal(current, pixels):
            raise RuntimeError(f"immutable output differs: {path}")
    else:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    return sha256(path)


def read_contract(config_path: Path, *, verify_images: bool) -> tuple[dict, list[dict[str, str]], str]:
    config = json.loads(config_path.read_text())
    config_sha = sha256(config_path)
    if config.get("schema_version") != 1 or config.get("status") != "user_authorized_official_sow600_released_foundir_comparison":
        raise RuntimeError("official comparison contract changed")
    scope = config.get("scope", {})
    if scope.get("official_foundir_test_accessed") is not True or scope.get("model_weight_updates") is not False or scope.get("training") is not False:
        raise RuntimeError("official comparison scope changed")
    authorization = config["authorization"]
    if sha256(resolve(authorization["path"])) != authorization["sha256"]:
        raise RuntimeError("authorization artifact changed")
    data = config["data"]
    manifest = resolve(data["manifest"])
    if sha256(manifest) != data["manifest_sha256"]:
        raise RuntimeError("official manifest changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 600 or len({row["record_id"] for row in rows}) != 600:
        raise RuntimeError("official manifest cardinality changed")
    counts = {name: sum(row["category"] == name for row in rows) for name in CATEGORIES}
    if counts != data["records_by_category"]:
        raise RuntimeError(f"official category counts changed: {counts}")
    foundir = config["released_foundir"]
    for path_key, sha_key in (
        ("runtime_config", "runtime_config_sha256"),
        ("generalist_checkpoint", "generalist_checkpoint_sha256"),
    ):
        if sha256(resolve(foundir[path_key])) != foundir[sha_key]:
            raise RuntimeError(f"released FoundIR artifact changed: {path_key}")
    for name in ("weather", "lowlight"):
        if sha256(resolve(foundir[name]["checkpoint"])) != foundir[name]["checkpoint_sha256"]:
            raise RuntimeError(f"released FoundIR specialist changed: {name}")
    metrics = config["metrics"]
    if sha256(resolve(metrics["official_metric_script"])) != metrics["official_metric_script_sha256"]:
        raise RuntimeError("official metric implementation changed")
    project = config["project_reference"]
    canonical = subprocess.run(
        ["git", "show", f"{project['canonical_commit']}:{project['result_path_in_commit']}"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    ).stdout
    if sha256_bytes(canonical) != project["result_content_sha256"]:
        raise RuntimeError("canonical project result changed")
    canonical_result = json.loads(canonical)
    for category in CATEGORIES:
        expected = project["category_metrics"][category]
        actual = canonical_result["category_metrics"][category]
        for key in ("records", "psnr", "lpips"):
            if actual[key] != expected[key]:
                raise RuntimeError(f"canonical project metric changed: {category} {key}")
    if verify_images:
        for row in rows:
            for role, path_key, sha_key in (("lq", "lq_path", "lq_sha256"), ("gt", "gt_path", "gt_sha256")):
                path = resolve(row[path_key])
                if not path.is_file() or sha256(path) != row[sha_key]:
                    raise RuntimeError(f"official {role} changed: {row['record_id']}")
                with Image.open(path) as opened:
                    if opened.size != (int(row["width"]), int(row["height"])):
                        raise RuntimeError(f"official dimensions changed: {row['record_id']}")
    return config, rows, config_sha


def setup_device() -> tuple[int, int, torch.device]:
    rank, world, local_rank = torchrun_worker_identity()
    if world != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("official comparison requires exactly eight visible GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    name = torch.cuda.get_device_properties(device).name
    if "H100" not in name and "H200" not in name:
        raise RuntimeError(f"unsupported GPU: {name}")
    return rank, world, device


def quantize(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()


def run_generalist_stage(config: dict, rows: Sequence[Mapping[str, str]], config_sha: str) -> None:
    rank, world, device = setup_device()
    extra = resolve(config["runtime"]["extra_packages"])
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        sys.path.insert(0, str(dependency))
    runtime_config = json.loads(resolve(config["released_foundir"]["runtime_config"]).read_text())
    model = build_generalist(released_config(runtime_config), device)
    spec = config["released_foundir"]
    tile, overlap = int(spec["generalist_tile_size"]), int(spec["generalist_overlap"])
    output = resolve(config["runtime"]["output_dir"])
    for index in range(rank, len(rows), world):
        row = rows[index]
        destination = output / "generalist" / "images" / f"{row['record_id']}.png"
        metadata = output / "generalist" / "per_record" / f"{row['record_id']}.json"
        if destination.exists() or metadata.exists():
            if not destination.is_file() or not metadata.is_file():
                raise RuntimeError(f"partial Generalist artifact: {row['record_id']}")
            cached = json.loads(metadata.read_text())
            if cached.get("output_sha256") != sha256(destination) or cached.get("config_sha256") != config_sha:
                raise RuntimeError(f"cached Generalist identity changed: {row['record_id']}")
            continue
        pixels = load_rgb(resolve(row["lq_path"]))
        source = torch.from_numpy(pixels.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0)
        original_h, original_w = pixels.shape[:2]
        started = time.perf_counter()
        if original_h * original_w <= tile * tile:
            restored = run_generalist(model, source, device, row["record_id"])[0]
            merged = restored.detach().cpu().permute(1, 2, 0).mul(255).numpy()
            tiles = 1
        else:
            padded, keep_h, keep_w = pad_minimum(source, tile)
            height, width = padded.shape[-2:]
            accumulator = np.zeros((height, width, 3), dtype=np.float32)
            counts = np.zeros((height, width, 1), dtype=np.float32)
            boxes = tile_boxes(height, width, tile, overlap)
            for y0, x0, y1, x1 in boxes:
                restored = run_generalist(model, padded[:, :, y0:y1, x0:x1], device, row["record_id"])[0]
                array = restored.detach().cpu().permute(1, 2, 0).mul(255).numpy()
                accumulator[y0:y1, x0:x1] += array
                counts[y0:y1, x0:x1] += 1
            merged, tiles = (accumulator / counts)[:keep_h, :keep_w], len(boxes)
        result = np.rint(merged).clip(0, 255).astype(np.uint8)
        output_sha = save_png(destination, result)
        write_json(metadata, {
            "schema_version": 1, "status": "complete", "config_sha256": config_sha,
            "record_id": row["record_id"], "category": row["category"],
            "checkpoint_sha256": spec["generalist_checkpoint_sha256"],
            "input_sha256": row["lq_sha256"], "output_sha256": output_sha,
            "width": int(row["width"]), "height": int(row["height"]),
            "tiles": tiles, "seconds": time.perf_counter() - started,
            "model_weight_updates": False, "official_foundir_test_accessed": True,
        })
        print(f"FOUNDIR_OFFICIAL_GENERALIST rank={rank} record={index+1}/600 tiles={tiles}", flush=True)


def run_specialist_stage(config: dict, rows: Sequence[Mapping[str, str]], config_sha: str, name: str) -> None:
    rank, world, device = setup_device()
    extra = resolve(config["runtime"]["extra_packages"])
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        sys.path.insert(0, str(dependency))
    released = config["released_foundir"]
    spec = released[name]
    selected = [row for row in rows if row["category"] in set(spec["categories"])]
    model_config = {"specialist": {"repository": released["specialist_repository"], "checkpoint": spec["checkpoint"]}}
    model = build_weather_specialist(model_config, device) if name == "weather" else build_specialist(model_config, device)
    model.eval().requires_grad_(False)
    output = resolve(config["runtime"]["output_dir"])
    for index in range(rank, len(selected), world):
        row = selected[index]
        source_path = output / "generalist" / "images" / f"{row['record_id']}.png"
        destination = output / "specialist" / "images" / f"{row['record_id']}.png"
        metadata = output / "specialist" / "per_record" / f"{row['record_id']}.json"
        if destination.exists() or metadata.exists():
            if not destination.is_file() or not metadata.is_file():
                raise RuntimeError(f"partial Specialist artifact: {row['record_id']}")
            cached = json.loads(metadata.read_text())
            if cached.get("output_sha256") != sha256(destination) or cached.get("config_sha256") != config_sha:
                raise RuntimeError(f"cached Specialist identity changed: {row['record_id']}")
            continue
        source = load_rgb(source_path)
        started = time.perf_counter()
        result, tiles = run_weather(model, source, device) if name == "weather" else run_lowlight(model, source, device)
        output_sha = save_png(destination, result)
        write_json(metadata, {
            "schema_version": 1, "status": "complete", "config_sha256": config_sha,
            "record_id": row["record_id"], "category": row["category"], "specialist": name,
            "protocol": spec["protocol"], "checkpoint_sha256": spec["checkpoint_sha256"],
            "generalist_input_sha256": sha256(source_path), "output_sha256": output_sha,
            "width": int(row["width"]), "height": int(row["height"]),
            "tiles": tiles, "seconds": time.perf_counter() - started,
            "model_weight_updates": False, "official_foundir_test_accessed": True,
        })
        print(f"FOUNDIR_OFFICIAL_SPECIALIST name={name} rank={rank} record={index+1}/{len(selected)} tiles={tiles}", flush=True)


def metric_tensor(path: Path) -> torch.Tensor:
    array = load_rgb(path)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255).unsqueeze(0)


def summarize(values: Sequence[Mapping[str, object]], project: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {"records": len(values), "project": dict(project)}
    for role in ("released_foundir_generalist", "full_foundir_generalist_specialist"):
        result[role] = {
            key: float(np.mean([item[role][key] for item in values]))
            for key in ("psnr", "lpips", "ssim")
        }
    full = result["full_foundir_generalist_specialist"]
    result["project_minus_full_foundir"] = {
        "psnr_db": float(project["psnr"]) - full["psnr"],
        "lpips": float(project["lpips"]) - full["lpips"],
    }
    result["project_beats_full_foundir_on_both"] = (
        float(project["psnr"]) > full["psnr"] and float(project["lpips"]) < full["lpips"]
    )
    return result


def run_metrics(config: dict, rows: Sequence[Mapping[str, str]], config_sha: str) -> None:
    rank, world, device = setup_device()
    import pyiqa
    metrics = {
        "psnr": pyiqa.create_metric("psnr").to(device).eval(),
        "lpips": pyiqa.create_metric("lpips", device=device).eval(),
        "ssim": pyiqa.create_metric("ssim").to(device).eval(),
    }
    output = resolve(config["runtime"]["output_dir"])
    routed = {"08Haze": "weather", "14Lowlight": "lowlight"}
    for index in range(rank, len(rows), world):
        row = rows[index]
        destination = output / "metrics_per_record" / f"{row['record_id']}.json"
        if destination.exists():
            cached = json.loads(destination.read_text())
            if cached.get("config_sha256") != config_sha or cached.get("record_id") != row["record_id"]:
                raise RuntimeError(f"cached metric identity changed: {row['record_id']}")
            continue
        gt = metric_tensor(resolve(row["gt_path"])).to(device)
        generalist_path = output / "generalist" / "images" / f"{row['record_id']}.png"
        specialist = routed.get(row["category"])
        full_path = output / "specialist" / "images" / f"{row['record_id']}.png" if specialist else generalist_path
        values: dict[str, dict[str, float]] = {}
        with torch.inference_mode():
            for role, path in (
                ("released_foundir_generalist", generalist_path),
                ("full_foundir_generalist_specialist", full_path),
            ):
                prediction = metric_tensor(path).to(device)
                if prediction.shape != gt.shape:
                    raise RuntimeError(f"official output dimensions changed: {row['record_id']}")
                values[role] = {key: float(metric(prediction, gt).item()) for key, metric in metrics.items()}
        write_json(destination, {
            "schema_version": 1, "status": "complete", "config_sha256": config_sha,
            "record_id": row["record_id"], "category": row["category"], "specialist": specialist,
            **values, "model_weight_updates": False, "official_foundir_test_accessed": True,
        })
        print(f"FOUNDIR_OFFICIAL_COMPARISON_METRICS rank={rank} record={index+1}/600 specialist={specialist}", flush=True)

    def finalize() -> None:
        payloads = [json.loads((output / "metrics_per_record" / f"{row['record_id']}.json").read_text()) for row in rows]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for payload in payloads:
            grouped[payload["category"]].append(payload)
        project_categories = config["project_reference"]["category_metrics"]
        category_metrics = {
            category: summarize(grouped[category], project_categories[category])
            for category in CATEGORIES
        }
        counts = {category: int(project_categories[category]["records"]) for category in CATEGORIES}
        project_overall = {
            key: float(sum(counts[c] * float(project_categories[c][key]) for c in CATEGORIES) / 600)
            for key in ("psnr", "lpips")
        }
        result = {
            "schema_version": 1,
            "status": "foundir_official_sow600_released_full_pipeline_comparison_complete",
            "config_sha256": config_sha,
            "records": 600,
            "specialist_records": 250,
            "category_metrics": category_metrics,
            "overall_metrics": summarize(payloads, {"records": 600, **project_overall}),
            "all_categories_project_beats_full_foundir_on_both": all(
                value["project_beats_full_foundir_on_both"] for value in category_metrics.values()
            ),
            "baseline_identity": "Released FoundIR Generalist then documented degradation-aware Specialist",
            "project_identity": config["project_reference"]["identity"],
            "metric_protocol": config["metrics"],
            "model_weight_updates": False,
            "official_foundir_test_accessed": True,
        }
        write_json(output / "result.json", result)
        print("FOUNDIR_OFFICIAL_SOW600_RELEASED_FULL_PIPELINE_COMPLETE " + json.dumps(result["overall_metrics"], sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize, rank=rank, world=world, output=output,
        identity={"config_sha256": config_sha, "records": 600},
        phase="official FoundIR baseline metrics", timeout_seconds=3600,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "generalist", "weather", "lowlight", "metrics"), required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config, rows, config_sha = read_contract(config_path, verify_images=args.mode == "preflight")
    if args.mode == "preflight":
        import pyiqa
        for name in ("psnr", "lpips", "ssim"):
            pyiqa.create_metric(name, device="cpu")
        print(json.dumps({
            "status": "FOUNDIR_OFFICIAL_SOW600_RELEASED_FULL_PIPELINE_PREFLIGHT_PASS",
            "records": 600, "specialist_records": 250,
            "pyiqa_version": importlib.metadata.version("pyiqa"),
            "official_foundir_test_accessed": True, "config_sha256": config_sha,
        }, sort_keys=True))
    elif args.mode == "generalist":
        run_generalist_stage(config, rows, config_sha)
    elif args.mode in ("weather", "lowlight"):
        run_specialist_stage(config, rows, config_sha, args.mode)
    else:
        run_metrics(config, rows, config_sha)


if __name__ == "__main__":
    main()
