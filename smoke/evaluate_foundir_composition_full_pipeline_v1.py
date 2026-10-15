#!/usr/bin/env python3
"""Complete the 660-image comparison with released FoundIR specialists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
from smoke.evaluate_foundir_composition_whole_image_v1 import full_psnr, tiled_lpips  # noqa: E402
from smoke.evaluate_foundir_five_type_flux2_ddp import (  # noqa: E402
    synchronize_rank_zero_finalization,
    torchrun_worker_identity,
)
from smoke.run_foundir_lowlight_gate2_teachers import (  # noqa: E402
    SPECIALIST_DEPENDENCY_DIRECTORIES,
    build_specialist,
    build_weather_specialist,
)


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
            raise RuntimeError(f"immutable specialist output differs: {path}")
    else:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        Image.fromarray(pixels, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    return sha256(path)


def read_contract(config_path: Path, *, verify_images: bool) -> tuple[dict, list[dict[str, str]], str]:
    config = json.loads(config_path.read_text())
    config_sha = sha256(config_path)
    if config.get("schema_version") != 1 or config.get("scope", {}).get("official_foundir_test_accessed") is not False:
        raise RuntimeError("full-pipeline scope changed")
    source = config["source"]
    manifest = resolve(source["manifest"])
    if sha256(manifest) != source["manifest_sha256"]:
        raise RuntimeError("composition manifest changed")
    if sha256(resolve(source["generalist_result"])) != source["generalist_result_sha256"]:
        raise RuntimeError("generalist result changed")
    for name in ("weather", "lowlight"):
        spec = config["specialists"][name]
        if sha256(resolve(spec["checkpoint"])) != spec["checkpoint_sha256"]:
            raise RuntimeError(f"{name} specialist checkpoint changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 660 or len({row["record_id"] for row in rows}) != 660:
        raise RuntimeError("composition manifest cardinality changed")
    categories = {name: set(config["specialists"][name]["categories"]) for name in ("weather", "lowlight")}
    if categories["weather"] & categories["lowlight"]:
        raise RuntimeError("specialist category routing overlaps")
    counts = {name: sum(row["category"] in values for row in rows) for name, values in categories.items()}
    if counts != {"weather": 120, "lowlight": 240}:
        raise RuntimeError(f"specialist routing cardinality changed: {counts}")
    if verify_images:
        generalist = resolve(source["generalist_output_dir"])
        sidecars = resolve(source["generalist_sidecar_dir"])
        for row in rows:
            image = generalist / f"{row['record_id']}.png"
            sidecar = sidecars / f"{row['record_id']}.json"
            if not image.is_file() or not sidecar.is_file():
                raise RuntimeError(f"missing frozen generalist evidence: {row['record_id']}")
            with Image.open(image) as opened:
                if opened.size != (int(row["width"]), int(row["height"])):
                    raise RuntimeError(f"generalist dimensions changed: {row['record_id']}")
    return config, rows, config_sha


def padded(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    height, width = tensor.shape[-2:]
    bottom, right = (-height) % multiple, (-width) % multiple
    if bottom or right:
        tensor = F.pad(tensor, (0, right, 0, bottom), mode="reflect")
    return tensor, height, width


def quantize(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).cpu().numpy()


def run_weather(model, source: np.ndarray, device: torch.device) -> tuple[np.ndarray, int]:
    tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
    tensor, height, width = padded(tensor, 512)
    output = torch.zeros_like(tensor)
    tiles = 0
    with torch.inference_mode():
        for y in range(0, tensor.shape[-2], 512):
            for x in range(0, tensor.shape[-1], 512):
                output[:, :, y:y + 512, x:x + 512] = model(tensor[:, :, y:y + 512, x:x + 512])
                tiles += 1
    return quantize(output[:, :, :height, :width]), tiles


def run_lowlight(model, source: np.ndarray, device: torch.device) -> tuple[np.ndarray, int]:
    tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).float().div_(255).unsqueeze(0).to(device)
    tensor, height, width = padded(tensor, 128)
    with torch.inference_mode():
        output = model.restoration_network(tensor)[:, :, :height, :width]
    return quantize(output), 1


def run_specialist(config: dict, rows: Sequence[Mapping[str, str]], config_sha: str, name: str) -> None:
    rank, world, local_rank = torchrun_worker_identity()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    gpu_name = torch.cuda.get_device_properties(device).name
    if "H100" not in gpu_name and "H200" not in gpu_name:
        raise RuntimeError(f"unsupported GPU: {gpu_name}")
    extra = resolve(config["runtime"]["extra_packages"])
    sys.path.insert(0, str(extra))
    for dependency in reversed(SPECIALIST_DEPENDENCY_DIRECTORIES):
        sys.path.insert(0, str(dependency))
    spec = config["specialists"][name]
    selected = [row for row in rows if row["category"] in set(spec["categories"])]
    model_config = {"specialist": {"repository": config["specialists"]["repository"], "checkpoint": spec["checkpoint"]}}
    model = build_weather_specialist(model_config, device) if name == "weather" else build_specialist(model_config, device)
    model.eval().requires_grad_(False)
    source_root = resolve(config["source"]["generalist_output_dir"])
    output_root = resolve(config["runtime"]["output_dir"])
    for index in range(rank, len(selected), world):
        row = selected[index]
        destination = output_root / "specialist" / "images" / f"{row['record_id']}.png"
        metadata = output_root / "specialist" / "per_record" / f"{row['record_id']}.json"
        source_path = source_root / f"{row['record_id']}.png"
        if destination.exists() or metadata.exists():
            if not destination.is_file() or not metadata.is_file():
                raise RuntimeError(f"partial specialist artifact: {row['record_id']}")
            cached = json.loads(metadata.read_text())
            expected = {
                "status": "complete", "config_sha256": config_sha,
                "record_id": row["record_id"], "category": row["category"],
                "specialist": name, "protocol": spec["protocol"],
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "generalist_input_sha256": sha256(source_path),
                "width": int(row["width"]), "height": int(row["height"]),
                "model_weight_updates": False, "official_foundir_test_accessed": False,
            }
            if any(cached.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"cached specialist identity changed: {row['record_id']}")
            if cached.get("output_sha256") != sha256(destination):
                raise RuntimeError(f"cached specialist output changed: {row['record_id']}")
            with Image.open(destination) as opened:
                if opened.size != (int(row["width"]), int(row["height"])):
                    raise RuntimeError(f"cached specialist dimensions changed: {row['record_id']}")
            print(f"FOUNDIR_SPECIALIST_RESUME name={name} rank={rank} record={index+1}/{len(selected)}", flush=True)
            continue
        source = load_rgb(source_path)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        result, tiles = run_weather(model, source, device) if name == "weather" else run_lowlight(model, source, device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        output_sha = save_png(destination, result)
        write_json(metadata, {
            "schema_version": 1, "status": "complete", "config_sha256": config_sha,
            "record_id": row["record_id"], "category": row["category"], "specialist": name,
            "protocol": spec["protocol"], "checkpoint_sha256": spec["checkpoint_sha256"],
            "generalist_input_sha256": sha256(source_path), "output_sha256": output_sha,
            "width": int(row["width"]), "height": int(row["height"]), "tiles": tiles,
            "seconds": elapsed, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "model_weight_updates": False, "official_foundir_test_accessed": False,
        })
        print(f"FOUNDIR_SPECIALIST name={name} rank={rank} record={index+1}/{len(selected)} tiles={tiles} seconds={elapsed:.3f}", flush=True)


def summarize(values: Sequence[Mapping[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"records": len(values)}
    for role in ("project", "released_foundir_generalist", "full_foundir_generalist_specialist"):
        result[role] = {key: float(np.mean([item[role][key] for item in values])) for key in ("psnr", "lpips")}
    full = result["full_foundir_generalist_specialist"]
    ours = result["project"]
    result["project_minus_full_foundir"] = {"psnr_db": ours["psnr"] - full["psnr"], "lpips": ours["lpips"] - full["lpips"]}
    result["project_win_fraction_vs_full_foundir"] = {
        "psnr": float(np.mean([item["project"]["psnr"] > item["full_foundir_generalist_specialist"]["psnr"] for item in values])),
        "lpips": float(np.mean([item["project"]["lpips"] < item["full_foundir_generalist_specialist"]["lpips"] for item in values])),
    }
    return result


def run_metrics(config: dict, rows: Sequence[Mapping[str, str]], config_sha: str) -> None:
    rank, world, local_rank = torchrun_worker_identity()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    output = resolve(config["runtime"]["output_dir"])
    old_sidecars = resolve(config["source"]["generalist_sidecar_dir"])
    weather = set(config["specialists"]["weather"]["categories"])
    lowlight = set(config["specialists"]["lowlight"]["categories"])
    routed = weather | lowlight
    for index in range(rank, len(rows), world):
        row = rows[index]
        destination = output / "per_record" / f"{row['record_id']}.json"
        if destination.exists():
            cached = json.loads(destination.read_text())
            expected = {
                "status": "complete", "config_sha256": config_sha,
                "record_id": row["record_id"], "category": row["category"],
                "composition_size": int(row["composition_size"]),
                "model_weight_updates": False, "official_foundir_test_accessed": False,
            }
            if any(cached.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"cached full-pipeline metric identity changed: {row['record_id']}")
            continue
        old = json.loads((old_sidecars / f"{row['record_id']}.json").read_text())
        generalist = {key: float(old["foundir"][key]) for key in ("psnr", "lpips")}
        project = {key: float(old["ours"][key]) for key in ("psnr", "lpips")}
        specialist_name = "weather" if row["category"] in weather else "lowlight" if row["category"] in lowlight else None
        if row["category"] in routed:
            gt = load_rgb(resolve(row["gt_path"]))
            prediction = load_rgb(output / "specialist" / "images" / f"{row['record_id']}.png")
            if gt.shape != prediction.shape:
                raise RuntimeError(f"specialist shape mismatch: {row['record_id']}")
            full = {"psnr": full_psnr(gt, prediction), "lpips": tiled_lpips(metric, gt, prediction, device)}
        else:
            full = dict(generalist)
        write_json(destination, {
            "schema_version": 1, "status": "complete", "config_sha256": config_sha,
            "record_id": row["record_id"], "category": row["category"],
            "composition_size": int(row["composition_size"]), "specialist": specialist_name,
            "project": project, "released_foundir_generalist": generalist,
            "full_foundir_generalist_specialist": full,
            "model_weight_updates": False, "official_foundir_test_accessed": False,
        })
        print(f"FOUNDIR_FULL_METRICS rank={rank} record={index+1}/{len(rows)} specialist={specialist_name}", flush=True)

    def finalize() -> None:
        payloads = [json.loads((output / "per_record" / f"{row['record_id']}.json").read_text()) for row in rows]
        by_category: dict[str, list[dict]] = defaultdict(list)
        by_composition: dict[int, list[dict]] = defaultdict(list)
        for payload in payloads:
            by_category[payload["category"]].append(payload)
            by_composition[int(payload["composition_size"])].append(payload)
        result = {
            "schema_version": 1, "status": "foundir_composition_full_pipeline_complete",
            "config_sha256": config_sha, "records": len(payloads), "specialist_records": 360,
            "overall_metrics": summarize(payloads),
            "composition_metrics": {str(key): summarize(value) for key, value in sorted(by_composition.items())},
            "category_metrics": {key: summarize(value) for key, value in sorted(by_category.items())},
            "baseline_identity": "Released FoundIR full pipeline: Generalist then documented degradation-aware specialist",
            "metric_protocol": config["metric_protocol"], "model_weight_updates": False,
            "development_pixels_accessed": True, "official_foundir_test_accessed": False,
        }
        write_json(output / "result.json", result)
        print("FOUNDIR_COMPOSITION_FULL_PIPELINE_COMPLETE " + json.dumps(result["overall_metrics"], sort_keys=True), flush=True)

    synchronize_rank_zero_finalization(
        finalize, rank=rank, world=world, output=output,
        identity={"config_sha256": config_sha, "records": len(rows)},
        phase="FoundIR full-pipeline metrics", timeout_seconds=3600,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "weather", "lowlight", "metrics"), required=True)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config, rows, config_sha = read_contract(config_path, verify_images=args.mode == "preflight")
    if args.mode == "preflight":
        print(json.dumps({"status": "FOUNDIR_COMPOSITION_FULL_PIPELINE_PREFLIGHT_PASS", "records": 660, "specialist_records": 360, "official_foundir_test_accessed": False}, sort_keys=True))
    elif args.mode in ("weather", "lowlight"):
        run_specialist(config, rows, config_sha, args.mode)
    else:
        run_metrics(config, rows, config_sha)


if __name__ == "__main__":
    main()
