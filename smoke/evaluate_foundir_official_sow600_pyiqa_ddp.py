#!/usr/bin/env python3
"""Evaluate exact-size official SOW600 outputs with the FoundIR pyiqa contract."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smoke.export_foundir_official_sow600_qwen_features_ddp import (  # noqa: E402
    load_contract,
    workspace,
)
from smoke.run_foundir_qwen_five_class_gate_ddp import sha256  # noqa: E402


def tensor(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    value = torch.from_numpy(array).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    return value, (array.shape[1], array.shape[0])


def write_json_once(path: Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"existing official metric sidecar differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(encoded)
    os.link(temporary, path)
    temporary.unlink()


def mean(rows: list[dict], role: str) -> dict[str, float]:
    return {
        name: float(np.mean([float(row[role][name]) for row in rows]))
        for name in ("psnr", "lpips", "ssim")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, rows = load_contract(args.config.resolve())
    output = workspace(config["runtime"]["output_dir"])
    restored_paths = [output / "restored" / f"{row['record_id']}.png" for row in rows]
    if args.check_only:
        import pyiqa

        for name in ("psnr", "lpips", "ssim"):
            pyiqa.create_metric(name, device="cpu")
        present = sum(path.is_file() for path in restored_paths)
        print(
            f"FOUNDIR_OFFICIAL_PYIQA_PREFLIGHT_PASS records=600 "
            f"restored_present={present} pyiqa_version={importlib.metadata.version('pyiqa')} "
            f"config_sha256={config_sha}"
        )
        return
    if any(name not in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        raise RuntimeError("launch official pyiqa with torchrun")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or torch.cuda.device_count() != 8:
        raise RuntimeError("official pyiqa requires exactly eight visible GPUs")
    if any(not path.is_file() for path in restored_paths):
        raise RuntimeError("official restored outputs are incomplete")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Metric ranks are record-sharded and exchange no CUDA tensors.  Gloo is
    # sufficient for the two barriers and keeps evaluation independent of
    # NCCL/NVLink fabric initialization on otherwise healthy GPU nodes.
    dist.init_process_group("gloo")
    started = time.monotonic()
    try:
        import pyiqa

        metrics = {
            "psnr": pyiqa.create_metric("psnr").to(device).eval(),
            "lpips": pyiqa.create_metric("lpips", device=device).eval(),
            "ssim": pyiqa.create_metric("ssim").to(device).eval(),
        }
        sidecars = output / "official_metrics_per_record"
        for index in range(rank, len(rows), world):
            row = rows[index]
            sidecar = sidecars / f"{row['record_id']}.json"
            if sidecar.exists():
                continue
            lq, lq_size = tensor(workspace(row["lq_path"]))
            gt, gt_size = tensor(workspace(row["gt_path"]))
            restored, restored_size = tensor(output / "restored" / f"{row['record_id']}.png")
            expected = (int(row["width"]), int(row["height"]))
            if lq_size != gt_size or gt_size != restored_size or restored_size != expected:
                raise RuntimeError(f"official metric dimensions differ: {row['record_id']}")
            gt = gt.to(device)
            values: dict[str, dict[str, float]] = {}
            with torch.inference_mode():
                for role, prediction in (("input", lq), ("restored", restored)):
                    prediction = prediction.to(device)
                    values[role] = {
                        name: float(metric(prediction, gt).item())
                        for name, metric in metrics.items()
                    }
            write_json_once(sidecar, {
                "schema_version": 1,
                "status": "complete",
                "record_id": row["record_id"],
                "category": row["category"],
                "pair_id": row["pair_id"],
                "width": expected[0],
                "height": expected[1],
                "config_sha256": config_sha,
                **values,
            })
            print(f"FOUNDIR_OFFICIAL_METRIC_PROGRESS rank={rank} complete={index + 1}/600", flush=True)
        dist.barrier()
        if rank == 0:
            paths = [sidecars / f"{row['record_id']}.json" for row in rows]
            if any(not path.is_file() for path in paths):
                raise RuntimeError("official metric sidecars are incomplete")
            payloads = [json.loads(path.read_text()) for path in paths]
            grouped: dict[str, list[dict]] = defaultdict(list)
            for payload in payloads:
                grouped[payload["category"]].append(payload)
            category_metrics = {}
            checks = {}
            for category in config["data"]["records_by_category"]:
                values = grouped[category]
                baseline, restored = mean(values, "input"), mean(values, "restored")
                sow = config["metrics"]["sow"][category]
                psnr_minimum = float(sow["psnr_minimum"])
                lpips_maximum = float(sow["lpips_maximum"])
                category_metrics[category] = {
                    "records": len(values),
                    "input": baseline,
                    "restored": restored,
                    "delta": {
                        "psnr_db": restored["psnr"] - baseline["psnr"],
                        "lpips": restored["lpips"] - baseline["lpips"],
                        "ssim": restored["ssim"] - baseline["ssim"],
                    },
                    "sow": sow,
                    "sow_margin": {
                        "psnr_db": restored["psnr"] - psnr_minimum,
                        "psnr_percent": (restored["psnr"] / psnr_minimum - 1.0) * 100.0,
                        "lpips_absolute": lpips_maximum - restored["lpips"],
                        "lpips_safety_percent": (1.0 - restored["lpips"] / lpips_maximum) * 100.0,
                    },
                }
                checks[category] = {
                    "psnr": restored["psnr"] >= psnr_minimum,
                    "lpips": restored["lpips"] <= lpips_maximum,
                }
            decisions_path = workspace(config["runtime"]["router_decisions"])
            with decisions_path.open(newline="") as handle:
                route_rows = list(csv.DictReader(handle))
            result = {
                "schema_version": 1,
                "status": "pass" if all(all(item.values()) for item in checks.values()) else "failed",
                "all_ten_sow_checks_pass": all(all(item.values()) for item in checks.values()),
                "records": len(payloads),
                "category_metrics": category_metrics,
                "checks": checks,
                "routing": {
                    "counts": dict(Counter(row["status"] for row in route_rows)),
                    "wrong": sum(row["status"] == "wrong" for row in route_rows),
                    "truth_used_as_router_input": False,
                },
                "protocol": {
                    "original_size": True,
                    "resize_crop_or_border_shave": False,
                    "color_space": "RGB",
                    "implementation": "pyiqa",
                    "pyiqa_version": importlib.metadata.version("pyiqa"),
                    "official_metric_script": "third_party/foundir/cal_metrics.py",
                    "official_metric_script_sha256": sha256(ROOT / "third_party/foundir/cal_metrics.py"),
                    "aggregation": "unweighted arithmetic mean within each category",
                },
                "config_sha256": config_sha,
                "wall_seconds": time.monotonic() - started,
            }
            result_path = output / "official_result.json"
            if result_path.exists():
                raise RuntimeError("refusing to overwrite official terminal result")
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print("FOUNDIR_OFFICIAL_SOW600_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
