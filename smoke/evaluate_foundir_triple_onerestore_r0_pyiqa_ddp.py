#!/usr/bin/env python3
"""Evaluate the three frozen R0 arms without using held-out data for selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/foundir_triple_onerestore_r0_v1.json"
EXPECTED_CONFIG_SHA = "415ffaa126393e0125ca070830596c413645919836e290b7a16970b2f27c2c64"
EXPECTED_ALEXNET_SHA = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
EXPECTED_LPIPS_SHA = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json_atomic(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def write_text_atomic(path: Path, text: str) -> None:
    payload = text.encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)


def composition_name(record: dict[str, Any]) -> str:
    return "+".join(map(str, record["degradations"]))


def validate(config: dict[str, Any], require_outputs: bool) -> dict[str, list[dict[str, Any]]]:
    if sha256_file(CONFIG_PATH) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("frozen config hash mismatch")
    torch_root = Path(os.environ.get("TORCH_HOME", ROOT / "models/torch"))
    weights = {
        torch_root / "hub/checkpoints/alexnet-owt-7be5be79.pth": EXPECTED_ALEXNET_SHA,
        torch_root / "hub/pyiqa/LPIPS_v0.1_alex-df73285e.pth": EXPECTED_LPIPS_SHA,
    }
    for path, expected in weights.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"offline metric weight missing or changed: {path}")
    rows = load_jsonl(ROOT / config["parent_protocol"]["manifest"])
    roles = {
        role: sorted(
            (row for row in rows if row["protocol_role"] == role),
            key=lambda row: str(row["parent_record_id"]),
        )
        for role in ("development", "heldout_triple")
    }
    if len(roles["development"]) != 30 or len(roles["heldout_triple"]) != 40:
        raise RuntimeError("frozen evaluation cardinality changed")
    if any(int(row["composition_size"]) > 2 for row in roles["development"]):
        raise RuntimeError("triple checkpoint-selection leakage")
    if any(int(row["composition_size"]) != 3 for row in roles["heldout_triple"]):
        raise RuntimeError("heldout role is not triple-only")
    if require_outputs:
        output_root = ROOT / config["runtime"]["output_root"]
        for arm in config["arms"]:
            checkpoint = output_root / arm["name"] / f"step_{int(config['training']['total_steps_per_arm']):06d}.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"missing final checkpoint: {checkpoint}")
            for role, role_rows in roles.items():
                for row in role_rows:
                    image = output_root / arm["name"] / "outputs" / role / f"{row['parent_record_id']}.png"
                    sidecar = image.with_suffix(".json")
                    if not image.is_file() or not sidecar.is_file():
                        raise RuntimeError(f"missing output or provenance sidecar: {image}")
                    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
                    if sha256_file(image) != metadata["output_sha256"]:
                        raise RuntimeError(f"output hash mismatch: {image}")
                    if metadata.get("official_foundir_test_accessed") is not False:
                        raise RuntimeError("official test access flag changed")
                    if metadata.get("foundir_60_image_triple_accessed") is not False:
                        raise RuntimeError("FoundIR-60 access flag changed")
    return roles


def init_distributed() -> tuple[int, int, torch.device]:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise RuntimeError("frozen R0 evaluation requires exactly eight ranks")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    names: list[str | None] = [None] * world
    name = torch.cuda.get_device_name(device)
    family = "H200" if "H200" in name else "H100" if "H100" in name else ""
    dist.all_gather_object(names, family)
    if not family or len(set(names)) != 1:
        raise RuntimeError(f"unsupported or mixed GPU families: {names}")
    return rank, world, device


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot summarize empty metric rows")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["composition"]].append(row)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "count": len(items),
            "psnr": sum(float(item["psnr"]) for item in items) / len(items),
            "lpips": sum(float(item["lpips"]) for item in items) / len(items),
        }

    return {
        **stats(rows),
        "by_composition": {name: stats(grouped[name]) for name in sorted(grouped)},
    }


def evaluate(
    config: dict[str, Any],
    roles: dict[str, list[dict[str, Any]]],
    rank: int,
    world: int,
    device: torch.device,
) -> None:
    import pyiqa

    lpips_metric = pyiqa.create_metric("lpips", device=device).eval()
    output_root = ROOT / config["runtime"]["output_root"]
    local_rows: list[dict[str, Any]] = []
    work = [
        (arm["name"], role, record)
        for arm in config["arms"]
        for role in ("development", "heldout_triple")
        for record in roles[role]
    ]
    for work_index, (arm, role, record) in enumerate(work):
        if work_index % world != rank:
            continue
        output = output_root / arm / "outputs" / role / f"{record['parent_record_id']}.png"
        provenance = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        prediction = load_rgb(output, device)
        target_path = ROOT / provenance["gt_path"]
        if sha256_file(target_path) != provenance["gt_sha256"]:
            raise RuntimeError(f"target hash mismatch: {target_path}")
        target = load_rgb(target_path, device)
        mse = torch.mean((prediction - target) ** 2)
        psnr = float((-10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))).item())
        lpips_value = float(lpips_metric(prediction, target).item())
        if not math.isfinite(psnr) or not math.isfinite(lpips_value):
            raise RuntimeError(f"nonfinite metric: {output}")
        row = {
            "arm": arm,
            "role": role,
            "parent_record_id": record["parent_record_id"],
            "composition": composition_name(record),
            "composition_size": int(record["composition_size"]),
            "psnr": psnr,
            "lpips": lpips_value,
            "output_path": output.relative_to(ROOT).as_posix(),
            "output_sha256": provenance["output_sha256"],
            "gt_path": provenance["gt_path"],
            "gt_sha256": provenance["gt_sha256"],
        }
        write_json_atomic(output.with_suffix(".metrics.json"), row)
        local_rows.append(row)
    gathered: list[list[dict[str, Any]] | None] = [None] * world
    dist.all_gather_object(gathered, local_rows)
    if rank != 0:
        return
    rows = sorted(
        (row for shard in gathered if shard is not None for row in shard),
        key=lambda row: (row["arm"], row["role"], row["parent_record_id"]),
    )
    if len(rows) != 3 * (30 + 40):
        raise RuntimeError(f"expected 210 metric rows, got {len(rows)}")
    arms: dict[str, Any] = {}
    for arm in config["arms"]:
        name = str(arm["name"])
        arms[name] = {
            role: summarize([row for row in rows if row["arm"] == name and row["role"] == role])
            for role in ("development", "heldout_triple")
        }
    a = arms["A_unconditioned"]["development"]
    b = arms["B_fixed_categorical"]["development"]
    c = arms["C_fixed_categorical_cdrl"]["development"]
    margins = {
        "B_minus_A_development_psnr_db": b["psnr"] - a["psnr"],
        "B_minus_A_development_lpips": b["lpips"] - a["lpips"],
        "C_minus_B_development_psnr_db": c["psnr"] - b["psnr"],
        "C_minus_B_development_lpips": c["lpips"] - b["lpips"],
    }
    thresholds = config["evaluation"]["primary_gate"]
    checks = {
        "B_psnr_not_worse_than_A": margins["B_minus_A_development_psnr_db"]
        >= float(thresholds["B_minus_A_development_psnr_min_db"]),
        "B_lpips_not_worse_than_A": margins["B_minus_A_development_lpips"]
        <= float(thresholds["B_minus_A_development_lpips_max"]),
        "C_psnr_within_0.2db_of_B": margins["C_minus_B_development_psnr_db"]
        >= float(thresholds["C_minus_B_development_psnr_min_db"]),
        "C_lpips_not_worse_than_B": margins["C_minus_B_development_lpips"]
        <= float(thresholds["C_minus_B_development_lpips_max"]),
        "heldout_triple_metrics_reported_without_selection": all(
            arms[name]["heldout_triple"]["count"] == 40 for name in arms
        ),
    }
    result = {
        "schema_version": 1,
        "status": "foundir_triple_onerestore_r0_v1_terminal",
        "scientific_role": config["scientific_role"],
        "config_sha256": EXPECTED_CONFIG_SHA,
        "metric_implementation": {
            "psnr": "RGB_[0,1]_per_image_mean_then_dataset_mean",
            "lpips": "pyiqa_LPIPS_Alex_per_image_mean_then_dataset_mean",
            "pyiqa_version": importlib.metadata.version("pyiqa"),
            "torch_version": torch.__version__,
            "alexnet_sha256": EXPECTED_ALEXNET_SHA,
            "lpips_weight_sha256": EXPECTED_LPIPS_SHA,
        },
        "arms": arms,
        "development_margins": margins,
        "primary_gate_checks": checks,
        "primary_gate_pass": all(checks.values()),
        "metric_rows": rows,
        "heldout_used_for_checkpoint_selection": False,
        "official_foundir_test_accessed": False,
        "foundir_60_image_triple_accessed": False,
        "next_action": "STOP after R0; do not automatically enter R1, R2, or CSR",
    }
    table_rows = []
    for name, values in arms.items():
        development = values["development"]
        triple = values["heldout_triple"]
        table_rows.append(
            f"| {name} | {development['psnr']:.6f} | {development['lpips']:.6f} | "
            f"{triple['psnr']:.6f} | {triple['lpips']:.6f} |"
        )
    check_rows = [f"- {name}: `{'PASS' if passed else 'FAIL'}`" for name, passed in checks.items()]
    report = "\n".join(
        [
            "# FoundIR triple OneRestore-style R0 terminal result",
            "",
            "Status: R0 terminal; stop after this diagnostic.",
            "",
            "| Arm | Development PSNR | Development LPIPS | Held-out triple PSNR | Held-out triple LPIPS |",
            "|---|---:|---:|---:|---:|",
            *table_rows,
            "",
            f"Primary gate: `{'PASS' if result['primary_gate_pass'] else 'FAIL'}`",
            "",
            *check_rows,
            "",
            "The held-out triple set was accessed only after all three final checkpoints were frozen and was not used for checkpoint selection.",
            "This is a synthetic diagnostic, not a FLUX.2 mainline result and not an official FoundIR/SOW result.",
            "The implementation is OneRestore-style, not an exact CDD-11 reproduction: it uses frozen seeded Gaussian descriptors rather than GloVe semantic descriptors, and a stable `[0,1]` residual clamp rather than the public output normalization.",
            "Neither the official FoundIR test nor the 60-image FoundIR triple endpoint was accessed.",
            "",
            "Next action: **STOP**. Do not automatically enter R1, R2, continuous severity, spatial factorization, or CSR.",
            "",
        ]
    )
    write_text_atomic(
        ROOT / "reports/foundir_triple_onerestore_r0_v1_terminal_result_2026-09-26.md",
        report,
    )
    # Commit result.json last: its presence is the terminal marker.  An
    # interrupted report write can therefore be resumed safely, while a
    # terminal result always implies that all human-readable artifacts exist.
    write_json_atomic(output_root / "result.json", result)
    print("FOUNDIR_TRIPLE_ONERESTORE_R0_RESULT " + json.dumps({
        "primary_gate_pass": result["primary_gate_pass"],
        "development_margins": margins,
        "heldout_triple": {name: value["heldout_triple"] for name, value in arms.items()},
    }, sort_keys=True), flush=True)


def preflight(config: dict[str, Any], roles: dict[str, list[dict[str, Any]]]) -> None:
    import pyiqa

    metric = pyiqa.create_metric("lpips", device="cpu").eval()
    left = torch.zeros(1, 3, 32, 32)
    right = torch.ones(1, 3, 32, 32)
    value = metric(left, right)
    if value.numel() != 1 or not torch.isfinite(value):
        raise RuntimeError("LPIPS CPU preflight failed")
    print(
        "FOUNDIR_TRIPLE_ONERESTORE_R0_METRIC_PREFLIGHT_PASS "
        f"development={len(roles['development'])} heldout_triple={len(roles['heldout_triple'])} "
        f"pyiqa={importlib.metadata.version('pyiqa')} config_sha256={EXPECTED_CONFIG_SHA}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.config.resolve() != CONFIG_PATH:
        raise RuntimeError("only the frozen R0 config is accepted")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    roles = validate(config, require_outputs=not args.preflight)
    if args.preflight:
        preflight(config, roles)
        return
    rank, world, device = init_distributed()
    evaluate(config, roles, rank, world, device)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
