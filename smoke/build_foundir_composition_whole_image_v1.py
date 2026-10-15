#!/usr/bin/env python3
"""Freeze 660 routed validation records for whole-image FoundIR comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/foundir_final_eleven_all_in_one_development_rehearsal_v3.json"
MANIFEST = ROOT / "manifests/foundir_composition_whole_image_v1.csv"
CONFIG = ROOT / "configs/foundir_composition_whole_image_v1.json"
FOUNDIR_CONFIG = ROOT / "configs/foundir_five_type_flux2_dit_v3.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def main() -> None:
    parent = json.loads(PARENT.read_text())
    source = ROOT / parent["data"]["manifest"]
    if sha256(source) != parent["data"]["manifest_sha256"]:
        raise RuntimeError("parent routed manifest changed")
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    selected = []
    for category in parent["orders"]:
        values = sorted(grouped[category], key=lambda row: row["record_id"])
        if len(values) < 60:
            raise RuntimeError(f"not enough routed records for {category}")
        selected.extend(values[:60])
    selected.sort(key=lambda row: (row["category"], row["record_id"]))
    counts = Counter(row["category"] for row in selected)
    if len(selected) != 660 or set(counts.values()) != {60}:
        raise RuntimeError("whole-image selection is not 60 records per category")

    from io import StringIO
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(selected[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)
    replace(MANIFEST, buffer.getvalue().encode())

    config = json.loads(json.dumps(parent))
    config["status"] = "user_authorized_final_eleven_all_in_one_development_rehearsal_v4"
    config["scope"]["limitation"] = (
        "Whole-image validation comparison on 60 frozen routed records per FoundIR category; "
        "not the protected official test set."
    )
    config["data"] = {
        "manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST),
        "records": 660,
        "records_by_category": dict(sorted(counts.items())),
        "source_groups": len({row["source_group"] for row in selected}),
        "crop_size": 512,
        "crop_seed": 2026092201,
    }
    config["whole_image"] = {
        "ours_tile_size": 512,
        "ours_overlap": 256,
        "foundir_tile_size": 1024,
        "foundir_overlap": 512,
        "overlap_merge": "uniform_mean_then_round_uint8",
        "psnr": "full_rgb_uint8_image",
        "lpips": "area_weighted_nonoverlap_512_rgb_uint8",
        "released_foundir_config": FOUNDIR_CONFIG.relative_to(ROOT).as_posix(),
        "released_foundir_config_sha256": sha256(FOUNDIR_CONFIG),
    }
    config["runtime"] = {
        "python": ".venv/flux2-smoke/bin/python",
        "torch_home": "models/torch",
        "output_dir": "checkpoints/foundir_composition_whole_image_v1",
        "log": "logs/foundir_composition_whole_image_v1.log",
        "gpu_telemetry": "logs/foundir_composition_whole_image_v1_gpu.csv",
    }
    config["visual_record_ids"] = []
    replace(CONFIG, (json.dumps(config, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({
        "status": "FOUNDIR_COMPOSITION_WHOLE_IMAGE_CONTRACT_READY",
        "records": 660,
        "records_by_category": dict(sorted(counts.items())),
        "source_groups": config["data"]["source_groups"],
        "manifest_sha256": config["data"]["manifest_sha256"],
        "official_foundir_test_accessed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
