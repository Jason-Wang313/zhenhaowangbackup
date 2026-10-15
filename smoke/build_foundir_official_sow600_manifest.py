#!/usr/bin/env python3
"""Build a fail-closed manifest for the five advisor-authorized FoundIR SOW categories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED = {
    "01Blur": 150,
    "05Noise": 100,
    "08Haze": 200,
    "10Rain": 100,
    "14Lowlight": 50,
}
AUTHORIZATION = (
    ROOT
    / "reports/advisor_governance/"
    "foundir_official_sow600_qwen_authorization_2026-09-21.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to replace differing manifest: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.link(temporary, path)
    temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT / "downloads/foundir_official",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "manifests/foundir_official_sow600_qwen_v1.csv",
    )
    args = parser.parse_args()

    authorization = json.loads(AUTHORIZATION.read_text())
    if (
        authorization.get("status") != "authorized"
        or authorization.get("authorized_scope", {}).get("categories") != AUTHORIZED
        or authorization["authorized_scope"].get("total_pairs") != 600
        or "Qwen inference" not in authorization["authorized_scope"].get("pipeline", "")
    ):
        raise RuntimeError("advisor authorization does not match the SOW600 Qwen scope")

    lq_root = args.runtime_root / "extracted_lq"
    gt_root = args.runtime_root / "extracted_gt"
    rows: list[dict[str, object]] = []
    for category, expected in AUTHORIZED.items():
        # Deliberately enumerate only the five named directories. Never walk the root.
        lq_paths = sorted((lq_root / category).glob("*.png"))
        gt_paths = sorted((gt_root / category).glob("*.png"))
        if len(lq_paths) != expected or len(gt_paths) != expected:
            raise RuntimeError(
                f"official category cardinality changed: {category} "
                f"LQ={len(lq_paths)} GT={len(gt_paths)} expected={expected}"
            )
        lq_by_stem = {path.stem: path for path in lq_paths}
        gt_by_stem = {path.stem: path for path in gt_paths}
        if set(lq_by_stem) != set(gt_by_stem):
            raise RuntimeError(f"official pair identities differ: {category}")
        for pair_id in sorted(lq_by_stem):
            lq_path, gt_path = lq_by_stem[pair_id], gt_by_stem[pair_id]
            with Image.open(lq_path) as lq_image, Image.open(gt_path) as gt_image:
                lq_image.load()
                gt_image.load()
                if lq_image.size != gt_image.size:
                    raise RuntimeError(f"official pair dimensions differ: {category}/{pair_id}")
                width, height = lq_image.size
            rows.append(
                {
                    "record_id": f"official_{category}_{pair_id}",
                    "category": category,
                    "pair_id": pair_id,
                    "lq_path": lq_path.resolve().relative_to(ROOT).as_posix(),
                    "gt_path": gt_path.resolve().relative_to(ROOT).as_posix(),
                    "lq_sha256": sha256(lq_path),
                    "gt_sha256": sha256(gt_path),
                    "width": width,
                    "height": height,
                }
            )
    if len(rows) != 600:
        raise RuntimeError(f"official SOW manifest must contain 600 pairs, got {len(rows)}")

    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_new(args.output, buffer.getvalue().encode())
    print(
        "FOUNDIR_OFFICIAL_SOW600_MANIFEST_PASS "
        + json.dumps(
            {
                "records": len(rows),
                "categories": AUTHORIZED,
                "manifest": args.output.resolve().relative_to(ROOT).as_posix(),
                "manifest_sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
