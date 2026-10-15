#!/usr/bin/env python3
"""Build advisor triptychs from immutable FLUX.2 sealed-internal outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
TASK_CATEGORIES = {
    "blur": "01Blur",
    "noise": "05Noise",
    "haze": "08Haze",
    "rain": "10Rain",
    "low_light": "14Lowlight",
}
QUANTILES = (1 / 3, 2 / 3)
PANEL_SIZE = 512
HEADER_HEIGHT = 58


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing advisor artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def select_quantile_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["psnr"]), row["record_id"]))
    selected: list[dict[str, Any]] = []
    used_groups: set[str] = set()
    for quantile in QUANTILES:
        target = round((len(ordered) - 1) * quantile)
        candidates = sorted(
            range(len(ordered)), key=lambda index: (abs(index - target), index)
        )
        choice = next(
            ordered[index]
            for index in candidates
            if ordered[index]["source_group"] not in used_groups
            and ordered[index] not in selected
        )
        choice = dict(choice)
        choice["selection_quantile"] = quantile
        choice["selection_rank_zero_based"] = ordered.index(next(row for row in ordered if row["record_id"] == choice["record_id"]))
        selected.append(choice)
        used_groups.add(choice["source_group"])
    return selected


def aligned_crop(path: Path, crop_xy: list[int]) -> Image.Image:
    x, y = map(int, crop_xy)
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        return image.crop((x, y, x + PANEL_SIZE, y + PANEL_SIZE)).copy()


def build_triptych(lq: Image.Image, output: Image.Image, gt: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_SIZE * 3, PANEL_SIZE + HEADER_HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(25)
    for index, (label, panel) in enumerate(
        (("Input (LQ)", lq), ("FLUX.2 output", output), ("Ground Truth", gt))
    ):
        if panel.size != (PANEL_SIZE, PANEL_SIZE):
            raise RuntimeError(f"unexpected panel size: {panel.size}")
        x = index * PANEL_SIZE
        canvas.paste(panel, (x, HEADER_HEIGHT))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text((x + (PANEL_SIZE - text_width) // 2, 14), label, fill="black", font=font)
        if index:
            draw.line((x, 0, x, canvas.height), fill=(80, 80, 80), width=2)
    return canvas


def png_bytes(image: Image.Image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "manifests/foundir_five_type_pairs_v1.csv"
    )
    parser.add_argument(
        "--sealed-dir",
        type=Path,
        default=ROOT / "checkpoints/foundir_five_type_flux2_dit_v3/sealed_internal/qwen",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/advisor_visuals/flux2_v3_sealed_triptychs_2026-09-11",
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    sealed_dir = args.sealed_dir.resolve()
    output_dir = args.output.resolve()
    for path in (manifest_path, sealed_dir, output_dir):
        path.relative_to(ROOT)

    with manifest_path.open(newline="") as handle:
        manifest = {row["record_id"]: row for row in csv.DictReader(handle)}

    by_category: dict[str, list[dict[str, Any]]] = {value: [] for value in TASK_CATEGORIES.values()}
    for sidecar_path in sorted((sealed_dir / "per_record").glob("*.json")):
        sidecar = json.loads(sidecar_path.read_text())
        category = sidecar.get("category")
        if category not in by_category:
            continue
        if sidecar.get("split") != "internal_test" or sidecar.get("status") != "complete":
            raise RuntimeError(f"invalid sealed sidecar: {sidecar_path}")
        record = manifest.get(sidecar["record_id"])
        if record is None or record["split"] != "internal_test" or record["category"] != category:
            raise RuntimeError(f"sidecar/manifest identity mismatch: {sidecar_path}")
        output_path = ROOT / sidecar["output_path"]
        if sha256(output_path) != sidecar["output_sha256"]:
            raise RuntimeError(f"sealed output changed: {output_path}")
        if sha256(ROOT / record["lq_path"]) != record["lq_sha256"]:
            raise RuntimeError(f"LQ changed: {record['record_id']}")
        if sha256(ROOT / record["gt_path"]) != record["gt_sha256"]:
            raise RuntimeError(f"GT changed: {record['record_id']}")
        by_category[category].append({**sidecar, "sidecar_path": sidecar_path.as_posix()})

    artifact_rows: list[dict[str, Any]] = []
    triptychs: list[Image.Image] = []
    for task, category in TASK_CATEGORIES.items():
        rows = by_category[category]
        if len(rows) != 60:
            raise RuntimeError(f"expected 60 sealed isolated records for {task}, got {len(rows)}")
        for example_index, selected in enumerate(select_quantile_records(rows), start=1):
            record = manifest[selected["record_id"]]
            lq = aligned_crop(ROOT / record["lq_path"], selected["crop_xy"])
            gt = aligned_crop(ROOT / record["gt_path"], selected["crop_xy"])
            with Image.open(ROOT / selected["output_path"]) as handle:
                restored = handle.convert("RGB").copy()
            triptych = build_triptych(lq, restored, gt)
            filename = f"{task}_{example_index}_{selected['record_id']}.png"
            path = output_dir / filename
            content = png_bytes(triptych)
            write_immutable(path, content)
            triptychs.append(triptych)
            artifact_rows.append(
                {
                    "task": task,
                    "category": category,
                    "example": example_index,
                    "selection_policy": "approximately_1_3_and_2_3_psnr_quantiles_distinct_source_groups",
                    "selection_quantile": selected["selection_quantile"],
                    "selection_rank_zero_based": selected["selection_rank_zero_based"],
                    "population_records": len(rows),
                    "record_id": selected["record_id"],
                    "source_group": selected["source_group"],
                    "crop_xy": selected["crop_xy"],
                    "input_psnr": selected["input_psnr"],
                    "output_psnr": selected["psnr"],
                    "output_lpips": selected["lpips"],
                    "lq_path": record["lq_path"],
                    "flux_output_path": selected["output_path"],
                    "gt_path": record["gt_path"],
                    "triptych_path": path.relative_to(ROOT).as_posix(),
                    "triptych_sha256": hashlib.sha256(content).hexdigest(),
                }
            )

    overview = Image.new("RGB", (PANEL_SIZE * 3, len(triptychs) * (PANEL_SIZE + HEADER_HEIGHT)), "white")
    for index, triptych in enumerate(triptychs):
        overview.paste(triptych, (0, index * triptych.height))
    overview_path = output_dir / "all_10_triptychs_vertical.png"
    overview_content = png_bytes(overview)
    write_immutable(overview_path, overview_content)

    manifest_payload = {
        "schema_version": 1,
        "status": "advisor_visual_package_complete",
        "system": "VLM-guided FLUX.2 Klein 4B v3 selected step 42500",
        "evaluation_source": "foundir_training_archive_source_group_disjoint_sealed_internal",
        "official_foundir_test_accessed": False,
        "new_inference_performed": False,
        "selection_policy": "two deterministic representative PSNR quantiles per isolated task; no best-case selection",
        "records": artifact_rows,
        "overview_path": overview_path.relative_to(ROOT).as_posix(),
        "overview_sha256": hashlib.sha256(overview_content).hexdigest(),
    }
    write_immutable(
        output_dir / "manifest.json",
        (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(f"FOUNDIR_FLUX2_ADVISOR_TRIPTYCHS_PASS records={len(artifact_rows)} output={output_dir}")


if __name__ == "__main__":
    main()
