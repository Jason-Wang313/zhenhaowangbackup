#!/usr/bin/env python3
"""Render labeled contact sheets for real-sanity v2 pre-inference screening."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import write_new_or_verify

INDEX = ROOT / "manifests/qwen_multidegradation_real_sanity_v2_candidates.json"
OUTPUT = ROOT / "reports/qwen_multidegradation_real_sanity_v2_candidate_sheets"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_grid(records: list[dict[str, object]], title: str) -> Image.Image:
    columns, tile_width, image_height, label_height = 5, 320, 230, 40
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * tile_width, 60 + rows * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), title, fill="black", font=font(27))
    label_font = font(17)
    for index, record in enumerate(records):
        column, row = index % columns, index // columns
        left, top = column * tile_width, 60 + row * (image_height + label_height)
        with Image.open(ROOT / str(record["path"])) as handle:
            image = ImageOps.exif_transpose(handle.convert("RGB"))
        image.thumbnail((tile_width - 8, image_height - 8), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (tile_width, image_height), "#dddddd")
        tile.paste(image, ((tile_width - image.width) // 2, (image_height - image.height) // 2))
        canvas.paste(tile, (left, top))
        draw.text((left + 5, top + image_height + 8), str(record["candidate_id"]), fill="black", font=label_font)
    return canvas


def render_noise(records: list[dict[str, object]]) -> Image.Image:
    row_height, width = 590, 1600
    canvas = Image.new("RGB", (width, 60 + row_height * len(records)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 14), "noise: overview (left) and native center crop (right)", fill="black", font=font(27))
    label_font = font(21)
    for index, record in enumerate(records):
        top = 60 + index * row_height
        with Image.open(ROOT / str(record["path"])) as handle:
            image = ImageOps.exif_transpose(handle.convert("RGB"))
        overview = image.copy()
        overview.thumbnail((760, 520), Image.Resampling.LANCZOS)
        canvas.paste(overview, ((780 - overview.width) // 2, top + 45 + (520 - overview.height) // 2))
        crop_size = min(512, image.width, image.height)
        left = (image.width - crop_size) // 2
        upper = (image.height - crop_size) // 2
        crop = image.crop((left, upper, left + crop_size, upper + crop_size))
        crop = crop.resize((520, 520), Image.Resampling.NEAREST)
        canvas.paste(crop, (950, top + 45))
        draw.text((15, top + 10), str(record["candidate_id"]), fill="black", font=label_font)
        draw.line((0, top + row_height - 1, width, top + row_height - 1), fill="#777777")
    return canvas


def save(image: Image.Image, name: str) -> None:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=94, subsampling=0)
    write_new_or_verify(OUTPUT / name, buffer.getvalue())


def main() -> None:
    payload = json.loads(INDEX.read_text())
    if payload.get("status") != "preinference_screening_candidates_not_final_evaluation_pool":
        raise RuntimeError("candidate index is not pre-inference screening evidence")
    records = payload["records"]
    for category in ("blur", "haze", "low_light"):
        subset = [record for record in records if record["expected_primary_type"] == category]
        save(render_grid(subset, f"{category}: pre-inference candidates"), f"{category}.jpg")
    noise = [record for record in records if record["expected_primary_type"] == "noise"]
    save(render_noise(noise), "noise.jpg")
    blur_after_index = ROOT / "manifests/qwen_multidegradation_realblur_after_selection_v2.json"
    if blur_after_index.is_file():
        blur_after = json.loads(blur_after_index.read_text())["candidate_records"]
        save(render_grid(blur_after, "blur: all transformed AFTER candidates"), "blur_after_all.jpg")
    print(f"QWEN_REAL_SANITY_V2_CANDIDATE_RENDER_PASS root={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
