#!/usr/bin/env python3
"""Render readable real-image Qwen prediction sheets for manual review."""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import write_new_or_verify


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render_sheet(records: list[dict[str, object]], expected: str) -> Image.Image:
    width = 1500
    row_height = 310
    canvas = Image.new("RGB", (width, 70 + row_height * len(records)), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(28)
    body_font = font(21)
    draw.text((20, 18), f"Real sanity: expected primary type = {expected}", fill="black", font=title_font)
    for index, record in enumerate(records):
        top = 70 + index * row_height
        with Image.open(ROOT / str(record["path"])) as handle:
            image = ImageOps.exif_transpose(handle.convert("RGB"))
        image.thumbnail((520, 280), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (540, 290), "#eeeeee")
        tile.paste(image, ((540 - image.width) // 2, (290 - image.height) // 2))
        canvas.paste(tile, (10, top + 5))
        prediction = record.get("raw_output", "not yet inferred")
        lines = [
            str(record["record_id"]),
            f"dataset category: {expected}; absolute severity truth: unavailable",
            f"Qwen: {prediction}",
        ]
        y = top + 18
        for block in lines:
            for line in textwrap.wrap(block, width=78) or [""]:
                draw.text((575, y), line, fill="black", font=body_font)
                y += 30
            y += 8
        draw.line((0, top + row_height - 1, width, top + row_height - 1), fill="#888888", width=1)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json").read_text()
    )
    if args.input_only:
        records = manifest["records"]
        output_root = ROOT / "reports/qwen_multidegradation_real_sanity_inputs_v3"
    else:
        result = json.loads(
            (ROOT / "checkpoints/qwen_multidegradation_lora_v1/real_sanity/predictions.json").read_text()
        )
        records = result["records"]
        output_root = ROOT / "checkpoints/qwen_multidegradation_lora_v1/real_sanity/contact_sheets"
    for expected in ("blur", "noise", "rain", "haze", "low_light"):
        subset = [record for record in records if record["expected_primary_type"] == expected]
        if len(subset) != 5:
            raise RuntimeError(f"real sanity category count changed: {expected}")
        sheet = render_sheet(subset, expected)
        output = output_root / f"{expected}.jpg"
        from io import BytesIO
        buffer = BytesIO()
        sheet.save(buffer, format="JPEG", quality=92, subsampling=0)
        write_new_or_verify(output, buffer.getvalue())
    print(f"QWEN_MULTIDEGRADATION_REAL_SANITY_RENDER_PASS root={output_root}", flush=True)


if __name__ == "__main__":
    main()
