#!/usr/bin/env python3
"""Render ordinary PNG overview sheets for corrective-holdout human review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests/qwen_lowlight_corrective_holdout_candidates.json"
DEFAULT_OUTPUT = (
    ROOT
    / "data/real_lowlight_corrective_holdout/human_annotation_package_20260821"
    / "contact_sheets"
)
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def font(size: int) -> ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def write_new_or_verify(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing contact sheet: {path}")
        return
    path.write_bytes(content)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
        rgb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    left = (size[0] - rgb.width) // 2
    top = (size[1] - rgb.height) // 2
    canvas.paste(rgb, (left, top))
    return canvas


def render_page(
    page_items: list[tuple[int, dict[str, object]]],
    *,
    page_number: int,
    page_count: int,
) -> Image.Image:
    columns, rows = 4, 5
    tile_width, tile_height = 420, 350
    image_height = 270
    header_height = 58
    page = Image.new(
        "RGB", (columns * tile_width, header_height + rows * tile_height), "#202020"
    )
    draw = ImageDraw.Draw(page)
    draw.text(
        (18, 12),
        f"Lowlight holdout candidates — page {page_number}/{page_count}",
        fill="white",
        font=font(26),
    )
    small = font(15)
    tiny = font(12)
    for offset, (review_number, component) in enumerate(page_items):
        column, row = offset % columns, offset // columns
        left = column * tile_width
        top = header_height + row * tile_height
        display_path = ROOT / str(component["display_path"])
        picture = fit_image(display_path, (tile_width - 12, image_height))
        page.paste(picture, (left + 6, top + 4))
        draw.rectangle(
            (left + 5, top + 3, left + tile_width - 6, top + image_height + 4),
            outline="#888888",
            width=1,
        )
        filename = display_path.name
        subsets = "/".join(str(value) for value in component["subsets"])
        draw.text(
            (left + 8, top + image_height + 10),
            f"#{review_number:03d}  severity: ____  usable? Y/N",
            fill="#ffffff",
            font=small,
        )
        draw.text(
            (left + 8, top + image_height + 32),
            f"{subsets} | {filename[:38]} | variants={len(component['members'])}",
            fill="#cccccc",
            font=tiny,
        )
    return page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    if payload.get("status") != "screened_unlabeled_no_model_inference":
        raise RuntimeError("manifest is not a screened unlabeled package")
    components = [
        item
        for item in payload["components"]
        if item["screening_status"] == "pending_human_annotation"
    ]
    if len(components) != 190:
        raise RuntimeError(f"component count changed: {len(components)}")
    indexed = list(enumerate(components, start=1))
    page_size = 20
    page_count = (len(indexed) + page_size - 1) // page_size
    inventory = []
    for page_index in range(page_count):
        page_number = page_index + 1
        page_items = indexed[page_index * page_size : (page_index + 1) * page_size]
        page = render_page(
            page_items, page_number=page_number, page_count=page_count
        )
        from io import BytesIO

        buffer = BytesIO()
        page.save(buffer, format="PNG", optimize=True)
        content = buffer.getvalue()
        path = args.output / f"overview_{page_number:02d}.png"
        write_new_or_verify(path, content)
        inventory.append(
            {
                "page": page_number,
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(content),
                "review_numbers": [number for number, _ in page_items],
            }
        )
    index_content = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode()
    write_new_or_verify(args.output / "inventory.json", index_content)
    print(
        f"CONTACT_SHEETS_PASS pages={page_count} components={len(components)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
