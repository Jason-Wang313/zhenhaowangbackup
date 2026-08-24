#!/usr/bin/env python3
"""Acquire paired references to screen five distinct RealBlur-J scenes."""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import write_new_or_verify

REVISION = "b66421985334b570984036e431006bd13131b63e"
ENDPOINT = (
    "https://datasets-server.huggingface.co/first-rows?dataset="
    "seiya12/instruct-pix2pix-dataset_RealblurJ&config=default&split=train"
)
POOL = ROOT / "data/qwen_multidegradation_real_sanity_v2_candidates/blur_gt_reference"
INDEX = ROOT / "manifests/qwen_multidegradation_realblur_pair_screening_v2.json"
SHEET = ROOT / "reports/qwen_multidegradation_real_sanity_v2_candidate_sheets/blur_pair_screening.jpg"
ROWS = (0, 25, 45, 65, 85)


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "qwen-realblur-pair-screen-v2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sharpness(image: Image.Image) -> float:
    array = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    laplace = (
        -4.0 * array[1:-1, 1:-1]
        + array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
    )
    return float(np.var(laplace))


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def tile(image: Image.Image, size: tuple[int, int], *, native_crop: bool = False) -> Image.Image:
    if native_crop:
        edge = min(300, image.width, image.height)
        left, top = (image.width - edge) // 2, (image.height - edge) // 2
        image = image.crop((left, top, left + edge, top + edge))
    else:
        image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    output = Image.new("RGB", size, "#dddddd")
    output.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return output


def main() -> None:
    response = json.loads(request_bytes(ENDPOINT))
    rows = {int(item["row_idx"]): item["row"] for item in response["rows"]}
    candidates = json.loads(
        (ROOT / "manifests/qwen_multidegradation_real_sanity_v2_candidates.json").read_text()
    )["records"]
    by_id = {record["candidate_id"]: record for record in candidates}
    records = []
    for row_index in ROWS:
        candidate_id = f"realblurj_row_{row_index:03d}"
        candidate = by_id[candidate_id]
        before_path = ROOT / candidate["path"]
        after_url = rows[row_index]["after"]["src"]
        if REVISION not in after_url:
            raise RuntimeError("RealBlur-J reference revision is not pinned")
        after_payload = request_bytes(after_url)
        after_path = POOL / f"{candidate_id}_after.jpg"
        after_digest = write_new_or_verify(after_path, after_payload)
        with Image.open(before_path) as handle:
            before = ImageOps.exif_transpose(handle.convert("RGB"))
        with Image.open(io.BytesIO(after_payload)) as handle:
            after = ImageOps.exif_transpose(handle.convert("RGB"))
        before_score, after_score = sharpness(before), sharpness(after)
        records.append(
            {
                "candidate_id": candidate_id,
                "before_path": candidate["path"],
                "before_sha256": candidate["sha256"],
                "after_reference_path": after_path.relative_to(ROOT).as_posix(),
                "after_reference_sha256": after_digest,
                "before_laplacian_variance": before_score,
                "after_laplacian_variance": after_score,
                "after_to_before_sharpness_ratio": after_score / max(before_score, 1e-12),
                "dataset": "RealBlur-J via seiya12 mirror",
                "repository": "seiya12/instruct-pix2pix-dataset_RealblurJ",
                "revision": REVISION,
                "license": "mirror_card_unspecified_internal_research_only",
                "source_key": f"row_{row_index:03d}",
                "after_reference_url": after_url,
            }
        )
    if len(records) != 5:
        raise RuntimeError("RealBlur-J pair-screen record count changed")
    payload = {
        "schema_version": 1,
        "status": "preinference_paired_input_screening_evidence",
        "foundir_used": False,
        "qwen_inference_performed": False,
        "records": records,
    }
    digest = write_new_or_verify(
        INDEX, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    canvas = Image.new("RGB", (1480, 80 + 340 * len(records)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), "RealBlur-J paired screening: before | reference | native crops", fill="black", font=font(25))
    for index, record in enumerate(records):
        top = 80 + index * 340
        with Image.open(ROOT / record["before_path"]) as handle:
            before = ImageOps.exif_transpose(handle.convert("RGB"))
        with Image.open(ROOT / record["after_reference_path"]) as handle:
            after = ImageOps.exif_transpose(handle.convert("RGB"))
        canvas.paste(tile(before, (350, 285)), (0, top + 40))
        canvas.paste(tile(after, (350, 285)), (365, top + 40))
        canvas.paste(tile(before, (350, 285), native_crop=True), (730, top + 40))
        canvas.paste(tile(after, (350, 285), native_crop=True), (1095, top + 40))
        draw.text(
            (8, top + 5),
            f"{record['candidate_id']} after/before sharpness={record['after_to_before_sharpness_ratio']:.3f}",
            fill="black",
            font=font(19),
        )
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=94, subsampling=0)
    write_new_or_verify(SHEET, buffer.getvalue())
    print(f"QWEN_REALBLUR_PAIR_SCREEN_PASS records={len(records)} sha256={digest}")


if __name__ == "__main__":
    main()
