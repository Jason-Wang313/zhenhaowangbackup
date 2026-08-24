#!/usr/bin/env python3
"""Select one clearly blurred RealBlur-J `after` image per distinct scene burst."""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
POOL = ROOT / "data/qwen_multidegradation_real_sanity_v2_candidates/blur_after"
BEFORE_POOL = ROOT / "data/qwen_multidegradation_real_sanity_v2_candidates/blur_before_reference"
INDEX = ROOT / "manifests/qwen_multidegradation_realblur_after_selection_v2b.json"
SHEET = ROOT / "reports/qwen_multidegradation_real_sanity_v2_candidate_sheets/blur_after_selected_v2b.jpg"
SCENE_BURSTS = ((0, 21), (21, 42), (42, 63), (63, 84), (84, 100))


def request_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "qwen-realblur-after-v2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def decode(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as handle:
        return ImageOps.exif_transpose(handle.convert("RGB"))


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


def tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    output = Image.new("RGB", size, "#dddddd")
    output.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return output


def existing_or_request(path: Path, url: str) -> bytes:
    if path.is_file():
        return path.read_bytes()
    return request_bytes(url)


def process_row(item: dict[str, object]) -> dict[str, object]:
    row_index = int(item["row_idx"])
    row = item["row"]
    prompt = str(row["prompt"])
    if "blur" not in prompt.lower():
        raise RuntimeError(f"RealBlur-J transform direction is not explicit: row {row_index}")
    before_url, after_url = row["before"]["src"], row["after"]["src"]
    if REVISION not in before_url or REVISION not in after_url:
        raise RuntimeError("RealBlur-J response is not pinned to the declared revision")
    before_path = BEFORE_POOL / f"realblurj_before_row_{row_index:03d}.jpg"
    after_path = POOL / f"realblurj_after_row_{row_index:03d}.jpg"
    before_payload = existing_or_request(before_path, before_url)
    after_payload = existing_or_request(after_path, after_url)
    before_digest = write_new_or_verify(before_path, before_payload)
    after_digest = write_new_or_verify(after_path, after_payload)
    before, after = decode(before_payload), decode(after_payload)
    before_score, after_score = sharpness(before), sharpness(after)
    return {
        "candidate_id": f"realblurj_after_row_{row_index:03d}",
        "row_index": row_index,
        "expected_primary_type": "blur",
        "path": after_path.relative_to(ROOT).as_posix(),
        "sha256": after_digest,
        "before_reference_path": before_path.relative_to(ROOT).as_posix(),
        "before_reference_sha256": before_digest,
        "width": after.width,
        "height": after.height,
        "format": "JPEG",
        "prompt_proving_transform_direction": prompt,
        "before_laplacian_variance": before_score,
        "after_laplacian_variance": after_score,
        "after_to_before_sharpness_ratio": after_score / max(before_score, 1e-12),
        "dataset": "RealBlur-J via seiya12 instruct-pix2pix mirror",
        "repository": "seiya12/instruct-pix2pix-dataset_RealblurJ",
        "revision": REVISION,
        "license": "mirror_card_unspecified_internal_research_only",
        "source_key": f"row_{row_index:03d}:after",
        "source_url": after_url,
    }


def main() -> None:
    response = json.loads(request_bytes(ENDPOINT))
    rows = response.get("rows")
    if not isinstance(rows, list) or len(rows) != 100:
        raise RuntimeError("RealBlur-J pinned first-rows response changed")
    with ThreadPoolExecutor(max_workers=16) as executor:
        candidates = list(executor.map(process_row, rows))
    selected = []
    for scene_index, (start, stop) in enumerate(SCENE_BURSTS):
        subset = [record for record in candidates if start <= record["row_index"] < stop]
        winner = min(subset, key=lambda record: record["after_to_before_sharpness_ratio"])
        winner = {**winner, "screening_scene_burst": scene_index}
        selected.append(winner)
    if any(record["after_to_before_sharpness_ratio"] >= 0.8 for record in selected):
        raise RuntimeError("one RealBlur-J scene burst lacks a clearly softer transformed image")
    payload = {
        "schema_version": 1,
        "status": "preinference_selected_blurred_after_images",
        "foundir_used": False,
        "qwen_inference_performed": False,
        "direction_correction": "before_is_sharp_input_after_is_blurred_output_as_proven_by_prompt",
        "supersedes": "qwen_multidegradation_realblur_after_selection_v2_wrong_visual_burst_boundaries",
        "selection": "minimum_after_to_before_laplacian_variance_ratio_within_each_of_five_visual_scene_bursts",
        "scene_bursts": [list(bounds) for bounds in SCENE_BURSTS],
        "selected_records": selected,
        "candidate_records": candidates,
    }
    digest = write_new_or_verify(
        INDEX, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    canvas = Image.new("RGB", (1500, 70 + 340 * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 14), "Selected blurred AFTER image from each distinct scene burst", fill="black", font=font(26))
    for index, record in enumerate(selected):
        top = 70 + index * 340
        before = decode((ROOT / record["before_reference_path"]).read_bytes())
        after = decode((ROOT / record["path"]).read_bytes())
        canvas.paste(tile(before, (700, 285)), (0, top + 45))
        canvas.paste(tile(after, (700, 285)), (750, top + 45))
        draw.text(
            (8, top + 8),
            f"scene={record['screening_scene_burst']} row={record['row_index']} sharp | blurred ratio={record['after_to_before_sharpness_ratio']:.3f}",
            fill="black",
            font=font(19),
        )
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=94, subsampling=0)
    write_new_or_verify(SHEET, buffer.getvalue())
    print(f"QWEN_REALBLUR_AFTER_SELECTION_PASS selected=5 candidates=100 sha256={digest}")


if __name__ == "__main__":
    main()
