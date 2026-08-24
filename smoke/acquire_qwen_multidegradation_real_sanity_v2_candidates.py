#!/usr/bin/env python3
"""Acquire broad, pinned real-image candidates for pre-inference v2 screening.

This script deliberately does not freeze the final 25-image manifest.  Its
outputs are screening candidates only; a separate finalizer records the five
visually accepted inputs per degradation before any Qwen inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import write_new_or_verify


POOL = ROOT / "data/qwen_multidegradation_real_sanity_v2_candidates"
INDEX = ROOT / "manifests/qwen_multidegradation_real_sanity_v2_candidates.json"
TIMEOUT = 120

REALBLUR_REVISION = "b66421985334b570984036e431006bd13131b63e"
REALBLUR_ROWS = tuple(range(0, 100, 5))
RTTS_REVISION = "6f5aaf4031975a6016157f144e5b373a8ba6f392"
# Thirty-seven spread candidates are ample for selecting five; tree index 740
# repeatedly stalls at the pinned mirror and is excluded before model inference.
RTTS_PATHS = (
    "VOCdevkit/VOC2007/JPEGImages/AM_Bing_211.png",
    "VOCdevkit/VOC2007/JPEGImages/AM_Google_246.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Baidu_023.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Baidu_095.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Baidu_199.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Baidu_299.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Baidu_476.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Bing_302.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Bing_664.png",
    "VOCdevkit/VOC2007/JPEGImages/BD_Google_372.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Baidu_069.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Baidu_1170.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Baidu_181.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Baidu_320.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Baidu_535.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Baidu_748.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Bing_059.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Bing_174.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Bing_385.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Bing_575.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Bing_759.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Google_178.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Google_326.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Google_447.png",
    "VOCdevkit/VOC2007/JPEGImages/BJ_Google_579.png",
    "VOCdevkit/VOC2007/JPEGImages/BL_Baidu_134.png",
    "VOCdevkit/VOC2007/JPEGImages/BL_Google_047.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Baidu_040.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Baidu_146.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Baidu_236.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Bing_112.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Bing_390.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Bing_668.png",
    "VOCdevkit/VOC2007/JPEGImages/CC_Google_089.png",
    "VOCdevkit/VOC2007/JPEGImages/CD_Baidu_225.png",
    "VOCdevkit/VOC2007/JPEGImages/CD_Bing_144.png",
    "VOCdevkit/VOC2007/JPEGImages/CD_Bing_662.png",
)
SIDD_SCENES = (
    (
        "0013_001_S6_03200_01250_3200_L",
        "http://130.63.97.225/share/SIDD_Full/0013_001_S6_03200_01250_3200_L/0013_NOISY_SRGB.zip",
    ),
    (
        "0070_003_IP_02000_04000_3200_N",
        "http://130.63.97.225/share/SIDD_Full/0070_003_IP_02000_04000_3200_N/0070_NOISY_SRGB.zip",
    ),
    (
        "0076_004_N6_03200_00320_3200_L",
        "http://130.63.97.225/share/SIDD_Full/0076_004_N6_03200_00320_3200_L/0076_NOISY_SRGB.zip",
    ),
    (
        "0104_005_S6_03200_01600_4400_L",
        "http://130.63.97.225/share/SIDD_Full/0104_005_S6_03200_01600_4400_L/0104_NOISY_SRGB.zip",
    ),
    (
        "0127_006_S6_01600_00800_4400_L",
        "http://130.63.97.225/share/SIDD_Full/0127_006_S6_01600_00800_4400_L/0127_NOISY_SRGB.zip",
    ),
)


def request_bytes(url: str, *, start: int | None = None, end: int | None = None) -> bytes:
    headers = {"User-Agent": "qwen-real-sanity-v2-preinference-screening"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        payload = response.read()
        if start is not None and response.status != 206:
            raise RuntimeError(f"server ignored required byte range for {url}")
        return payload


def request_json(url: str) -> object:
    return json.loads(request_bytes(url))


def request_kaggle_dataset_file(file_name: str) -> bytes:
    """Download one public SIDD-Small file from a pinned Kaggle mirror."""
    body = json.dumps(
        {
            "ownerSlug": "rajat95gupta",
            "datasetSlug": "smartphone-image-denoising-dataset",
            "datasetVersionNumber": 1,
            "fileName": file_name,
        }
    ).encode()
    request = urllib.request.Request(
        "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "kaggle-api/v1.7.0 qwen-real-sanity-v2-screening",
        },
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        archive_payload = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise RuntimeError(f"Kaggle single-file response changed for {file_name}")
        return archive.read(members[0])


def validate_image(payload: bytes) -> tuple[int, int, str]:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if image.width < 224 or image.height < 224:
            raise RuntimeError(f"candidate image is too small: {image.size}")
        return image.width, image.height, image.format or "unknown"


def save_candidate(
    category: str,
    candidate_id: str,
    payload: bytes,
    provenance: dict[str, object],
) -> dict[str, object]:
    width, height, image_format = validate_image(payload)
    suffix = ".png" if image_format.upper() == "PNG" else ".jpg"
    path = POOL / category / f"{candidate_id}{suffix}"
    digest = write_new_or_verify(path, payload)
    return {
        "candidate_id": candidate_id,
        "expected_primary_type": category,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "width": width,
        "height": height,
        "format": image_format,
        **provenance,
    }


def existing_or_fetch(category: str, candidate_id: str, fetch: object) -> bytes:
    existing = sorted((POOL / category).glob(f"{candidate_id}.*"))
    if len(existing) > 1:
        raise RuntimeError(f"ambiguous existing candidate: {candidate_id}")
    if existing:
        return existing[0].read_bytes()
    if not callable(fetch):
        raise TypeError("candidate fetch must be callable")
    return fetch()


def acquire_realblur() -> list[dict[str, object]]:
    endpoint = (
        "https://datasets-server.huggingface.co/first-rows?dataset="
        "seiya12/instruct-pix2pix-dataset_RealblurJ&config=default&split=train"
    )
    response = request_json(endpoint)
    if not isinstance(response, dict) or not isinstance(response.get("rows"), list):
        raise RuntimeError("RealBlur-J first-rows response changed")
    by_index = {int(item["row_idx"]): item["row"] for item in response["rows"]}
    records = []
    for row_index in REALBLUR_ROWS:
        cell = by_index[row_index]["before"]
        url = cell["src"]
        if REALBLUR_REVISION not in url:
            raise RuntimeError("RealBlur-J revision is not pinned")
        records.append(
            save_candidate(
                "blur",
                f"realblurj_row_{row_index:03d}",
                existing_or_fetch("blur", f"realblurj_row_{row_index:03d}", lambda: request_bytes(url)),
                {
                    "dataset": "RealBlur-J via seiya12 mirror",
                    "repository": "seiya12/instruct-pix2pix-dataset_RealblurJ",
                    "revision": REALBLUR_REVISION,
                    "license": "mirror_card_unspecified_internal_research_only",
                    "source_key": f"row_{row_index:03d}",
                    "source_url": url,
                },
            )
        )
    return records


def acquire_rtts() -> list[dict[str, object]]:
    records = []
    for candidate_index, source_path in enumerate(RTTS_PATHS):
        tree_index = candidate_index * 20
        url = (
            "https://huggingface.co/datasets/fuhao7i/RTTS/resolve/"
            f"{RTTS_REVISION}/{source_path}"
        )
        records.append(
            save_candidate(
                "haze",
                f"rtts_tree_{tree_index:04d}",
                existing_or_fetch(
                    "haze", f"rtts_tree_{tree_index:04d}", lambda: request_bytes(url)
                ),
                {
                    "dataset": "RESIDE RTTS",
                    "repository": "fuhao7i/RTTS",
                    "revision": RTTS_REVISION,
                    "license": "apache-2.0_mirror_metadata",
                    "source_key": source_path,
                    "source_url": url,
                },
            )
        )
    return records


def acquire_sidd() -> list[dict[str, object]]:
    records = []
    for scene, canonical_url in SIDD_SCENES:
        member = f"SIDD_Small_sRGB_Only/Data/{scene}/NOISY_SRGB_010.PNG"
        candidate_id = f"sidd_{scene.split('_', 1)[0]}"
        payload = existing_or_fetch(
            "noise", candidate_id, lambda: request_kaggle_dataset_file(member)
        )
        records.append(
            save_candidate(
                "noise",
                candidate_id,
                payload,
                {
                    "dataset": "SIDD Full real noisy sRGB",
                    "repository": "official_SIDD_scene_archive",
                    "revision": scene,
                    "license": "official_research_use_only_citation_required",
                    "source_key": member,
                    "source_url": canonical_url,
                    "acquisition_url": "https://api.kaggle.com/v1/datasets.DatasetApiService/DownloadDataset",
                    "acquisition_mirror": "rajat95gupta/smartphone-image-denoising-dataset/version/1",
                    "acquisition": "single_file_from_public_SIDD_Small_sRGB_mirror",
                    "disclosure_url": "https://abdokamel.github.io/sidd/",
                },
            )
        )
    return records


def index_local_dicm() -> list[dict[str, object]]:
    source_root = (
        ROOT
        / "data/real_lowlight_corrective_holdout/official_test_0033eb74d4c/Test/DICM"
    )
    records = []
    for path in sorted(source_root.iterdir(), key=lambda item: item.name.lower()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        payload = path.read_bytes()
        width, height, image_format = validate_image(payload)
        records.append(
            {
                "candidate_id": f"dicm_{path.stem.lower()}",
                "expected_primary_type": "low_light",
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "width": width,
                "height": height,
                "format": image_format,
                "dataset": "DICM author testing images",
                "repository": "local_verified_official_testing_archive",
                "revision": "archive_sha256_0033eb74d4c964755f02d4b46e6db3eb457637c116415571fa2d0212577b3f29",
                "license": "author_archive_terms_unspecified_internal_research_only",
                "source_key": path.name,
                "source_url": None,
            }
        )
    if len(records) != 44:
        raise RuntimeError(f"DICM candidate count changed: {len(records)}")
    return records


def main() -> None:
    records = acquire_realblur() + acquire_sidd() + acquire_rtts() + index_local_dicm()
    expected = {"blur": 20, "noise": 5, "haze": 37, "low_light": 44}
    counts = {
        name: sum(record["expected_primary_type"] == name for record in records)
        for name in expected
    }
    if counts != expected:
        raise RuntimeError(f"candidate counts changed: {counts}")
    if len({record["sha256"] for record in records}) != len(records):
        raise RuntimeError("candidate set contains exact duplicate files")
    payload = {
        "schema_version": 1,
        "status": "preinference_screening_candidates_not_final_evaluation_pool",
        "foundir_used": False,
        "qwen_inference_performed": False,
        "selection_rule": "visually_screen_for_obvious_primary_degradation_and_scene_diversity_before_freezing_v2",
        "counts": counts,
        "records": records,
    }
    digest = write_new_or_verify(
        INDEX, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(
        f"QWEN_REAL_SANITY_V2_CANDIDATES_PASS records={len(records)} sha256={digest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
