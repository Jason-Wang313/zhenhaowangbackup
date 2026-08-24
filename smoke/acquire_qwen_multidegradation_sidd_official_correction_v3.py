#!/usr/bin/env python3
"""Freeze the one SIDD member whose public mirror bytes differ from official."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify
from smoke.verify_qwen_multidegradation_sidd_official_v2 import read_official_member

V2 = ROOT / "manifests/qwen_multidegradation_real_sanity_v2.json"
OUTPUT = ROOT / "manifests/qwen_multidegradation_sidd_official_correction_v3.json"
OFFICIAL_IMAGE = (
    ROOT
    / "data/qwen_multidegradation_real_sanity_v2_candidates/noise_official/sidd_0076.png"
)
RECORD_ID = "real_v2_noise_02"


def main() -> None:
    source = json.loads(V2.read_text())
    matches = [record for record in source["records"] if record["record_id"] == RECORD_ID]
    if len(matches) != 1:
        raise RuntimeError("the frozen v2 SIDD correction target changed")
    record = matches[0]
    archive_size, member, official_payload = read_official_member(record)
    official_hash = hashlib.sha256(official_payload).hexdigest()
    if official_hash == record["sha256"]:
        raise RuntimeError("the declared correction is no longer a byte mismatch")
    with Image.open(io.BytesIO(official_payload)) as official_handle:
        official = official_handle.convert("RGB")
        official_format = official_handle.format
    with Image.open(ROOT / record["path"]) as mirror_handle:
        mirror = mirror_handle.convert("RGB")
    if official_format != "PNG" or official.size != mirror.size:
        raise RuntimeError("official correction differs in format or dimensions")
    pixel_exact = ImageChops.difference(official, mirror).getbbox() is None
    digest = write_new_or_verify(OFFICIAL_IMAGE, official_payload)
    if digest != official_hash:
        raise RuntimeError("official correction write hash changed")
    payload = {
        "schema_version": 1,
        "status": "official_member_frozen_before_qwen_inference",
        "foundir_used": False,
        "qwen_inference_before_freeze": False,
        "supersedes_mirror_record_id": RECORD_ID,
        "source_real_sanity_manifest": V2.relative_to(ROOT).as_posix(),
        "source_real_sanity_manifest_sha256": sha256(V2),
        "mirror_path": record["path"],
        "mirror_sha256": record["sha256"],
        "official_path": OFFICIAL_IMAGE.relative_to(ROOT).as_posix(),
        "official_sha256": official_hash,
        "official_member": member,
        "official_member_bytes": len(official_payload),
        "official_archive_url": record["source_url"],
        "official_archive_content_length": archive_size,
        "decoded_rgb_pixel_exact_match_with_mirror": pixel_exact,
    }
    manifest_hash = write_new_or_verify(
        OUTPUT, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(
        "QWEN_SIDD_OFFICIAL_CORRECTION_V3_PASS "
        f"sha256={manifest_hash} image_sha256={official_hash} pixel_exact={pixel_exact}",
        flush=True,
    )


if __name__ == "__main__":
    main()
