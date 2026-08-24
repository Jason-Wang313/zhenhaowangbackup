#!/usr/bin/env python3
"""Freeze real-sanity v3 with every SIDD noise file from official bytes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify
from smoke.finalize_qwen_multidegradation_real_sanity_v2 import validate_records

V2 = ROOT / "manifests/qwen_multidegradation_real_sanity_v2.json"
CORRECTION = ROOT / "manifests/qwen_multidegradation_sidd_official_correction_v3.json"
OUTPUT = ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json"


def main() -> None:
    previous = json.loads(V2.read_text())
    correction = json.loads(CORRECTION.read_text())
    if (
        previous.get("status") != "frozen_before_qwen_inference"
        or correction.get("status") != "official_member_frozen_before_qwen_inference"
        or correction.get("source_real_sanity_manifest_sha256") != sha256(V2)
        or correction.get("qwen_inference_before_freeze") is not False
        or correction.get("foundir_used") is not False
    ):
        raise RuntimeError("SIDD correction provenance is not pre-inference and immutable")
    records = [dict(record) for record in previous["records"]]
    target = str(correction["supersedes_mirror_record_id"])
    matches = [record for record in records if record["record_id"] == target]
    if len(matches) != 1 or matches[0]["sha256"] != correction["mirror_sha256"]:
        raise RuntimeError("SIDD correction does not match frozen v2")
    record = matches[0]
    record.update(
        path=correction["official_path"],
        sha256=correction["official_sha256"],
        acquisition="selective_member_from_official_SIDD_Full_archive",
        acquisition_mirror=None,
        acquisition_url=correction["official_archive_url"],
        mirror_mismatch_evidence={
            "path": correction["mirror_path"],
            "sha256": correction["mirror_sha256"],
            "decoded_rgb_pixel_exact_match": correction[
                "decoded_rgb_pixel_exact_match_with_mirror"
            ],
        },
    )
    validation = validate_records(records)
    payload = {
        **previous,
        "schema_version": 3,
        "status": "frozen_before_qwen_inference_official_sidd_bytes",
        "supersedes": V2.relative_to(ROOT).as_posix(),
        "supersedes_sha256": sha256(V2),
        "sidd_official_correction": CORRECTION.relative_to(ROOT).as_posix(),
        "sidd_official_correction_sha256": sha256(CORRECTION),
        "validation": validation,
        "records": records,
    }
    digest = write_new_or_verify(
        OUTPUT, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    )
    print(f"QWEN_MULTIDEGRADATION_REAL_SANITY_V3_FREEZE_PASS records=25 sha256={digest}")


if __name__ == "__main__":
    main()
