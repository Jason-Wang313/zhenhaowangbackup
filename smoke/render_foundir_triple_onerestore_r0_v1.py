#!/usr/bin/env python3
"""Render or verify the frozen R0 ordered-subset cache on CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import generate_record  # noqa: E402


CONFIG_PATH = ROOT / "configs/foundir_triple_onerestore_r0_v1.json"
EXPECTED_CONFIG_SHA = "415ffaa126393e0125ca070830596c413645919836e290b7a16970b2f27c2c64"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_line(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_or_verify(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite differing cache artifact: {path}")
        return
    path.write_bytes(payload)


def render() -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("frozen R0 config hash mismatch")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["parent_protocol"]["manifest"]
    if sha256_file(manifest_path) != config["parent_protocol"]["manifest_sha256"]:
        raise RuntimeError("frozen lattice manifest hash mismatch")
    cache_root = ROOT / config["data"]["cache_root"]
    rows = []
    for lattice in load_jsonl(manifest_path):
        parent_id = str(lattice["parent_record_id"])
        for node in lattice["nodes"]:
            node_id = str(node["node_id"])
            order = tuple(map(str, node["application_order"]))
            image, encoded_sha = generate_record(lattice, application_order=order)
            success, encoded = cv2.imencode(".png", image)
            if not success:
                raise RuntimeError(f"PNG encode failed: {parent_id}/{node_id}")
            payload = encoded.tobytes()
            if hashlib.sha256(payload).hexdigest() != encoded_sha:
                raise RuntimeError("deterministic PNG hash mismatch")
            relative = Path(str(lattice["protocol_role"])) / parent_id / f"{node_id}.png"
            write_or_verify(cache_root / relative, payload)
            rows.append(
                {
                    "parent_record_id": parent_id,
                    "protocol_role": lattice["protocol_role"],
                    "source_id": lattice["source_id"],
                    "composition_size": lattice["composition_size"],
                    "parent_degradations": lattice["degradations"],
                    "node_id": node_id,
                    "application_order": list(order),
                    "path": (Path(config["data"]["cache_root"]) / relative).as_posix(),
                    "sha256": encoded_sha,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                }
            )
    manifest_payload = b"".join(canonical_line(row) for row in rows)
    cache_manifest = cache_root / "manifest.jsonl"
    write_or_verify(cache_manifest, manifest_payload)
    metadata = {
        "schema_version": 1,
        "status": "foundir_triple_onerestore_r0_v1_cache_complete",
        "config_sha256": EXPECTED_CONFIG_SHA,
        "lattice_manifest_sha256": config["parent_protocol"]["manifest_sha256"],
        "records": len(rows),
        "roles": {
            role: sum(row["protocol_role"] == role for row in rows)
            for role in ("train", "development", "heldout_triple")
        },
        "cache_manifest": cache_manifest.relative_to(ROOT).as_posix(),
        "cache_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "official_foundir_test_accessed": False,
        "foundir_60_image_triple_accessed": False,
    }
    write_or_verify(cache_root / "metadata.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return metadata


def check() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cache_root = ROOT / config["data"]["cache_root"]
    metadata = json.loads((cache_root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["config_sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("cache/config binding mismatch")
    manifest_path = ROOT / metadata["cache_manifest"]
    if sha256_file(manifest_path) != metadata["cache_manifest_sha256"]:
        raise RuntimeError("cache manifest hash mismatch")
    rows = load_jsonl(manifest_path)
    if len(rows) != metadata["records"]:
        raise RuntimeError("cache record count mismatch")
    for row in rows:
        path = ROOT / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"cached PNG hash mismatch: {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check() if args.check else render()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
