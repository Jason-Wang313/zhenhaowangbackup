#!/usr/bin/env python3
"""Verify mirrored SIDD noise inputs byte-for-byte against official archives."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.multidegradation_data import sha256, write_new_or_verify

MANIFEST = ROOT / "manifests/qwen_multidegradation_real_sanity_v3.json"
OUTPUT = ROOT / "manifests/qwen_multidegradation_sidd_official_verification_v3.json"
TIMEOUT = 180
RETRIES = 4
BLOCK_SIZE = 8 * 1024 * 1024


def request_range(url: str, start: int, stop: int) -> bytes:
    expected = stop - start + 1
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "qwen-sidd-official-byte-verification-v2",
                    "Range": f"bytes={start}-{stop}",
                },
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.status != 206:
                    raise RuntimeError(f"official SIDD server ignored byte range: {url}")
                payload = response.read()
            if len(payload) != expected:
                raise RuntimeError(
                    f"short official SIDD range response: expected={expected} actual={len(payload)}"
                )
            return payload
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


def remote_size(url: str) -> int:
    for attempt in range(1, RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                method="HEAD",
                headers={"User-Agent": "qwen-sidd-official-byte-verification-v2"},
            )
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                if response.headers.get("Accept-Ranges", "").lower() != "bytes":
                    raise RuntimeError("official SIDD server does not advertise byte ranges")
                return int(response.headers["Content-Length"])
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(attempt)
    raise AssertionError("unreachable")


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url: str, size: int, block_size: int = BLOCK_SIZE):
        self.url = url
        self.size = size
        self.block_size = block_size
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid seek whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self.position = min(position, self.size)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.size if size < 0 else min(self.size, self.position + size)
        pieces = []
        while self.position < end:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                stop = min(self.size, start + self.block_size) - 1
                self.cache[block] = request_range(self.url, start, stop)
            payload = self.cache[block]
            within = self.position - block * self.block_size
            take = min(end - self.position, len(payload) - within)
            if take <= 0:
                raise RuntimeError("short official SIDD range response")
            pieces.append(payload[within : within + take])
            self.position += take
        return b"".join(pieces)


def read_official_member(record: dict[str, object]) -> tuple[int, str, bytes]:
    url = str(record["source_url"])
    if "130.63.97.225/share/SIDD_Full/" not in url or not url.endswith("_NOISY_SRGB.zip"):
        raise RuntimeError("noise record lacks an official SIDD noisy-sRGB archive URL")
    size = remote_size(url)
    reader = RemoteZipReader(url, size)
    with zipfile.ZipFile(reader) as archive:
        matches = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.upper().endswith("NOISY_SRGB_010.PNG")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"official SIDD member selection is ambiguous: {url}")
        member = matches[0]
        official_payload = archive.read(member)
    return size, member, official_payload


def verify(record: dict[str, object]) -> dict[str, object]:
    url = str(record["source_url"])
    size, member, official_payload = read_official_member(record)
    official_hash = hashlib.sha256(official_payload).hexdigest()
    local_path = ROOT / str(record["path"])
    local_hash = sha256(local_path)
    result = {
        "record_id": record["record_id"],
        "local_path": record["path"],
        "local_sha256": local_hash,
        "official_archive_url": url,
        "official_archive_content_length": size,
        "official_archive_sha256": "not_computed_selective_member_verification",
        "official_member": member,
        "official_member_bytes": len(official_payload),
        "official_member_sha256": official_hash,
        "exact_match": official_hash == local_hash,
    }
    print(
        "QWEN_SIDD_OFFICIAL_MEMBER_CHECK "
        f"record_id={record['record_id']} exact_match={result['exact_match']} "
        f"bytes={len(official_payload)}",
        flush=True,
    )
    return result


def main() -> None:
    payload = json.loads(MANIFEST.read_text())
    records = [
        record for record in payload["records"] if record["expected_primary_type"] == "noise"
    ]
    if len(records) != 5:
        raise RuntimeError("frozen v2 noise record count changed")
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(verify, records))
    if not all(result["exact_match"] for result in results):
        raise RuntimeError("a mirrored SIDD file differs from its official archive member")
    verification = {
        "schema_version": 1,
        "status": "all_five_exact_official_member_matches",
        "real_sanity_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "real_sanity_manifest_sha256": sha256(MANIFEST),
        "foundir_used": False,
        "records": results,
    }
    digest = write_new_or_verify(
        OUTPUT, (json.dumps(verification, indent=2, sort_keys=True) + "\n").encode()
    )
    print(f"QWEN_SIDD_OFFICIAL_VERIFICATION_PASS records=5 sha256={digest}")


if __name__ == "__main__":
    main()
