#!/usr/bin/env python3
"""Quota-compatible 64 KiB range transport for selective SDSD extraction."""

from __future__ import annotations

import random
import time

import requests

import smoke.build_sdsd_selective_development_manifest as implementation
from smoke.run_sdsd_selective_v3 import volumes_from_config


MAX_RANGE_BYTES = 64 * 1024


def small_range(file_id: str, start: int, end: int, total: int) -> bytes:
    failure: Exception | None = None
    for _ in range(20):
        try:
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t&nonce={random.getrandbits(64):016x}"
            )
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(20, 60),
            )
            if response.status_code != 206:
                raise RuntimeError(f"HTTP {response.status_code}")
            actual = implementation.parse_content_range(
                response.headers.get("Content-Range", "")
            )
            if (
                actual != (start, end, total)
                or response.headers.get("Content-Type") != "application/octet-stream"
                or len(response.content) != end - start + 1
            ):
                raise RuntimeError("range response identity mismatch")
            return response.content
        except (requests.RequestException, RuntimeError) as exc:
            failure = exc
            time.sleep(0.25)
    raise RuntimeError(f"small range retries exhausted: {failure}")


def range_get(file_id: str, start: int, end: int, total: int) -> bytes:
    pieces = []
    cursor = start
    while cursor <= end:
        piece_end = min(cursor + MAX_RANGE_BYTES, end + 1) - 1
        pieces.append(small_range(file_id, cursor, piece_end, total))
        cursor = piece_end + 1
    return b"".join(pieces)


implementation.volumes_from_config = volumes_from_config
implementation.range_get = range_get


if __name__ == "__main__":
    raise SystemExit(implementation.main())
