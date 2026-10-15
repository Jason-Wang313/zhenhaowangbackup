#!/usr/bin/env python3
"""Low-latency retry transport plus correct split-volume ordering for SDSD."""

from __future__ import annotations

import random
import re
import time
from typing import Mapping

import requests

import smoke.build_sdsd_selective_development_manifest as implementation


VOLUME = re.compile(r"\.z(\d+)$")


def volumes_from_config(config: Mapping[str, object], subset: str) -> list[dict[str, object]]:
    files = config["files"]
    assert isinstance(files, list)
    selected = [
        dict(item) for item in files if str(item["name"]).startswith(f"{subset}_np")
    ]
    final_index = len(selected) - 1

    def disk_number(item: Mapping[str, object]) -> int:
        match = VOLUME.search(str(item["name"]))
        return int(match.group(1)) - 1 if match else final_index

    selected.sort(key=disk_number)
    return selected


def range_get(file_id: str, start: int, end: int, total: int) -> bytes:
    failure: Exception | None = None
    for _ in range(40):
        try:
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t&nonce={random.getrandbits(64):016x}"
            )
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=(20, 90),
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
            time.sleep(0.5)
    raise RuntimeError(f"fast range retries exhausted: {failure}")


implementation.volumes_from_config = volumes_from_config
implementation.range_get = range_get


if __name__ == "__main__":
    raise SystemExit(implementation.main())
