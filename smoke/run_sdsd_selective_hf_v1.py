#!/usr/bin/env python3
"""Commit-pinned Hugging Face range transport for selective SDSD extraction."""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Mapping
from urllib.parse import quote

import requests

import smoke.build_sdsd_selective_development_manifest as implementation


CONFIG = implementation.ROOT / "configs/sdsd_development_acquisition_v1d.json"
VOLUME = re.compile(r"\.z(\d+)$")
_identity_lock = threading.Lock()
_verified_locators: set[str] = set()


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
    mirror = config["transport_mirror"]
    assert isinstance(mirror, dict)
    repository = quote(str(mirror["repository"]), safe="/")
    revision = str(mirror["revision"])
    for item in selected:
        item["google_drive_id"] = "|".join(
            (
                repository,
                revision,
                quote(str(item["huggingface_path"]), safe="/"),
                str(item["huggingface_lfs_sha256"]),
            )
        )
    return selected


def _resolve_url(locator: str) -> tuple[str, str, str]:
    repository, revision, path, expected_sha256 = locator.split("|", 3)
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}",
        revision,
        expected_sha256,
    )


def _verify_remote_identity(locator: str, total: int) -> None:
    with _identity_lock:
        if locator in _verified_locators:
            return
        url, revision, expected_sha256 = _resolve_url(locator)
        response = requests.head(url, allow_redirects=False, timeout=(20, 60))
        if response.status_code != 302:
            raise RuntimeError(f"Hugging Face identity HEAD returned HTTP {response.status_code}")
        actual_revision = response.headers.get("X-Repo-Commit")
        actual_size = response.headers.get("X-Linked-Size")
        actual_sha256 = response.headers.get("X-Linked-ETag", "").strip('"')
        if (
            actual_revision != revision
            or actual_size != str(total)
            or actual_sha256 != expected_sha256
        ):
            raise RuntimeError("Hugging Face LFS identity mismatch")
        _verified_locators.add(locator)


def range_get(locator: str, start: int, end: int, total: int) -> bytes:
    _verify_remote_identity(locator, total)
    url, _, _ = _resolve_url(locator)
    failure: Exception | None = None
    for attempt in range(12):
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                allow_redirects=True,
                timeout=(20, 180),
            )
            if response.status_code != 206:
                raise RuntimeError(f"HTTP {response.status_code}")
            actual = implementation.parse_content_range(
                response.headers.get("Content-Range", "")
            )
            if actual != (start, end, total) or len(response.content) != end - start + 1:
                raise RuntimeError("Hugging Face range response identity mismatch")
            return response.content
        except (requests.RequestException, RuntimeError) as exc:
            failure = exc
            time.sleep(min(0.25 * (2**attempt), 10.0))
    raise RuntimeError(f"Hugging Face range retries exhausted: {failure}")


implementation.DEFAULT_CONFIG = CONFIG
implementation.volumes_from_config = volumes_from_config
implementation.range_get = range_get


if __name__ == "__main__":
    raise SystemExit(implementation.main())
