#!/usr/bin/env python3
"""Range-resumable SDSD acquisition through Google's direct content endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Mapping

import requests

from smoke.acquire_sdsd_development_archives import (
    ROOT,
    sha256,
    validate_config,
    workspace_path,
    write_once_or_verify,
)


DEFAULT_CONFIG = ROOT / "configs/sdsd_development_acquisition_v1c.json"
CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def parse_content_range(value: str) -> tuple[int, int, int]:
    match = CONTENT_RANGE.fullmatch(value)
    if match is None:
        raise RuntimeError(f"invalid Content-Range: {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def download_one(item: Mapping[str, object], archive_directory: Path) -> dict[str, object]:
    name = str(item["name"])
    expected_size = int(item["expected_size"])
    target = archive_directory / name
    partial = archive_directory / f"{name}.partial"
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError(f"existing final archive has wrong identity: {target}")
    else:
        for attempt in range(1, 7):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected_size:
                raise RuntimeError(f"partial archive exceeds expected size: {partial}")
            if offset == expected_size:
                break
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={item['google_drive_id']}&export=download&confirm=t"
            )
            try:
                with requests.get(
                    url,
                    headers={"Range": f"bytes={offset}-"},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"direct endpoint returned HTTP {response.status_code} for {name}"
                        )
                    start, end, total = parse_content_range(
                        response.headers.get("Content-Range", "")
                    )
                    if start != offset or total != expected_size or end < start:
                        raise RuntimeError(
                            f"range identity mismatch for {name}: {start}-{end}/{total}"
                        )
                    with partial.open("ab") as handle:
                        for block in response.iter_content(8 * 1024 * 1024):
                            if block:
                                handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == 6:
                    raise RuntimeError(f"direct download failed for {name}: {exc}") from exc
                time.sleep(min(2**attempt, 30))
                continue
        actual = partial.stat().st_size if partial.exists() else None
        if actual != expected_size:
            raise RuntimeError(
                f"downloaded archive size mismatch for {name}: {actual} != {expected_size}"
            )
        try:
            os.link(partial, target)
        except FileExistsError:
            pass
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError(f"failed to publish verified archive: {target}")
    digest = sha256(target)
    print(f"SDSD_ARCHIVE_READY name={name} size={expected_size} sha256={digest}", flush=True)
    return {
        "name": name,
        "google_drive_id": item["google_drive_id"],
        "size": expected_size,
        "sha256": digest,
        "path": str(target.relative_to(ROOT)),
    }


def run(config_path: Path) -> dict[str, object]:
    config = validate_config(config_path)
    archive_directory = workspace_path(str(config["archive_directory"]))
    archive_directory.mkdir(parents=True, exist_ok=True)
    files = config["files"]
    assert isinstance(files, list)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["max_parallel_downloads"])
    ) as executor:
        futures = [executor.submit(download_one, item, archive_directory) for item in files]
        records = [future.result() for future in futures]
    records.sort(key=lambda row: str(row["name"]))
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "automatic_gate_pass": True,
        "config_sha256": sha256(config_path),
        "downloader_sha256": sha256(Path(__file__)),
        "transport": "google_drive_direct_content_range",
        "official_repository_revision": config["official_repository"]["revision"],
        "records": records,
        "total_bytes": sum(int(row["size"]) for row in records),
        "protected_foundir_pixels_used": False,
        "redistribution_authorized": False,
    }
    output = workspace_path(str(config["download_result"]))
    write_once_or_verify(output, result)
    print(
        f"SDSD_DEVELOPMENT_DOWNLOAD_PASS files={len(records)} bytes={result['total_bytes']}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    try:
        run(arguments.config.resolve())
    except Exception as exc:
        print(f"SDSD_DEVELOPMENT_DOWNLOAD_FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
