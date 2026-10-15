#!/usr/bin/env python3
"""Parallel, immutable segment acquisition for quota-limited SDSD archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import threading
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
from smoke.acquire_sdsd_development_archives_direct import parse_content_range


DEFAULT_CONFIG = ROOT / "configs/sdsd_development_acquisition_v1c.json"
SEGMENT_BYTES = 64 * 1024 * 1024
PARALLEL_REQUESTS = 16
_progress_lock = threading.Lock()
_completed_segments = 0


def segment_jobs(files: list[Mapping[str, object]]) -> list[tuple[Mapping[str, object], int, int, int]]:
    jobs: list[tuple[Mapping[str, object], int, int, int]] = []
    for item in files:
        total = int(item["expected_size"])
        index = 0
        for start in range(0, total, SEGMENT_BYTES):
            end = min(start + SEGMENT_BYTES, total) - 1
            jobs.append((item, index, start, end))
            index += 1
    return jobs


def download_segment(
    job: tuple[Mapping[str, object], int, int, int], archive_directory: Path, total_jobs: int
) -> tuple[str, int, Path]:
    global _completed_segments
    item, index, start, end = job
    name = str(item["name"])
    expected_size = end - start + 1
    segment_directory = archive_directory / f"{name}.segments"
    segment_directory.mkdir(parents=True, exist_ok=True)
    segment = segment_directory / f"{index:05d}_{start}_{end}.part"
    if segment.exists():
        if not segment.is_file() or segment.stat().st_size != expected_size:
            raise RuntimeError(f"existing segment has wrong identity: {segment}")
    else:
        url = (
            "https://drive.usercontent.google.com/download"
            f"?id={item['google_drive_id']}&export=download&confirm=t"
        )
        failure: Exception | None = None
        for attempt in range(1, 11):
            try:
                response = requests.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 180),
                )
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code}")
                actual_start, actual_end, actual_total = parse_content_range(
                    response.headers.get("Content-Range", "")
                )
                if (
                    actual_start != start
                    or actual_end != end
                    or actual_total != int(item["expected_size"])
                    or response.headers.get("Content-Type") != "application/octet-stream"
                    or len(response.content) != expected_size
                ):
                    raise RuntimeError("segment response identity mismatch")
                with segment.open("xb") as handle:
                    handle.write(response.content)
                    handle.flush()
                    os.fsync(handle.fileno())
                failure = None
                break
            except (requests.RequestException, RuntimeError) as exc:
                failure = exc
                time.sleep(min(2**attempt, 30))
        if failure is not None:
            raise RuntimeError(f"segment failed {name}[{index}]: {failure}") from failure
    with _progress_lock:
        _completed_segments += 1
        if _completed_segments % 25 == 0 or _completed_segments == total_jobs:
            print(
                f"SDSD_SEGMENT_PROGRESS completed={_completed_segments}/{total_jobs}",
                flush=True,
            )
    return name, index, segment


def assemble(
    item: Mapping[str, object], indexed_segments: list[tuple[int, Path]], archive_directory: Path
) -> dict[str, object]:
    name = str(item["name"])
    expected_size = int(item["expected_size"])
    target = archive_directory / name
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError(f"existing final archive has wrong identity: {target}")
    else:
        attempt = archive_directory / f"{name}.assembled.{os.getpid()}.partial"
        with attempt.open("xb") as output:
            for _, segment in sorted(indexed_segments):
                with segment.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if attempt.stat().st_size != expected_size:
            raise RuntimeError(f"assembled archive size mismatch: {attempt}")
        try:
            os.link(attempt, target)
        except FileExistsError:
            pass
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
    jobs = segment_jobs(files)
    by_name: dict[str, list[tuple[int, Path]]] = {str(item["name"]): [] for item in files}
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_REQUESTS) as executor:
        futures = [executor.submit(download_segment, job, archive_directory, len(jobs)) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            name, index, segment = future.result()
            by_name[name].append((index, segment))
    records = [assemble(item, by_name[str(item["name"])], archive_directory) for item in files]
    records.sort(key=lambda row: str(row["name"]))
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "automatic_gate_pass": True,
        "config_sha256": sha256(config_path),
        "downloader_sha256": sha256(Path(__file__)),
        "transport": "google_drive_parallel_immutable_content_range_segments",
        "segment_bytes": SEGMENT_BYTES,
        "parallel_requests": PARALLEL_REQUESTS,
        "segment_count": len(jobs),
        "official_repository_revision": config["official_repository"]["revision"],
        "records": records,
        "total_bytes": sum(int(row["size"]) for row in records),
        "protected_foundir_pixels_used": False,
        "redistribution_authorized": False,
    }
    output = workspace_path(str(config["download_result"]))
    write_once_or_verify(output, result)
    print(f"SDSD_DEVELOPMENT_DOWNLOAD_PASS files={len(records)}", flush=True)
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
