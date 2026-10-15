#!/usr/bin/env python3
"""Resume the two public CCDD-11 archives by verified HTTP byte ranges.

The existing curl prefix is never modified. Remaining ranges are retained as
separate files and assembled only after every range is complete; the final ZIP
replaces the prefix only after its published SHA-256 matches exactly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/third_party/ccdd11"
ARCHIVES = {
    "half_train": (74063577783, "fd957ac7690060652d9d24c6f03292bad799b6e9e44e05df9dc3183aaac3b268"),
    "half_test": (16392141571, "2ed9fd82d4cb5b05615b7258b06c840d60b60abf69cd5e49f78827dd1f02bf79"),
}
CHUNK = 512 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_range(split: str, start: int, end: int, destination: Path) -> tuple[int, int]:
    expected = end - start + 1
    if destination.exists():
        if destination.stat().st_size == expected:
            return start, end
        raise RuntimeError(f"incomplete range requires audit: {destination}")
    url = f"https://huggingface.co/datasets/ses7720/CCDD-11/resolve/main/{split}.zip"
    temporary = destination.with_suffix(".partial")
    for attempt in range(5):
        try:
            received = temporary.stat().st_size if temporary.exists() else 0
            if received > expected:
                raise RuntimeError(f"oversized partial range: {temporary}")
            if received == expected:
                os.replace(temporary, destination)
                return start, end
            remaining_start = start + received
            request = urllib.request.Request(url, headers={"Range": f"bytes={remaining_start}-{end}"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(f"range server returned HTTP {response.status}")
                content_range = response.headers.get("Content-Range", "")
                if content_range != f"bytes {remaining_start}-{end}/{ARCHIVES[split][0]}":
                    raise RuntimeError(f"wrong Content-Range: {content_range}")
                with temporary.open("ab") as output:
                    shutil.copyfileobj(response, output, 8 * 1024 * 1024)
            if temporary.stat().st_size == expected:
                os.replace(temporary, destination)
                return start, end
            if temporary.stat().st_size > expected:
                raise RuntimeError(f"range response exceeded request: {temporary}")
            print(f"{split}: resuming short range {start}-{end} from {temporary.stat().st_size} bytes", flush=True)
        except (OSError, RuntimeError) as error:
            if attempt == 4:
                raise
            print(f"{split}: range retry {start}-{end} attempt={attempt + 1} error={error}", flush=True)
            time.sleep(min(20, 2**attempt))
    raise AssertionError("unreachable")


def run(split: str, workers: int) -> None:
    size, expected_sha = ARCHIVES[split]
    archive = DATA / f"{split}.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    archive_size = archive.stat().st_size
    if archive_size == size:
        if sha256(archive) != expected_sha:
            raise RuntimeError("complete archive has wrong SHA-256")
        print(f"{split}: already verified", flush=True)
        return
    if archive_size >= size:
        raise RuntimeError("prefix is larger than expected archive")
    parts_root = DATA / "parts" / split
    parts_root.mkdir(parents=True, exist_ok=True)
    state_path = parts_root / "state.json"
    state = {"split": split, "prefix_bytes": archive_size, "expected_bytes": size}
    content = json.dumps(state, sort_keys=True, indent=2) + "\n"
    if state_path.exists():
        if state_path.read_text() != content:
            raise RuntimeError("download prefix changed after range state was frozen")
    else:
        state_path.write_text(content)
    ranges = []
    for start in range(archive_size, size, CHUNK):
        end = min(start + CHUNK, size) - 1
        part = parts_root / f"{start:012d}-{end:012d}.part"
        ranges.append((start, end, part))
    print(f"{split}: prefix={archive_size} bytes, remaining_ranges={len(ranges)}, workers={workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_range, split, start, end, part) for start, end, part in ranges]
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            start, end = future.result()
            print(f"{split}: range {count}/{len(ranges)} complete ({start}-{end})", flush=True)
    assembled = DATA / f"{split}.assembled"
    if assembled.exists():
        raise RuntimeError(f"refusing to overwrite existing assembly: {assembled}")
    with assembled.open("xb") as output, archive.open("rb") as prefix:
        shutil.copyfileobj(prefix, output, 8 * 1024 * 1024)
        for start, end, part in ranges:
            if part.stat().st_size != end - start + 1:
                raise RuntimeError(f"range size changed: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if assembled.stat().st_size != size or sha256(assembled) != expected_sha:
        raise RuntimeError(f"assembled {split} failed published size/SHA check")
    os.replace(assembled, archive)
    print(f"{split}: VERIFIED SHA-256 {expected_sha}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(ARCHIVES), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be 1..16")
    run(args.split, args.workers)


if __name__ == "__main__":
    main()
