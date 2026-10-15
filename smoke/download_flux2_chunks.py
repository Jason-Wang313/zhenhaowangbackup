#!/usr/bin/env python3
"""Resumable 50 MiB range downloader for the official FLUX.2 Klein Base 4B file."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/resolve/main/flux-2-klein-base-4b.safetensors"
EXPECTED = 7_751_105_712
CHUNK = 50_000_000
OUT = ROOT / "models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors"
WORK = OUT.parent / ".flux2-klein-base-4b.chunks"
ASSEMBLED = OUT.with_suffix(".safetensors.reassembled")


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = []
    for start in range(0, EXPECTED, CHUNK):
        end = min(start + CHUNK, EXPECTED) - 1
        jobs.append((start, end, WORK / f"chunk_{start:012d}_{end:012d}.part"))

    def fetch(job):
        start, end, path = job
        expected_size = end - start + 1
        if path.exists() and path.stat().st_size == expected_size:
            return path
        if path.exists():
            path.unlink()
        print(f"downloading {start}-{end}", flush=True)
        subprocess.run([
            "curl", "-L", "--fail", "--retry", "8", "--retry-delay", "5",
            "--range", f"{start}-{end}", "--output", str(path), URL,
        ], check=True)
        actual = path.stat().st_size
        if actual != expected_size:
            raise RuntimeError(f"bad chunk {path}: {actual} != {expected_size}")
        return path

    chunks = []
    for job in jobs:
        chunks.append(fetch(job))

    if ASSEMBLED.exists():
        ASSEMBLED.unlink()
    with ASSEMBLED.open("wb") as out:
        for path in chunks:
            with path.open("rb") as inp:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
    if ASSEMBLED.stat().st_size != EXPECTED:
        raise RuntimeError(f"bad assembled size: {ASSEMBLED.stat().st_size} != {EXPECTED}")
    digest = hashlib.sha256(ASSEMBLED.read_bytes()).hexdigest()
    print(f"assembled={ASSEMBLED}", flush=True)
    print(f"bytes={EXPECTED}", flush=True)
    print(f"sha256={digest}", flush=True)
    print("Validated size. SafeTensors validation should run before promotion.", flush=True)


if __name__ == "__main__":
    main()
