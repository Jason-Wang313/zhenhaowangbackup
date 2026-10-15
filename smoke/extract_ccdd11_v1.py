#!/usr/bin/env python3
"""Hash-verify and safely extract the pinned CCDD-11 ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import zipfile
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/third_party/ccdd11"
ARCHIVES = {
    "half_train": (74063577783, "fd957ac7690060652d9d24c6f03292bad799b6e9e44e05df9dc3183aaac3b268"),
    "half_test": (16392141571, "2ed9fd82d4cb5b05615b7258b06c840d60b60abf69cd5e49f78827dd1f02bf79"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32(path: Path) -> int:
    value = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return value


def safe_members(archive: zipfile.ZipFile, split: str) -> list[tuple[zipfile.ZipInfo, Path]]:
    members = []
    total_uncompressed = 0
    for info in archive.infolist():
        if info.is_dir():
            continue
        path = Path(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise RuntimeError(f"unsafe ZIP path: {info.filename}")
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"ZIP symlink rejected: {info.filename}")
        parts = path.parts
        if parts and parts[0] == split:
            parts = parts[1:]
        if not parts or parts[0] not in {"main_data", "sub_data"}:
            raise RuntimeError(f"unexpected CCDD-11 archive layout: {info.filename}")
        target = Path(*parts)
        members.append((info, target))
        total_uncompressed += info.file_size
    if not members or total_uncompressed > 5_000_000_000_000:
        raise RuntimeError("empty or implausibly large CCDD-11 ZIP")
    if len({target for _, target in members}) != len(members):
        raise RuntimeError("duplicate ZIP target path")
    return members


def verify_and_extract(data: Path, split: str, *, extract: bool) -> dict:
    data = data.resolve()
    data.relative_to(ROOT)
    expected_size, expected_sha = ARCHIVES[split]
    archive_path = data / f"{split}.zip"
    if archive_path.stat().st_size != expected_size:
        raise RuntimeError(f"archive incomplete or wrong size: {archive_path}")
    actual_sha = sha256(archive_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"archive SHA-256 mismatch: {archive_path}")
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive, split)
        total = sum(info.file_size for info, _ in members)
        if extract:
            destination = data / "extracted" / split
            for index, (info, relative) in enumerate(members, start=1):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size == info.file_size and crc32(target) == info.CRC:
                        continue
                    raise RuntimeError(f"differing existing extracted file: {target}")
                temporary = target.with_name(target.name + ".partial")
                if temporary.exists():
                    raise RuntimeError(f"partial file requires audit before resume: {temporary}")
                with archive.open(info) as source, temporary.open("xb") as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                if temporary.stat().st_size != info.file_size or crc32(temporary) != info.CRC:
                    raise RuntimeError(f"extracted size mismatch: {temporary}")
                os.replace(temporary, target)
                if index % 1000 == 0:
                    print(f"{split}: extracted {index}/{len(members)} files", flush=True)
    return {
        "split": split,
        "archive": str(archive_path.relative_to(ROOT)),
        "archive_sha256": actual_sha,
        "members": len(members),
        "uncompressed_bytes": total,
        "extracted": extract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--split", choices=tuple(ARCHIVES), required=True)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_and_extract(args.data, args.split, extract=args.extract), indent=2))


if __name__ == "__main__":
    main()
