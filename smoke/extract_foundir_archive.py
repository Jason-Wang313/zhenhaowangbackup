#!/usr/bin/env python3
"""Safely extract one official FoundIR ZIP into a new directory."""
from __future__ import annotations

import argparse
import os
import stat
import zipfile
from pathlib import Path, PurePosixPath


def validate_member(info: zipfile.ZipInfo) -> PurePosixPath:
    name = info.filename.replace("\\", "/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"UNSAFE_ARCHIVE_MEMBER {info.filename!r}")
    if path.parts and ":" in path.parts[0]:
        raise SystemExit(f"UNSAFE_ARCHIVE_MEMBER {info.filename!r}")
    unix_mode = info.external_attr >> 16
    if unix_mode and stat.S_ISLNK(unix_mode):
        raise SystemExit(f"SYMLINK_ARCHIVE_MEMBER {info.filename!r}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    destination = args.destination.resolve()
    partial = destination.with_name(destination.name + ".partial")
    if not archive.is_file():
        raise SystemExit(f"MISSING_ARCHIVE {archive}")
    if destination.exists() or partial.exists():
        raise SystemExit(f"REFUSE_EXISTING_DESTINATION {destination} or {partial}")

    with zipfile.ZipFile(archive) as handle:
        bad_member = handle.testzip()
        if bad_member is not None:
            raise SystemExit(f"BAD_ZIP_CRC {bad_member}")
        members = handle.infolist()
        if not members:
            raise SystemExit("EMPTY_ARCHIVE")
        normalized: set[str] = set()
        uncompressed_bytes = 0
        for info in members:
            path = validate_member(info)
            key = path.as_posix().rstrip("/")
            if key in normalized:
                raise SystemExit(f"DUPLICATE_ARCHIVE_MEMBER {key}")
            normalized.add(key)
            uncompressed_bytes += info.file_size

        partial.mkdir(parents=True)
        handle.extractall(partial)

    os.rename(partial, destination)
    print(
        "FOUNDIR_ARCHIVE_EXTRACT_PASS",
        f"archive={archive}",
        f"destination={destination}",
        f"members={len(members)}",
        f"uncompressed_bytes={uncompressed_bytes}",
    )


if __name__ == "__main__":
    main()
