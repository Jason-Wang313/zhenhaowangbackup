#!/usr/bin/env python3
"""Resume-safe acquisition of the bounded official SDSD NumPy archives."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/sdsd_development_acquisition_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(raw: str) -> Path:
    path = (ROOT / raw).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise RuntimeError(f"path escapes workspace: {path}") from exc
    return path


def write_once_or_verify(path: Path, payload: Mapping[str, object]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable result differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def validate_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise RuntimeError("SDSD acquisition schema changed")
    repository = config["official_repository"]
    assert isinstance(repository, dict)
    repository_path = workspace_path(str(repository["path"]))
    revision = subprocess.check_output(
        ["git", "-C", str(repository_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != repository["revision"]:
        raise RuntimeError("official SDSD repository revision changed")
    files = config["files"]
    if not isinstance(files, list) or len(files) != 9:
        raise RuntimeError("expected exactly nine official split-archive files")
    names = [item["name"] for item in files]
    if len(set(names)) != len(names):
        raise RuntimeError("duplicate SDSD archive name")
    return config


def download_one(
    item: Mapping[str, object], archive_directory: Path, gdown: Path
) -> dict[str, object]:
    name = str(item["name"])
    expected_size = int(item["expected_size"])
    target = archive_directory / name
    partial = archive_directory / f"{name}.partial"
    if target.exists():
        if not target.is_file() or target.stat().st_size != expected_size:
            raise RuntimeError(f"existing final archive has wrong identity: {target}")
    else:
        command = [
            str(gdown),
            "--id",
            str(item["google_drive_id"]),
            "--continue",
            "--output",
            str(partial),
        ]
        completed = subprocess.run(command, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"gdown failed for {name}: status {completed.returncode}")
        if not partial.is_file() or partial.stat().st_size != expected_size:
            actual = partial.stat().st_size if partial.exists() else None
            raise RuntimeError(
                f"downloaded archive size mismatch for {name}: {actual} != {expected_size}"
            )
        # Publish without replacing or moving the resume artifact.
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
    gdown = workspace_path(str(config["gdown_executable"]))
    if not gdown.is_file() or not os.access(gdown, os.X_OK):
        raise RuntimeError(f"gdown is not executable: {gdown}")
    files = config["files"]
    assert isinstance(files, list)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["max_parallel_downloads"])
    ) as executor:
        futures = [
            executor.submit(download_one, item, archive_directory, gdown)
            for item in files
        ]
        records = [future.result() for future in futures]
    records.sort(key=lambda row: str(row["name"]))
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "pass",
        "automatic_gate_pass": True,
        "config_sha256": sha256(config_path),
        "downloader_sha256": sha256(Path(__file__)),
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
