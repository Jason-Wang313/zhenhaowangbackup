#!/usr/bin/env python3
"""Selectively extract the frozen 500-pair SDSD development contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import struct
import sys
import threading
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import requests

from smoke.acquire_sdsd_development_archives import ROOT, sha256, validate_config, workspace_path
from smoke.acquire_sdsd_development_archives_direct import parse_content_range


DEFAULT_CONFIG = ROOT / "configs/sdsd_development_acquisition_v1c.json"
OUTPUT_ROOT = ROOT / "data/sdsd_official_selective_v1"
OUTPUT_CSV = ROOT / "manifests/sdsd_lowlight_development_v1.csv"
OUTPUT_JSON = ROOT / "manifests/sdsd_lowlight_development_v1.json"
CENTRAL_HEADER = struct.Struct("<4s6H3L5H2L")
LOCAL_HEADER = struct.Struct("<4s5H3L2H")
EOCD = struct.Struct("<4s4H2LH")
PATH_PATTERN = re.compile(
    r"^(indoor_np|outdoor_np)/(GT|input)/(pair\d+)/([^/]+\.npy)$"
)
_progress_lock = threading.Lock()
_completed = 0


def resolve_zip64_values(
    uncompressed_size: int,
    compressed_size: int,
    local_offset: int,
    disk: int,
    extra: bytes,
) -> tuple[int, int, int, int]:
    fields: list[int] = []
    cursor = 0
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise RuntimeError("truncated ZIP central extra header")
        header_id, length = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        end = cursor + length
        if end > len(extra):
            raise RuntimeError("truncated ZIP central extra value")
        if header_id == 0x0001:
            zip64 = extra[cursor:end]
            zip64_cursor = 0
            for needed, width in (
                (uncompressed_size == 0xFFFFFFFF, 8),
                (compressed_size == 0xFFFFFFFF, 8),
                (local_offset == 0xFFFFFFFF, 8),
                (disk == 0xFFFF, 4),
            ):
                if not needed:
                    continue
                if zip64_cursor + width > len(zip64):
                    raise RuntimeError("truncated required ZIP64 central value")
                fmt = "<Q" if width == 8 else "<L"
                fields.append(struct.unpack_from(fmt, zip64, zip64_cursor)[0])
                zip64_cursor += width
            break
        cursor = end
    required = sum(
        value
        for value in (
            uncompressed_size == 0xFFFFFFFF,
            compressed_size == 0xFFFFFFFF,
            local_offset == 0xFFFFFFFF,
            disk == 0xFFFF,
        )
    )
    if len(fields) != required:
        raise RuntimeError("required ZIP64 central extra field is missing")
    iterator = iter(fields)
    if uncompressed_size == 0xFFFFFFFF:
        uncompressed_size = next(iterator)
    if compressed_size == 0xFFFFFFFF:
        compressed_size = next(iterator)
    if local_offset == 0xFFFFFFFF:
        local_offset = next(iterator)
    if disk == 0xFFFF:
        disk = next(iterator)
    return uncompressed_size, compressed_size, local_offset, disk


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"existing immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)


def range_get(file_id: str, start: int, end: int, total: int) -> bytes:
    url = (
        "https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )
    failure: Exception | None = None
    for attempt in range(1, 13):
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
                or actual_total != total
                or response.headers.get("Content-Type") != "application/octet-stream"
                or len(response.content) != end - start + 1
            ):
                raise RuntimeError("range response identity mismatch")
            return response.content
        except (requests.RequestException, RuntimeError) as exc:
            failure = exc
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"range request exhausted retries: {failure}")


def volumes_from_config(config: Mapping[str, object], subset: str) -> list[dict[str, object]]:
    files = config["files"]
    assert isinstance(files, list)
    prefix = f"{subset}_np"
    selected = [dict(item) for item in files if str(item["name"]).startswith(prefix)]
    def disk_number(item: Mapping[str, object]) -> int:
        name = str(item["name"])
        return int(name.rsplit("z", 1)[1]) - 1 if ".z" in name else 999
    selected.sort(key=disk_number)
    return selected


def central_directory(volumes: Sequence[Mapping[str, object]], subset: str) -> list[dict[str, object]]:
    cache = OUTPUT_ROOT / "central_directory" / f"{subset}.bin"
    final_volume = volumes[-1]
    total = int(final_volume["expected_size"])
    tail_start = max(0, total - 1024 * 1024)
    tail = range_get(str(final_volume["google_drive_id"]), tail_start, total - 1, total)
    position = tail.rfind(b"PK\x05\x06")
    if position < 0 or position + EOCD.size > len(tail):
        raise RuntimeError(f"EOCD not found for {subset}")
    values = EOCD.unpack_from(tail, position)
    disk_number, central_disk, entries_disk, entries_total, size, offset, comment = values[1:]
    if (
        disk_number != len(volumes) - 1
        or central_disk != disk_number
        or entries_disk != entries_total
        or comment != 0
        or offset + size > total
    ):
        raise RuntimeError(f"unsupported split ZIP layout for {subset}: {values}")
    if cache.exists():
        data = cache.read_bytes()
        if len(data) != size:
            raise RuntimeError(f"cached central directory size changed: {cache}")
    else:
        data = range_get(str(final_volume["google_drive_id"]), offset, offset + size - 1, total)
        write_once(cache, data)
    entries: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(data):
        values = CENTRAL_HEADER.unpack_from(data, cursor)
        if values[0] != b"PK\x01\x02":
            raise RuntimeError(f"bad central header at {subset}:{cursor}")
        filename_length, extra_length, comment_length = values[10:13]
        raw_name = data[
            cursor + CENTRAL_HEADER.size : cursor + CENTRAL_HEADER.size + filename_length
        ]
        extra_start = cursor + CENTRAL_HEADER.size + filename_length
        extra = data[extra_start : extra_start + extra_length]
        uncompressed_size, compressed_size, local_offset, disk = resolve_zip64_values(
            int(values[9]), int(values[8]), int(values[16]), int(values[13]), extra
        )
        encoding = "utf-8" if values[3] & 0x800 else "cp437"
        entries.append(
            {
                "name": raw_name.decode(encoding),
                "flag": values[3],
                "compression": values[4],
                "crc32": values[7],
                "compressed_size": compressed_size,
                "uncompressed_size": uncompressed_size,
                "disk": disk,
                "local_offset": local_offset,
            }
        )
        cursor += CENTRAL_HEADER.size + filename_length + extra_length + comment_length
    if cursor != len(data) or len(entries) != entries_total:
        raise RuntimeError(f"central directory count mismatch for {subset}")
    return entries


def select_pairs(
    entries_by_subset: Mapping[str, Sequence[Mapping[str, object]]],
    policy: Mapping[str, object],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for subset in ("indoor", "outdoor"):
        grouped: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in entries_by_subset[subset]:
            match = PATH_PATTERN.fullmatch(str(entry["name"]))
            if match is None:
                continue
            root, kind, sequence, filename = match.groups()
            if root != f"{subset}_np":
                raise RuntimeError("subset/root mismatch")
            grouped[sequence][kind].append(entry)
        sequences: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
        for sequence, kinds in grouped.items():
            if set(kinds) != {"GT", "input"}:
                raise RuntimeError(f"missing SDSD stream: {subset}/{sequence}")
            gt_entries = sorted(kinds["GT"], key=lambda value: str(value["name"]))
            lq_entries = sorted(kinds["input"], key=lambda value: str(value["name"]))
            if len(gt_entries) != len(lq_entries):
                raise RuntimeError(
                    f"SDSD stream length mismatch: {subset}/{sequence} "
                    f"GT={len(gt_entries)} input={len(lq_entries)}"
                )
            sequences[sequence] = list(zip(gt_entries, lq_entries, strict=True))
        expected_sequences = 70 if subset == "indoor" else 80
        if len(sequences) != expected_sequences:
            raise RuntimeError(
                f"unexpected {subset} sequence count: {len(sequences)} != {expected_sequences}"
            )
        outer = set(policy[f"official_{subset}_outer_sequences"])
        if not outer.issubset(sequences):
            raise RuntimeError(f"missing official outer sequence in {subset}")
        for sequence in sorted(sequences, key=lambda value: int(value[4:])):
            frame_pairs = sequences[sequence]
            if sequence in outer:
                positions = [int(value) for value in policy["outer_frame_indices_zero_based"]]
                role = "outer_development"
            else:
                positions = [0, (len(frame_pairs) - 1) // 2, len(frame_pairs) - 1]
                role = "training"
            if len(set(positions)) != len(positions) or positions[-1] >= len(frame_pairs):
                raise RuntimeError(f"sequence too short for frozen selection: {subset}/{sequence}")
            for position in positions:
                gt_entry, lq_entry = frame_pairs[position]
                selected.append(
                    {
                        "family": "sdsd",
                        "subset": subset,
                        "role": role,
                        "sequence": sequence,
                        "frame_position": position,
                        "gt_frame_filename": Path(str(gt_entry["name"])).name,
                        "lq_frame_filename": Path(str(lq_entry["name"])).name,
                        "gt_entry": gt_entry,
                        "lq_entry": lq_entry,
                    }
                )
    if len(selected) != int(policy["expected_total_pairs"]):
        raise RuntimeError(f"bounded pair count changed: {len(selected)}")
    return selected


def read_across_volumes(
    volumes: Sequence[Mapping[str, object]], disk: int, offset: int, length: int
) -> bytes:
    pieces: list[bytes] = []
    remaining = length
    current_disk = disk
    current_offset = offset
    while remaining:
        if current_disk >= len(volumes):
            raise RuntimeError("entry extends beyond final split ZIP volume")
        volume = volumes[current_disk]
        size = int(volume["expected_size"])
        available = size - current_offset
        if available <= 0:
            current_disk += 1
            current_offset = 0
            continue
        count = min(remaining, available)
        pieces.append(
            range_get(
                str(volume["google_drive_id"]),
                current_offset,
                current_offset + count - 1,
                size,
            )
        )
        remaining -= count
        current_disk += 1
        current_offset = 0
    return b"".join(pieces)


def extract_entry(
    subset: str,
    entry: Mapping[str, object],
    volumes: Sequence[Mapping[str, object]],
    total_entries: int,
) -> Path:
    global _completed
    output = OUTPUT_ROOT / "arrays" / str(entry["name"])
    expected_size = int(entry["uncompressed_size"])
    expected_crc = int(entry["crc32"])
    if output.exists():
        content = output.read_bytes()
        if len(content) != expected_size or zlib.crc32(content) & 0xFFFFFFFF != expected_crc:
            raise RuntimeError(f"existing selective extraction differs: {output}")
    else:
        disk = int(entry["disk"])
        offset = int(entry["local_offset"])
        prefix = read_across_volumes(volumes, disk, offset, LOCAL_HEADER.size + 4096)
        values = LOCAL_HEADER.unpack_from(prefix)
        if values[0] != b"PK\x03\x04" or values[3] != entry["compression"]:
            raise RuntimeError(f"local header mismatch: {entry['name']}")
        filename_length, extra_length = values[9], values[10]
        data_offset = offset + LOCAL_HEADER.size + filename_length + extra_length
        compressed = read_across_volumes(
            volumes, disk, data_offset, int(entry["compressed_size"])
        )
        if int(entry["compression"]) == 0:
            content = compressed
        elif int(entry["compression"]) == 8:
            content = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"unsupported ZIP compression: {entry['compression']}")
        if len(content) != expected_size or zlib.crc32(content) & 0xFFFFFFFF != expected_crc:
            raise RuntimeError(f"extracted member integrity failure: {entry['name']}")
        write_once(output, content)
    with _progress_lock:
        _completed += 1
        if _completed % 50 == 0 or _completed == total_entries:
            print(f"SDSD_SELECTIVE_PROGRESS entries={_completed}/{total_entries}", flush=True)
    return output


def perceptual_hash_bgr(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    transformed = cv2.dct(resized)
    low = transformed[:8, :8]
    median = float(np.median(low[1:]))
    bits = (low >= median).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def manifest_rows(selected: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    protected = json.loads((ROOT / "manifests/foundir_lowlight_official.json").read_text())
    protected_hashes = {
        value
        for record in protected["records"]
        for value in (record["gt_sha256"], record["lq_sha256"])
    }
    rows: list[dict[str, object]] = []
    for pair in selected:
        gt_entry = pair["gt_entry"]
        lq_entry = pair["lq_entry"]
        assert isinstance(gt_entry, dict) and isinstance(lq_entry, dict)
        gt_path = OUTPUT_ROOT / "arrays" / str(gt_entry["name"])
        lq_path = OUTPUT_ROOT / "arrays" / str(lq_entry["name"])
        gt = np.load(gt_path, mmap_mode="r", allow_pickle=False)
        lq = np.load(lq_path, mmap_mode="r", allow_pickle=False)
        if gt.dtype != np.uint8 or lq.dtype != np.uint8 or gt.shape != (512, 960, 3) or lq.shape != gt.shape:
            raise RuntimeError(f"unexpected SDSD array identity: {pair['subset']}/{pair['sequence']}")
        gt_array = np.asarray(gt)
        lq_array = np.asarray(lq)
        gt_gray = cv2.cvtColor(gt_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lq_gray = cv2.cvtColor(lq_array, cv2.COLOR_BGR2GRAY).astype(np.float32)
        shift, response = cv2.phaseCorrelate(lq_gray, gt_gray)
        gt_file_hash = sha256(gt_path)
        lq_file_hash = sha256(lq_path)
        if gt_file_hash in protected_hashes or lq_file_hash in protected_hashes:
            raise RuntimeError("exact protected file SHA-256 overlap detected")
        gt_rgb = np.ascontiguousarray(gt_array[..., ::-1])
        lq_rgb = np.ascontiguousarray(lq_array[..., ::-1])
        rows.append(
            {
                "family": pair["family"],
                "subset": pair["subset"],
                "role": pair["role"],
                "sequence_id": f"sdsd_{pair['subset']}_{pair['sequence']}",
                "frame_position": pair["frame_position"],
                "gt_frame_filename": pair["gt_frame_filename"],
                "lq_frame_filename": pair["lq_frame_filename"],
                "lq_path": str(lq_path.relative_to(ROOT)),
                "gt_path": str(gt_path.relative_to(ROOT)),
                "lq_file_sha256": lq_file_hash,
                "gt_file_sha256": gt_file_hash,
                "lq_rgb_pixel_sha256": hashlib.sha256(lq_rgb.tobytes()).hexdigest(),
                "gt_rgb_pixel_sha256": hashlib.sha256(gt_rgb.tobytes()).hexdigest(),
                "lq_phash": perceptual_hash_bgr(lq_array),
                "gt_phash": perceptual_hash_bgr(gt_array),
                "width": 960,
                "height": 512,
                "stored_color_order": "BGR",
                "metric_color_order": "RGB",
                "alignment_dx_pixels": float(shift[0]),
                "alignment_dy_pixels": float(shift[1]),
                "alignment_phase_response": float(response),
                "protected_exact_file_overlap": False,
                "absolute_l1_l5_label_available": False,
            }
        )
    return rows


def run(config_path: Path) -> None:
    config = validate_config(config_path)
    policy = config["bounded_pair_policy"]
    assert isinstance(policy, dict)
    volumes = {
        subset: volumes_from_config(config, subset) for subset in ("indoor", "outdoor")
    }
    entries = {
        subset: central_directory(volumes[subset], subset) for subset in ("indoor", "outdoor")
    }
    selected = select_pairs(entries, policy)
    extraction_jobs = [
        (str(pair["subset"]), pair[key], volumes[str(pair["subset"])])
        for pair in selected
        for key in ("gt_entry", "lq_entry")
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(extract_entry, subset, entry, subset_volumes, len(extraction_jobs))
            for subset, entry, subset_volumes in extraction_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    rows = manifest_rows(selected)
    fieldnames = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_content = stream.getvalue().encode()
    write_once(OUTPUT_CSV, csv_content)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[f"{row['subset']}:{row['role']}"] += 1
    metadata = {
        "schema_version": 1,
        "status": "pass",
        "automatic_gate_pass": True,
        "dataset": "SDSD",
        "annotation_kind": "paired_real_restoration_no_absolute_l1_l5_labels",
        "use_policy": config["purpose"],
        "redistribution_authorized": False,
        "official_repository_revision": config["official_repository"]["revision"],
        "config_sha256": sha256(config_path),
        "builder_sha256": sha256(Path(__file__)),
        "manifest_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "manifest_csv_sha256": hashlib.sha256(csv_content).hexdigest(),
        "pairs": len(rows),
        "sequence_groups": len({row["sequence_id"] for row in rows}),
        "counts": dict(sorted(counts.items())),
        "alignment_policy": "audit_only_no_warp_no_exclusion",
        "protected_overlap": {
            "exact_file_sha256_matches": 0,
            "protected_manifest_only_used": True,
            "protected_pixels_opened": False,
            "perceptual_comparison": "not_run_no_persisted_protected_phash_index; candidate hashes frozen for future authorized comparison",
        },
        "foundir_test_access": False,
    }
    write_once(OUTPUT_JSON, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    print(
        f"SDSD_SELECTIVE_DEVELOPMENT_PASS pairs={len(rows)} sequences={metadata['sequence_groups']}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    try:
        run(arguments.config.resolve())
    except Exception as exc:
        print(f"SDSD_SELECTIVE_DEVELOPMENT_FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
