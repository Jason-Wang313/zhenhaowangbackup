"""Fail-closed paired low-light manifest loading and aligned cropping."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {
    "pair_id",
    "source_id",
    "clean_path",
    "degraded_path",
    "degradation_type",
    "project_level",
    "seed",
    "split",
    "clean_sha256",
    "degraded_sha256",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_LEVELS = {"L1", "L2", "L3", "L4", "L5"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    source_id: str
    clean_path: Path
    degraded_path: Path
    project_level: str
    seed: int
    split: str
    clean_sha256: str
    degraded_sha256: str

    @property
    def level_index(self) -> int:
        return int(self.project_level[1:])


def resolve_workspace_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes workspace: {value}") from exc
    return resolved


def load_pair_manifest(path: Path, root: Path = ROOT) -> list[PairRecord]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"manifest has no header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"manifest missing columns: {sorted(missing)}")
        rows = list(reader)

    records: list[PairRecord] = []
    seen_pairs: set[str] = set()
    for row in rows:
        pair_id = row["pair_id"]
        if not pair_id or pair_id in seen_pairs:
            raise ValueError(f"empty or duplicate pair_id: {pair_id!r}")
        seen_pairs.add(pair_id)
        if not row["source_id"]:
            raise ValueError(f"empty source_id for pair: {pair_id}")
        if row["degradation_type"] != "low_light":
            raise ValueError(f"non-low-light record is forbidden: {pair_id}")
        if row["project_level"] not in VALID_LEVELS:
            raise ValueError(f"bad project level for {pair_id}: {row['project_level']}")
        if row["split"] not in VALID_SPLITS:
            raise ValueError(f"bad split for {pair_id}: {row['split']}")
        for field in ("clean_sha256", "degraded_sha256"):
            if SHA256_RE.fullmatch(row[field]) is None:
                raise ValueError(f"bad {field} for {pair_id}")
        try:
            record_seed = int(row["seed"])
        except ValueError as exc:
            raise ValueError(f"bad seed for {pair_id}: {row['seed']!r}") from exc
        if record_seed < 0:
            raise ValueError(f"negative seed for {pair_id}: {record_seed}")
        records.append(
            PairRecord(
                pair_id=pair_id,
                source_id=row["source_id"],
                clean_path=resolve_workspace_path(row["clean_path"], root),
                degraded_path=resolve_workspace_path(row["degraded_path"], root),
                project_level=row["project_level"],
                seed=record_seed,
                split=row["split"],
                clean_sha256=row["clean_sha256"],
                degraded_sha256=row["degraded_sha256"],
            )
        )
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    validate_source_disjoint_splits(records)
    return records


def validate_source_disjoint_splits(records: list[PairRecord]) -> None:
    source_split: dict[str, str] = {}
    for record in records:
        previous = source_split.setdefault(record.source_id, record.split)
        if previous != record.split:
            raise ValueError(
                f"source split leakage: {record.source_id} appears in {previous} and {record.split}"
            )


def _crop_coordinates(record: PairRecord, width: int, height: int, crop_size: int, seed: int) -> tuple[int, int]:
    if width < crop_size or height < crop_size:
        raise ValueError(
            f"pair {record.pair_id} is smaller than crop {crop_size}: {width}x{height}"
        )
    digest = hashlib.sha256(f"{seed}:{record.pair_id}".encode()).digest()
    raw_x = int.from_bytes(digest[:8], "big")
    raw_y = int.from_bytes(digest[8:16], "big")
    left = raw_x % (width - crop_size + 1)
    top = raw_y % (height - crop_size + 1)
    return left, top


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return tensor


class LowLightPairDataset(Dataset[dict[str, object]]):
    """Aligned, deterministic crops from the authoritative pair manifest."""

    def __init__(
        self,
        records: list[PairRecord],
        split: str,
        crop_size: int,
        seed: int,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {split}")
        if crop_size <= 0 or crop_size % 16 != 0:
            raise ValueError("crop_size must be a positive multiple of 16")
        self.records = [record for record in records if record.split == split]
        if not self.records:
            raise ValueError(f"no records for split: {split}")
        self.split = split
        self.crop_size = crop_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        if not record.clean_path.is_file() or not record.degraded_path.is_file():
            raise FileNotFoundError(f"missing pair files: {record.pair_id}")
        with Image.open(record.clean_path) as clean_image:
            clean = clean_image.convert("RGB")
            clean_size = clean.size
            with Image.open(record.degraded_path) as degraded_image:
                degraded = degraded_image.convert("RGB")
                if degraded.size != clean_size:
                    raise ValueError(
                        f"unaligned pair dimensions for {record.pair_id}: "
                        f"clean={clean_size} degraded={degraded.size}"
                    )
                left, top = _crop_coordinates(
                    record,
                    width=clean_size[0],
                    height=clean_size[1],
                    crop_size=self.crop_size,
                    seed=self.seed,
                )
                box = (left, top, left + self.crop_size, top + self.crop_size)
                clean_crop = clean.crop(box)
                degraded_crop = degraded.crop(box)

        return {
            "clean": _image_to_tensor(clean_crop),
            "degraded": _image_to_tensor(degraded_crop),
            "pair_id": record.pair_id,
            "source_id": record.source_id,
            "project_level": record.project_level,
            "level_index": record.level_index,
            "crop_xy": torch.tensor([left, top], dtype=torch.int64),
        }
