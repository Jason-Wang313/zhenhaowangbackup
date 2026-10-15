#!/usr/bin/env python3
"""Prepare source-disjoint CCDD-11 views for the two original-code CURE tracks.

No image is copied or transformed. Every view is a symlink to an extracted
CCDD-11 image; the immutable archive hashes are recorded separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/third_party/ccdd11/extracted"
OUTPUT = ROOT / "data/third_party/cure_two_track_v1"
VOCAB = (
    "clear", "low", "haze", "rain", "snow", "low_haze", "low_rain",
    "low_snow", "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
)
TRIPLES = frozenset(("low_haze_rain", "low_haze_snow"))
COMMON = tuple(name for name in VOCAB if name not in TRIPLES)
RESTORATION = tuple(name for name in COMMON if name != "clear")
SALT = "cure-two-track-source-split-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_exact(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(source, target.parent)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"conflicting symlink: {target}")
        return
    if target.exists():
        raise RuntimeError(f"refusing to overwrite: {target}")
    target.symlink_to(relative)


def source_names(main: Path, *, require_zero: bool) -> list[str]:
    clean = main / "clear"
    if not clean.is_dir():
        raise FileNotFoundError(clean)
    names = sorted(path.name for path in clean.glob("*.png"))
    if not names:
        raise RuntimeError(f"no clean PNG sources: {clean}")
    for name in names:
        for category in (*VOCAB, *(("zero",) if require_zero else ())):
            if not (main / category / name).is_file():
                raise FileNotFoundError(main / category / name)
    return names


def split_name(name: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"{SALT}:{name}".encode()).digest()[:8], "big")
    return "development" if value % 10 == 0 else "train"


def prepare(data_root: Path, output: Path, *, check_only: bool) -> dict:
    data_root = data_root.resolve()
    output = output.resolve()
    data_root.relative_to(ROOT)
    output.relative_to(ROOT)
    train_main = data_root / "half_train/main_data"
    train_sub = data_root / "half_train/sub_data"
    test_main = data_root / "half_test/main_data"
    train_names = source_names(train_main, require_zero=True)
    test_names = source_names(test_main, require_zero=False)
    shared_names = sorted(set(train_names) & set(test_names))
    train_hash_by_name = {name: sha256(train_main / "clear" / name) for name in train_names}
    train_clean_hashes = set(train_hash_by_name.values())
    test_clean_hashes = {sha256(test_main / "clear" / name) for name in test_names}
    if train_clean_hashes & test_clean_hashes:
        raise RuntimeError("train/test clean-image hash overlap")
    splits = {name: split_name(name) for name in train_names}
    hash_split = {}
    for name, digest in train_hash_by_name.items():
        previous = hash_split.setdefault(digest, splits[name])
        if previous != splits[name]:
            raise RuntimeError("exact-duplicate clean source spans train/development")
    if set(splits.values()) != {"train", "development"}:
        raise RuntimeError("source split has an empty partition")
    expected = {
        "paired_train": 0,
        "embedder_strict_train": 0,
        "embedder_dev_common": 0,
    }
    for name in train_names:
        stem = Path(name).stem
        if splits[name] == "train":
            paired = output / "paired_train"
            for category in (*VOCAB, "zero"):
                source = train_main / category / name
                target = paired / "main_data" / category / name
                if check_only:
                    if not target.is_symlink() or target.resolve() != source.resolve():
                        raise RuntimeError(f"missing/mismatched paired link: {target}")
                else:
                    link_exact(source, target)
                expected["paired_train"] += 1
            for category in RESTORATION:
                parts = category.split("_")
                factors = (parts[0], "zero") if len(parts) == 1 else (parts[0], parts[1])
                for suffix in (*factors, "half"):
                    source = train_sub / category / stem / f"{stem}_{suffix}_.png"
                    target = paired / "sub_data" / category / stem / source.name
                    if check_only:
                        if not target.is_symlink() or target.resolve() != source.resolve():
                            raise RuntimeError(f"missing/mismatched partial link: {target}")
                    else:
                        link_exact(source, target)
                    expected["paired_train"] += 1
            strict = output / "embedder_strict_train/main_data"
            for category in COMMON:
                source = train_main / category / name
                target = strict / category / name
                if check_only:
                    if not target.is_symlink() or target.resolve() != source.resolve():
                        raise RuntimeError(f"missing/mismatched strict link: {target}")
                else:
                    link_exact(source, target)
                expected["embedder_strict_train"] += 1
            # Exactly two extra non-triple presentations per source make the
            # strict embedder's epoch length equal the published-style track.
            offset = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "big") % len(COMMON)
            for extra in (0, 1):
                category = COMMON[(offset + extra) % len(COMMON)]
                source = train_main / category / name
                target = strict / category / f"resample{extra}_{name}"
                if check_only:
                    if not target.is_symlink() or target.resolve() != source.resolve():
                        raise RuntimeError(f"missing/mismatched strict resample: {target}")
                else:
                    link_exact(source, target)
                expected["embedder_strict_train"] += 1
        else:
            for category in COMMON:
                source = train_main / category / name
                target = output / "embedder_dev_common/main_data" / category / name
                if check_only:
                    if not target.is_symlink() or target.resolve() != source.resolve():
                        raise RuntimeError(f"missing/mismatched dev link: {target}")
                else:
                    link_exact(source, target)
                expected["embedder_dev_common"] += 1
    # Official EmbeddingImageDataset requires directories for the full fixed
    # vocabulary even when the held-out class has no samples.
    for view in ("embedder_strict_train", "embedder_dev_common"):
        for category in TRIPLES:
            directory = output / view / "main_data" / category
            if check_only:
                if not directory.is_dir() or any(directory.iterdir()):
                    raise RuntimeError(f"triple exposure in {view}: {directory}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                if any(directory.iterdir()):
                    raise RuntimeError(f"triple exposure in {view}: {directory}")
    metadata = {
        "schema_version": 1,
        "dataset_revision": "4784b88382f8616008ac9ae1e9be36c75f871f4e",
        "split_salt": SALT,
        "train_sources": sum(value == "train" for value in splits.values()),
        "development_sources": sum(value == "development" for value in splits.values()),
        "test_sources": len(test_names),
        "train_test_shared_filenames": len(shared_names),
        "view_link_counts": expected,
        "strict_triple_training_samples": 0,
        "strict_triple_development_samples": 0,
        "embedder_presentations_per_epoch_each_track": expected["embedder_strict_train"],
        "source_assignments_sha256": hashlib.sha256(
            json.dumps(splits, sort_keys=True).encode()
        ).hexdigest(),
        "test_source_names_sha256": hashlib.sha256(
            json.dumps(test_names).encode()
        ).hexdigest(),
    }
    if expected["embedder_strict_train"] != 12 * metadata["train_sources"]:
        raise RuntimeError("strict embedder budget mismatch")
    manifest = output / "metadata.json"
    content = json.dumps(metadata, sort_keys=True, indent=2) + "\n"
    if check_only:
        if not manifest.is_file() or manifest.read_text() != content:
            raise RuntimeError("metadata missing or mismatched")
    elif manifest.exists():
        if manifest.read_text() != content:
            raise RuntimeError("refusing to overwrite different metadata")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(content)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root, args.output, check_only=args.check), indent=2))


if __name__ == "__main__":
    main()
