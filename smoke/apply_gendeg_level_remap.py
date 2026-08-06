#!/usr/bin/env python3
"""Apply the confirmed final GenDeg level mapping.

Mapping:
  new L1 = literal clean source
  new L2 = old L3
  new L3 = old L4
  new L4 = old L5
  new L5 = old L5 darkened substantially further

The raw GenDeg outputs are never modified. Existing final outputs are backed
up in a timestamped directory before the replacement is committed.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("L1", "L2", "L3", "L4", "L5")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def mean_luma(path: Path) -> float:
    with Image.open(path).convert("RGB") as im:
        px = list(im.getdata())
    return sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in px) / len(px)


def darken_to_target(source: Path, target: Path, target_luma: float) -> float:
    """Apply a measured global gamma curve and return resulting luminance.

    A log-ratio gamma estimate is not reliable for arbitrary RGB histograms,
    so this searches gamma values against the actual rendered mean luminance.
    """
    with Image.open(source).convert("RGB") as im:
        def render(gamma: float) -> Image.Image:
            return im.point(lambda value: int(round(255.0 * (value / 255.0) ** gamma)))

        def measured(image: Image.Image) -> float:
            pixels = list(image.getdata())
            return sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels) / len(pixels)

        raw = measured(im)
        if raw <= 0.001:
            out = im
        else:
            desired = max(0.5, min(255.0, target_luma))
            if desired >= raw:
                gamma = max(0.05, math.log(desired / 255.0) / math.log(raw / 255.0))
                out = render(gamma)
            else:
                low, high = 1.0, 32.0
                for _ in range(32):
                    gamma = (low + high) / 2.0
                    if measured(render(gamma)) > desired:
                        low = gamma
                    else:
                        high = gamma
                out = render(high)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.save(target)
    return mean_luma(target)


def source_for_name(name: str, clean: Path, smoke: bool) -> Path:
    if smoke:
        sources = image_files(clean)
        if not sources:
            raise RuntimeError(f"No clean source images found in {clean}")
        return sources[0]
    stem = Path(name).stem
    for suffix in ("_L1", "_L2", "_L3", "_L4", "_L5"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = clean / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Could not match {name} to a clean source in {clean}")


def apply_one(root: Path, clean: Path, smoke: bool) -> None:
    final_root = root / "smoke/gendeg" if smoke else root / "data/gendeg"
    folders = {level: final_root / f"low_light_{level}" for level in LEVELS}
    if smoke:
        # The smoke runner uses one file per level directly in smoke/gendeg.
        old = {level: image_files(final_root) for level in LEVELS}
        old = {level: [p for p in image_files(final_root) if f"_{level}" in p.stem] for level in LEVELS}
        if any(len(old[level]) != 1 for level in LEVELS):
            raise RuntimeError("Smoke baseline must contain exactly one *_L1..*_L5 image in smoke/gendeg")
        names = [old["L1"][0].name]
        old_by_level = {level: old[level][0] for level in LEVELS}
        targets = {level: final_root / names[0] for level in LEVELS}
    else:
        if any(not folder.is_dir() for folder in folders.values()):
            raise RuntimeError(f"Missing one or more final level folders under {final_root}")
        sets = [{p.name for p in image_files(folders[level])} for level in LEVELS]
        if not sets[0] or any(current != sets[0] for current in sets[1:]):
            raise RuntimeError("Final baseline folders must contain the same non-empty filenames at every level")
        names = sorted(sets[0])

    stage = final_root.parent / f".{final_root.name}.remap-stage-{int(time.time())}"
    backup = final_root.parent / f".{final_root.name}.before-remap-{int(time.time())}"
    if stage.exists() or backup.exists():
        raise RuntimeError("Timestamp collision; wait one second and retry")
    stage.mkdir(parents=True)
    try:
        for name in names:
            clean_source = source_for_name(name, clean, smoke)
            if smoke:
                old_paths = old_by_level
                suffix = Path(name).suffix
                base = Path(name).stem
                for old_level in LEVELS:
                    if base.endswith(f"_{old_level}"):
                        base = base[: -(len(old_level) + 1)]
                        break
                output_paths = {level: stage / f"{base}_{level}{suffix}" for level in LEVELS}
            else:
                old_paths = {level: folders[level] / name for level in LEVELS}
                output_paths = {level: stage / f"low_light_{level}" / name for level in LEVELS}
                for path in output_paths.values():
                    path.parent.mkdir(parents=True, exist_ok=True)

            # L1 is a byte-for-byte copy of the clean source.
            shutil.copy2(clean_source, output_paths["L1"])
            shutil.copy2(old_paths["L3"], output_paths["L2"])
            shutil.copy2(old_paths["L4"], output_paths["L3"])
            shutil.copy2(old_paths["L5"], output_paths["L4"])

            clean_luma = mean_luma(clean_source)
            old_l5_luma = mean_luma(old_paths["L5"])
            new_l4_luma = mean_luma(output_paths["L4"])
            # 10% of clean is the baseline; the min terms guarantee that the
            # new L5 is also materially darker than the current/new L4.
            target = min(clean_luma * 0.10, old_l5_luma * 0.55, new_l4_luma * 0.55)
            target = max(0.5, target)
            # A single gamma estimate can miss the requested mean after
            # 8-bit quantization, so tighten the target until the measured
            # result is unambiguously darker than L4.
            final_l5 = darken_to_target(old_paths["L5"], output_paths["L5"], target)
            for _ in range(6):
                if final_l5 < new_l4_luma * 0.65:
                    break
                target *= 0.5
                final_l5 = darken_to_target(old_paths["L5"], output_paths["L5"], target)
            if final_l5 >= new_l4_luma * 0.65:
                raise RuntimeError(f"L5 is not substantially darker than L4 for {name}: {final_l5:.2f} vs {new_l4_luma:.2f}")
            print(f"{name}: clean={clean_luma:.2f} new_L4={new_l4_luma:.2f} new_L5={final_l5:.2f} target={target:.2f}")

        # Keep the old final set recoverable, then atomically install the stage.
        if smoke:
            backup.mkdir()
            for path in image_files(final_root):
                shutil.copy2(path, backup / path.name)
            for path in image_files(final_root):
                path.unlink()
            for path in image_files(stage):
                shutil.move(str(path), str(final_root / path.name))
        else:
            final_root.rename(backup)
            stage.rename(final_root)
        print(f"Remap complete: {final_root}")
        print(f"Previous final outputs preserved at: {backup}")
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_only and args.data_only:
        parser.error("choose only one of --smoke-only or --data-only")
    if not args.data_only:
        apply_one(ROOT, ROOT / "data/clean", smoke=True)
    if not args.smoke_only:
        apply_one(ROOT, ROOT / "data/clean", smoke=False)


if __name__ == "__main__":
    main()
