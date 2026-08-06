#!/usr/bin/env python3
"""Generate and calibrate one independent GenDeg validation image."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

from PIL import Image

from apply_gendeg_level_remap import darken_to_target, mean_luma


ROOT = Path(__file__).resolve().parents[1]
IMPL_PATH = ROOT / "smoke/run_gendeg_custom_batch.pyc"


def load_impl():
    spec = importlib.util.spec_from_file_location("gendeg_impl_validation", IMPL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load GenDeg runner: {IMPL_PATH}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(IMPL_PATH)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="002.png")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026080410)
    args = parser.parse_args()

    source = ROOT / "data/clean" / args.source
    if not source.is_file():
        raise RuntimeError(f"missing clean source: {source}")
    impl = load_impl()
    checkpoint = ROOT / "checkpoints/gendeg.ckpt"
    config = impl.GENDEG_ROOT / "configs/generate.yaml"
    model, model_wrap, device = impl.load_runtime(checkpoint, config)
    prompt = "a photograph in low-light conditions"
    clean_model = impl.load_clean(source, args.resolution)

    source_id = source.stem
    raw_root = ROOT / f"smoke/gendeg_validation_{source_id}_raw"
    final_root = ROOT / f"smoke/gendeg_validation_{source_id}"
    raw_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)

    raw_paths = {}
    final_paths = {}
    for level, (mu, sigma) in impl.LEVELS.items():
        raw = impl.sample_one(
            model, model_wrap, device, source, prompt, mu, sigma,
            args.seed, args.resolution, args.steps, 7.5, 1.5,
        )
        calibrated, gamma = impl.calibrate_exposure(
            raw, clean_model, impl.TARGET_LUMINANCE_RATIO[level]
        )
        raw_path = raw_root / f"{source_id}_{level}.png"
        final_path = final_root / f"{source_id}_{level}.png"
        impl.write_output(raw, raw_path)
        impl.write_output(calibrated, final_path)
        raw_paths[level] = raw_path
        final_paths[level] = final_path
        print(f"generated {level}: raw={mean_luma(raw_path):.2f} calibrated={mean_luma(final_path):.2f} gamma={gamma:.4f}", flush=True)

    # Apply the approved final review mapping without modifying the raw set.
    shutil.copy2(source, final_paths["L1"])
    shutil.copy2(final_paths["L3"], final_paths["L2"])
    shutil.copy2(final_paths["L4"], final_paths["L3"])
    shutil.copy2(final_paths["L5"], final_paths["L4"])
    clean_luma = mean_luma(source)
    new_l4_luma = mean_luma(final_paths["L4"])
    target = min(clean_luma * 0.10, new_l4_luma * 0.55)
    final_l5 = darken_to_target(raw_paths["L5"], final_paths["L5"], target)
    for _ in range(6):
        if final_l5 < new_l4_luma * 0.65:
            break
        target *= 0.5
        final_l5 = darken_to_target(raw_paths["L5"], final_paths["L5"], target)
    if final_l5 >= new_l4_luma * 0.65:
        raise RuntimeError("validation L5 did not become substantially darker than L4")
    print(f"FINAL validation source={source.name} clean={clean_luma:.2f} L4={new_l4_luma:.2f} L5={final_l5:.2f} target={target:.2f}", flush=True)
    print(f"Final images: {final_root}", flush=True)


if __name__ == "__main__":
    main()
