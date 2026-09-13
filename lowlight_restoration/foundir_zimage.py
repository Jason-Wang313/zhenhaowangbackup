"""Contracts and aligned-pair utilities for the advisor-approved Z-Image Blur teacher."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STATUS = "advisor_authorized_zimage_blur_teacher_tiny_overfit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    path.write_bytes(content)


def load_config(path: Path) -> dict[str, object]:
    config = json.loads(path.read_text())
    validate_config(config)
    config["_config_path"] = str(path.resolve())
    return config


def validate_config(config: Mapping[str, object]) -> None:
    if config.get("schema_version") != 1 or config.get("status") != EXPECTED_STATUS:
        raise RuntimeError("config is not the authorized Z-Image Blur teacher tiny-overfit gate")
    scope = config["scope"]
    if scope.get("category") != "01Blur" or scope.get("teacher_only") is not True:
        raise RuntimeError("teacher Blur-only scope changed")
    forbidden = (
        "distillation",
        "decoder_finetuning",
        "qwen_degradation_conditioning",
        "mixed_degradations",
        "sealed_internal_accessed",
        "official_foundir_test_accessed",
    )
    if any(scope.get(key) is not False for key in forbidden):
        raise RuntimeError("a forbidden route or protected split is enabled")
    auth = config["authorization"]
    if auth.get("hold_id") != "blur_only_full_development_evaluation-d2f43558352e4365":
        raise RuntimeError("advisor authorization identity changed")
    model = config["model"]
    if model.get("name") != "Tongyi-MAI/Z-Image" or model.get("license") != "apache-2.0":
        raise RuntimeError("the undistilled official Z-Image foundation is required")
    controlnet = model["controlnet"]
    if controlnet.get("trainable") is not False or float(controlnet.get("control_scale", -1)) != 1.0:
        raise RuntimeError("the first gate requires the frozen full-strength LQ ControlNet")
    impl = config["implementation"]
    if impl.get("conditioning") != "frozen_nondistilled_controlnet_with_full_color_lq_latent":
        raise RuntimeError("teacher conditioning contract changed")
    training = config["training"]
    distributed = config["distributed"]
    if int(distributed.get("required_world_size", 0)) != 8:
        raise RuntimeError("teacher gate requires eight ranks")
    effective = (
        int(distributed["required_world_size"])
        * int(training["micro_batch_size_per_rank"])
        * int(training["gradient_accumulation_steps"])
    )
    if effective != int(training["global_batch_size"]):
        raise RuntimeError("global batch arithmetic changed")
    if int(training["max_steps"]) != 400 or training["save_steps"] != [200, 400]:
        raise RuntimeError("tiny-overfit step schedule changed")
    inference = config["inference"]
    if int(inference["teacher_steps"]) != 50 or float(inference["cfg_scale"]) != 4.0:
        raise RuntimeError("undistilled teacher inference protocol changed")
    if inference["evaluate_steps"] != [0, 400]:
        raise RuntimeError("tiny evaluation must compare initialization and the final checkpoint")


def load_frozen_rows(config: Mapping[str, object], *, verify_images: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = config["data"]
    manifest = ROOT / str(data["frozen_manifest"])
    if sha256(manifest) != data["frozen_manifest_sha256"]:
        raise RuntimeError("frozen sanity manifest identity changed")
    source = ROOT / str(data["source_manifest"])
    if sha256(source) != data["source_manifest_sha256"]:
        raise RuntimeError("FoundIR source manifest identity changed")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["role"] == data["tiny_role"]]
    visual = [row for row in rows if row["role"] == data["visual_role"]]
    expected = (
        (train, int(data["tiny_records"]), int(data["tiny_source_groups"]), "train"),
        (visual, int(data["visual_records"]), int(data["visual_source_groups"]), "validation"),
    )
    for selected, count, groups, split in expected:
        if len(selected) != count or len({row["source_group"] for row in selected}) != groups:
            raise RuntimeError(f"frozen {split} selection changed")
        if {row["split"] for row in selected} != {split}:
            raise RuntimeError(f"frozen {split} split label changed")
        if any(row["category"] != "01Blur" or row["composition_size"] != "1" for row in selected):
            raise RuntimeError("non-isolated Blur record entered the gate")
    if {row["source_group"] for row in train} & {row["source_group"] for row in visual}:
        raise RuntimeError("training and visual-development source groups overlap")
    if verify_images:
        for row in rows:
            for field, digest_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
                path = ROOT / row[field]
                if not path.is_file() or sha256(path) != row[digest_field]:
                    raise RuntimeError(f"image identity changed: {row['record_id']} {field}")
            load_aligned_pair(row, int(data["crop_size"]))
    return train, visual


def load_aligned_pair(row: Mapping[str, str], crop_size: int) -> tuple[Image.Image, Image.Image]:
    lq_path, gt_path = ROOT / row["lq_path"], ROOT / row["gt_path"]
    with Image.open(lq_path) as image:
        lq = image.convert("RGB")
    with Image.open(gt_path) as image:
        gt = image.convert("RGB")
    if lq.size != gt.size or lq.size != (int(row["width"]), int(row["height"])):
        raise RuntimeError(f"unaligned pair geometry: {row['record_id']}")
    x, y = int(row["crop_x"]), int(row["crop_y"])
    if x < 0 or y < 0 or x + crop_size > lq.width or y + crop_size > lq.height:
        raise RuntimeError(f"invalid frozen crop: {row['record_id']}")
    box = (x, y, x + crop_size, y + crop_size)
    return lq.crop(box), gt.crop(box)


def model_paths(config: Mapping[str, object]) -> dict[str, object]:
    model = config["model"]
    root = ROOT / str(model["root"])
    transformer = sorted(root.glob(str(model["transformer_glob"])))
    text_encoder = sorted(root.glob(str(model["text_encoder_glob"])))
    return {
        "transformer": transformer,
        "text_encoder": text_encoder,
        "vae": root / str(model["vae"]),
        "tokenizer": root / str(model["tokenizer"]),
        "controlnet": ROOT / str(model["controlnet"]["path"]),
    }


def verify_model_artifacts(config: Mapping[str, object], *, large_hashes: bool) -> dict[str, object]:
    paths = model_paths(config)
    if len(paths["transformer"]) != 2 or len(paths["text_encoder"]) != 3:
        raise RuntimeError("Z-Image shard count is incomplete")
    required = [*paths["transformer"], *paths["text_encoder"], paths["vae"], paths["controlnet"]]
    if not paths["tokenizer"].is_dir() or any(not path.is_file() for path in required):
        raise RuntimeError("Z-Image runtime artifacts are incomplete")
    checked: dict[str, str] = {}
    if large_hashes:
        model = config["model"]
        for relative, expected in model["expected_blob_sha256"].items():
            path = ROOT / str(model["root"]) / relative
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"Z-Image blob identity changed: {relative}")
            checked[relative] = actual
        control = paths["controlnet"]
        actual = sha256(control)
        if actual != model["controlnet"]["sha256"]:
            raise RuntimeError("Z-Image ControlNet identity changed")
        checked[str(control.relative_to(ROOT))] = actual
    return {"files": len(required), "large_hashes": large_hashes, "sha256": checked}


def validate_gpu_inventory(names: Sequence[str], memory_mib: Sequence[int], config: Mapping[str, object]) -> str:
    required = int(config["distributed"]["required_world_size"])
    if len(names) != required or len(memory_mib) != required:
        raise RuntimeError("exactly eight visible GPUs are required")
    families = []
    for name in names:
        matches = [item for item in config["distributed"]["allowed_gpu_name_fragments"] if item in name]
        if len(matches) != 1:
            raise RuntimeError(f"unsupported GPU: {name}")
        families.append(matches[0])
    if len(set(families)) != 1:
        raise RuntimeError("mixed H100/H200 ranks are not supported")
    minimum = int(config["distributed"]["minimum_memory_gib"]) * 1024
    if any(value < minimum for value in memory_mib):
        raise RuntimeError("visible GPU memory is below the frozen minimum")
    return families[0]
