#!/usr/bin/env python3
"""Fail-closed CPU/GPU preflight for the five-type FoundIR FLUX.2 phase."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.foundir_five_type import category_counts, load_five_type_metadata


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT.resolve())
    return path


def run(config_path: Path, *, require_data: bool, skip_large_hashes: bool) -> dict[str, object]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "advisor_approved_five_type_foundir_dit_readiness":
        raise RuntimeError("five-type DiT config is not approved")
    data = config["data"]
    metadata = workspace_path(data["official_metadata"])
    if sha256(metadata) != data["official_metadata_sha256"]:
        raise RuntimeError("official FoundIR training metadata changed")
    records, all_counts = load_five_type_metadata(metadata)
    if sum(all_counts.values()) != data["expected_official_records"]:
        raise RuntimeError("official FoundIR metadata cardinality changed")
    observed = category_counts(records)
    if len(records) != data["expected_five_type_records"] or observed != data["expected_categories"]:
        raise RuntimeError("five-type category contract changed")

    for section, path_key, hash_key in (
        (config["qwen_condition"], "training_config", "training_config_sha256"),
        (config["qwen_condition"], "adapter", "adapter_sha256"),
        (config["qwen_condition"], "validation_result", "validation_result_sha256"),
        (config["model"], "flow_weights", "flow_weights_sha256"),
        (config["model"], "autoencoder_weights", "autoencoder_weights_sha256"),
    ):
        path = workspace_path(section[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        if not skip_large_hashes or path.stat().st_size < 1024**3:
            if sha256(path) != section[hash_key]:
                raise RuntimeError(f"frozen artifact changed: {path}")

    distributed = config["distributed"]
    flow = config["flow_training"]
    if (
        distributed.get("required_world_size") != 8
        or distributed.get("micro_batch_size_per_rank") != 1
        or int(distributed.get("global_batch_size", -1))
        != 8 * int(distributed.get("gradient_accumulation_steps", -1))
        or config["lora"].get("rank") != 32
        or float(config["lora"].get("alpha", -1)) != 32.0
        or flow.get("objective") != "degraded_anchored_residual_flow"
        or not 0.0 <= float(flow.get("anchor_noise_scale", -1)) <= 0.25
    ):
        raise RuntimeError("eight-GPU FLUX training contract changed")

    training_root = workspace_path(data["training_root"])
    missing_examples: list[str] = []
    if training_root.is_dir():
        for record in records:
            for relative in (record.gt_relative, record.lq_relative):
                path = training_root / relative
                if not path.is_file():
                    missing_examples.append(relative)
                    if len(missing_examples) >= 20:
                        break
            if len(missing_examples) >= 20:
                break
    elif require_data:
        raise FileNotFoundError(
            "FoundIR official training root is missing: "
            f"{training_root}. Obtain the authorized ~4.85 TB training archive "
            "from the official FoundIR request form and extract it at this exact path."
        )
    if require_data and missing_examples:
        raise FileNotFoundError(f"FoundIR training pairs are incomplete: {missing_examples}")

    result = {
        "status": "pass" if training_root.is_dir() and not missing_examples else "metadata_only_pass_data_missing",
        "official_metadata_records": sum(all_counts.values()),
        "five_type_records": len(records),
        "category_counts": observed,
        "training_root": str(training_root),
        "training_root_present": training_root.is_dir(),
        "qwen_method": config["qwen_condition"]["method"],
        "lora_rank": config["lora"]["rank"],
        "lora_alpha": config["lora"]["alpha"],
        "protected_foundir_test_accessed": False,
    }
    print("FOUNDIR_FIVE_TYPE_PREFLIGHT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/foundir_five_type_flux2_dit_v1.json",
    )
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--skip-large-hashes", action="store_true")
    args = parser.parse_args()
    run(args.config, require_data=args.require_data, skip_large_hashes=args.skip_large_hashes)


if __name__ == "__main__":
    main()
