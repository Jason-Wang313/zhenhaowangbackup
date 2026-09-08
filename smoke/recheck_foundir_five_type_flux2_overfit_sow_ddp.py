#!/usr/bin/env python3
"""Re-evaluate the immutable v3 overfit checkpoint with SOW PSNR and LPIPS."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import inject_lora, load_restoration_checkpoint  # noqa: E402
from lowlight_restoration.pixel_refinement import (  # noqa: E402
    BoundedPixelRefiner,
    pixel_refiner_parameter_count,
)
from smoke.train_foundir_five_type_flux2_overfit_ddp import (  # noqa: E402
    aggregate_sow_overfit_metrics,
    encode_prompts,
    evaluate_arms,
    load_conditions,
    prepare_crops,
    sha256,
    sow_overfit_checks,
    write_once,
)


def validate_historical_source(config: dict[str, object]) -> tuple[Path, dict[str, object]]:
    overfit = config["overfit"]
    source = overfit["historical_recheck"]
    checkpoint = ROOT / str(source["checkpoint"])
    source_result_path = ROOT / str(source["source_result"])
    if sha256(checkpoint) != source["checkpoint_sha256"]:
        raise RuntimeError("historical v3 overfit checkpoint changed")
    if sha256(source_result_path) != source["source_result_sha256"]:
        raise RuntimeError("historical v3 overfit result changed")
    metadata = json.loads(checkpoint.with_suffix(checkpoint.suffix + ".json").read_text())
    if (
        metadata.get("config_sha256") != source["source_config_sha256"]
        or int(metadata.get("step", -1)) != int(overfit["steps"])
    ):
        raise RuntimeError("historical v3 checkpoint metadata changed")
    result = json.loads(source_result_path.read_text())
    if (
        result.get("config_sha256") != source["source_config_sha256"]
        or result.get("protected_foundir_test_accessed") is not False
        or not isinstance(result.get("probe_loss_ratio"), (int, float))
    ):
        raise RuntimeError("historical v3 overfit evidence is invalid")
    return checkpoint, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = json.loads(args.config.read_text())
    distributed = config["distributed"]
    overfit = config["overfit"]
    flow = config["flow_training"]
    refinement = config["pixel_refinement"]
    checkpoint, historical_result = validate_historical_source(config)
    conditions_path = ROOT / str(overfit["conditions_csv"])
    conditions_summary = json.loads((ROOT / str(overfit["conditions_json"])).read_text())
    if (
        conditions_summary.get("config_sha256") != sha256(args.config)
        or conditions_summary.get("conditions_csv_sha256") != sha256(conditions_path)
    ):
        raise RuntimeError("Qwen condition cache identity changed")
    rows = load_conditions(conditions_path, int(overfit["records"]))
    output_dir = (
        ROOT / str(config["runtime"]["output_dir"]) / str(overfit["output_subdirectory"])
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(distributed["backend"]), device_id=device)
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        if world != int(distributed["required_world_size"]) or torch.cuda.device_count() != world:
            raise RuntimeError("SOW overfit recheck requires exactly eight visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory < int(distributed["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError("GPU memory is below the H100/H200 contract")
        if not any(name in properties.name for name in ("H100", "H200")):
            raise RuntimeError(f"unsupported GPU: {properties.name}")

        seed = int(flow["seed"])
        random.seed(seed + rank)
        np.random.seed(seed + rank)
        torch.manual_seed(seed + rank)
        torch.cuda.manual_seed_all(seed + rank)
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / str(config["model"]["flow_weights"])),
            AE_MODEL_PATH=str(ROOT / str(config["model"]["autoencoder_weights"])),
            QWEN3_MODEL_PATH=str(ROOT / str(config["model"]["text_encoder"])),
        )
        configured_arms = tuple(config["evaluation"].get("condition_arms", ("qwen",)))
        prompt_keys = {
            "qwen": "qwen_text",
            "generic": "generic_text",
            "type_oracle": "type_oracle_text",
            "shuffled_wrong": "qwen_text",
        }
        prompts = [str(row[prompt_keys[arm]]) for row in rows for arm in configured_arms]
        contexts = encode_prompts(config, prompts, device)
        records = prepare_crops(rows, crop_size=int(config["data"]["crop_size"]), seed=seed)

        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(config["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(config["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model,
            list(config["lora"]["target_patterns"]),
            rank=int(config["lora"]["rank"]),
            alpha=float(config["lora"]["alpha"]),
            dropout=float(config["lora"]["dropout"]),
        )
        if len(names) != int(config["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX LoRA target count changed")
        pixel_refiner = BoundedPixelRefiner(
            width=int(refinement["width"]),
            blocks=int(refinement["blocks"]),
            maximum_residual_m11=float(refinement["maximum_residual_m11"]),
        ).to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        if pixel_refiner_parameter_count(pixel_refiner) != int(
            refinement["expected_trainable_parameters"]
        ):
            raise RuntimeError("pixel-refiner parameter count changed")
        load_restoration_checkpoint(model, pixel_refiner, checkpoint)
        metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device).eval()
        local_rows = evaluate_arms(
            model,
            autoencoder,
            records,
            contexts,
            batched_prc_img,
            config,
            rank=rank,
            world=world,
            device=device,
            pixel_refiner=pixel_refiner,
            lpips_metric=metric,
        )
        gathered: list[object | None] = [None] * world
        dist.all_gather_object(gathered, local_rows)
        if rank == 0:
            result_rows = [row for part in gathered if isinstance(part, list) for row in part]
            arm_metrics = aggregate_sow_overfit_metrics(result_rows, configured_arms)
            loss_ratio = float(historical_result["probe_loss_ratio"])
            checks = sow_overfit_checks(
                loss_ratio=loss_ratio, arm_metrics=arm_metrics, overfit=overfit
            )
            payload = {
                "schema_version": 1,
                "status": "pass" if all(checks.values()) else "failed",
                "automatic_gate_pass": all(checks.values()),
                "metric_contract": "foundir_sow_psnr_lpips_only",
                "checks": checks,
                "config_sha256": sha256(args.config),
                "conditions_sha256": sha256(conditions_path),
                "source_checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "source_checkpoint_sha256": sha256(checkpoint),
                "source_result": str(overfit["historical_recheck"]["source_result"]),
                "source_result_sha256": str(
                    overfit["historical_recheck"]["source_result_sha256"]
                ),
                "source_probe_loss_ratio": loss_ratio,
                "arm_metrics": arm_metrics,
                "evaluation_rows": result_rows,
                "world_size": world,
                "gpu_name": properties.name,
                "protected_foundir_test_accessed": False,
            }
            write_once(
                output_dir / "result.json",
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
            )
            print("FOUNDIR_FIVE_TYPE_SOW_RECHECK_COMPLETE " + json.dumps(payload, sort_keys=True))
            gate = 1 if payload["automatic_gate_pass"] else 0
        else:
            gate = 0
        gate_tensor = torch.tensor([gate], dtype=torch.int32, device=device)
        dist.broadcast(gate_tensor, src=0)
        dist.barrier()
        return 0 if int(gate_tensor.item()) == 1 else 2
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
