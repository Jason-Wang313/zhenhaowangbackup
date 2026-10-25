#!/usr/bin/env python3
"""Rain FLUX FP32-state qualification and fixed-training-set fit diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))
from lowlight_restoration.flux2_endpoint_precision import (  # noqa: E402
    assert_fp32_optimizer, promote_lora_to_fp32, restore_endpoint,
)
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, save_lora_checkpoint  # noqa: E402
from lowlight_restoration.pixel_refinement import spatial_highpass  # noqa: E402
from smoke import train_foundir_rain_flux2_direct_endpoint_distillation_v1_ddp as old  # noqa: E402


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes():
    paths = [Path(__file__), ROOT / "lowlight_restoration/flux2_endpoint_precision.py",
             ROOT / "lowlight_restoration/lora.py", ROOT / "lowlight_restoration/pixel_refinement.py",
             ROOT / "lowlight_restoration/task_native_restormer.py", Path(old.__file__),
             ROOT / "smoke/train_foundir_blur_only_flux2_overfit_ddp.py"]
    return {str(path.relative_to(ROOT)): old.sha256(path) for path in paths}


def read_contract(path: Path):
    config = json.loads(path.read_text())
    if config["status"] != "user_authorized_rain_precision_and_fit_diagnostic":
        raise RuntimeError("unexpected precision experiment")
    parent_path = ROOT / config["parent_config"]
    if old.sha256(parent_path) != config["parent_config_sha256"]:
        raise RuntimeError("parent objective/config changed")
    parent = json.loads(parent_path.read_text())
    old.validate_config(parent, parent_path)
    checks = [config["fit_sources"] == 32, config["crop_size"] == 512,
              config["initial_lora"] == "checkpoints/foundir_rain_flux2_direct_endpoint_distillation_formal_v1/step_002000.safetensors",
              config["initial_lora_sha256"] == "eff05dccc0d9bcdf538305ac4095304348a876eb6cb3a4ded3ea27767901f36a",
              config["selection"] == "sha256_rain_precision_fit_v1_source_then_record",
              config["modes"] == {"qualification": {"steps": 25, "milestones": [0, 25]},
                                   "fit": {"steps": 300, "milestones": [0, 100, 300]}},
              config["training"] == {
                  "seed": 2026102507, "global_batch_size": 32, "micro_batch_size_per_rank": 1,
                  "gradient_accumulation_steps": 4, "learning_rate": 1e-5, "weight_decay": 0.0,
                  "max_grad_norm": 1.0, "augmentation": False,
                  "losses": "unchanged_parent_coefficients_and_terms",
                  "trainable_dtype": "float32", "optimizer_state_dtype": "float32",
                  "compute_dtype": "bfloat16", "endpoint_subtraction_dtype": "float32"},
              config["evaluation"] == {"train_records": 32, "development_records": 100,
                  "legacy_baseline_at_step_zero": True, "psnr_minimum": 34.2,
                  "lpips_maximum": 0.08, "fit_is_not_generalization_evidence": True},
              config["distributed"] == {"world_size": 8, "allowed_gpu_families": ["H100", "H200"],
                                         "minimum_memory_gib": 75}]
    checks.extend(config[key] is False for key in (
        "automatic_formal_launch", "official_foundir_test_accessed", "sealed_internal_accessed", "pixel_refiner"))
    if not all(checks):
        raise RuntimeError("frozen precision diagnostic contract changed")
    return config, parent


def select_fit_rows(rows, count=32, crop=512):
    groups = {}
    for row in rows:
        if row["role"] != "direct_endpoint_train" or row["split"] != "train":
            raise RuntimeError("fit selection may only use training rows")
        if min(int(row["width"]), int(row["height"])) >= crop:
            groups.setdefault(row["source_group"], []).append(row)
    key = lambda text: hashlib.sha256(("rain_precision_fit_v1:" + text).encode()).hexdigest()
    if len(groups) < count:
        raise RuntimeError("insufficient eligible training sources")
    selected = []
    for group in sorted(groups, key=key)[:count]:
        row = dict(min(groups[group], key=lambda item: key(item["record_id"])))
        row["crop_x"] = str((int(row["width"]) - crop) // 2)
        row["crop_y"] = str((int(row["height"]) - crop) // 2)
        selected.append(row)
    return selected


def load_protocol(config, parent):
    training, development = old.load_rows(parent)
    fit = select_fit_rows(training, config["fit_sources"], config["crop_size"])
    if {r["source_group"] for r in fit} & {r["source_group"] for r in development}:
        raise RuntimeError("fit/development source overlap")
    for row in fit + development:
        if (int(row["crop_x"]) + 512 > int(row["width"])
                or int(row["crop_y"]) + 512 > int(row["height"])):
            raise RuntimeError("512 crop leaves image bounds")
        for field in ("lq_path", "gt_path"):
            if not (ROOT / row[field]).is_file():
                raise RuntimeError(f"missing image for {row['record_id']}")
    return fit, development


def write_json(path, payload):
    old.write_once(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def preflight(config_path):
    config, parent = read_contract(config_path)
    old.verify_artifacts(parent)
    if old.sha256(ROOT / config["initial_lora"]) != config["initial_lora_sha256"]:
        raise RuntimeError("initial step-2000 LoRA changed")
    fit, development = load_protocol(config, parent)
    def verify_pair(row):
        for field in ("lq", "gt"):
            if old.sha256(ROOT / row[field + "_path"]) != row[field + "_sha256"]:
                raise RuntimeError(f"image identity changed: {row['record_id']} {field}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_pair, fit + development))
    receipt = {"status": "rain_precision_fit_preflight_pass", "config_sha256": old.sha256(config_path),
               "fit_manifest_sha256": digest(fit), "fit_records": len(fit),
               "development_manifest_sha256": digest(development), "development_records": len(development),
               "crop_size": 512, "protected_pixels_accessed": False}
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def training_losses(result, clean, lq, gt, teacher_output, metric, pack_image, weights):
    decoded = result["decoded"]
    latent_target, _ = pack_image(result["degraded"] - clean)
    change_mask = ((gt - lq).abs().mean(1, keepdim=True) / weights["rain_change_scale_m11"]).clamp(0, 1).detach()
    change_weight = 1 + weights["rain_change_weight"] * change_mask
    epsilon = weights["charbonnier_epsilon"]
    terms = {
        "flow_mse": F.mse_loss(result["prediction"].float(), latent_target.float()),
        "predicted_clean_latent_l1": F.l1_loss(result["predicted_clean"], clean),
        "final_rgb_mse": ((decoded - gt).square() * change_weight).mean(),
        "rgb_charbonnier": (torch.sqrt((decoded - gt).square() + epsilon**2) * change_weight).mean(),
        "highpass_l1": F.l1_loss(spatial_highpass(decoded), spatial_highpass(gt)),
        "lpips": metric((decoded + 1) * .5, (gt + 1) * .5),
        "teacher_l1": F.l1_loss((decoded + 1) * .5, teacher_output),
        "unchanged_identity": (torch.sqrt((decoded - lq).square() + epsilon**2) * (1 - change_mask)).mean(),
    }
    total = sum(value * weights[name + "_weight"] for name, value in terms.items())
    return total, terms


def evaluate(model, ae, rows, context, context_ids, pack_image, metric, *, rank, world, device,
             output_dir, split, step, legacy=False):
    local = []
    model.eval()
    for index in range(rank, len(rows), world):
        record = rows[index]
        lq_cpu, gt_cpu = old.materialize(record, 512)
        lq, gt = lq_cpu[None].to(device), gt_cpu[None].to(device)
        with torch.inference_mode():
            if legacy:
                output = old.direct_endpoint(model, ae, lq, context, context_ids, pack_image, 10)
            else:
                output = restore_endpoint(model, ae, lq, context, context_ids, pack_image)["decoded"]
            if not torch.isfinite(output).all():
                raise RuntimeError(f"nonfinite evaluation output: {record['record_id']}")
            output_q = ((output + 1) * 127.5).round().clamp(0, 255) / 127.5 - 1
            local.append({"record_id": record["record_id"], "source_group": record["source_group"],
                          "input_psnr": old.psnr(gt, lq), "output_psnr": old.psnr(gt, output_q),
                          "input_lpips": old.lpips_value(metric, gt, lq),
                          "output_lpips": old.lpips_value(metric, gt, output_q)})
        if not legacy and index < 8:
            old.save_triptych(output_dir / "images" / split / f"step_{step:06d}" / f"{index:03d}.png",
                              record["record_id"], lq, output_q, gt)
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    model.train()
    if rank != 0:
        return None
    combined = sorted([r for shard in gathered for r in shard], key=lambda r: r["record_id"])
    if len(combined) != len(rows) or len({r["record_id"] for r in combined}) != len(rows):
        raise RuntimeError("metric gather lost/duplicated records")
    if any(not math.isfinite(row[key]) for row in combined
           for key in ("input_psnr", "output_psnr", "input_lpips", "output_lpips")):
        raise RuntimeError("nonfinite evaluation metric")
    summary = {"split": split, "step": step, "records": len(combined), "legacy_subtraction": legacy,
               "psnr": float(np.mean([r["output_psnr"] for r in combined])),
               "lpips": float(np.mean([r["output_lpips"] for r in combined])),
               "input_psnr": float(np.mean([r["input_psnr"] for r in combined])),
               "input_lpips": float(np.mean([r["input_lpips"] for r in combined])),
               "rows": combined}
    suffix = "legacy" if legacy else "shared_fp32"
    write_json(output_dir / "metrics" / f"{split}_{step:06d}_{suffix}.json", summary)
    print("RAIN_PRECISION_METRICS " + json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundir_rain_flux2_precision_fit_v1.json")
    parser.add_argument("--mode", choices=("qualification", "fit"), default="qualification")
    parser.add_argument("--run-id", default="qualification_v1")
    parser.add_argument("--qualification-result", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        preflight(config_path)
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise RuntimeError("unsafe run ID")
    config, parent = read_contract(config_path)
    fit, development = load_protocol(config, parent)
    fit_sha256 = digest(fit)
    if args.mode == "fit":
        if args.qualification_result is None:
            raise RuntimeError("fit requires a successful qualification result")
        qualification = json.loads(args.qualification_result.read_text())
        if (qualification["mode"] != "qualification" or not qualification["engineering_pass"]
                or qualification["config_sha256"] != old.sha256(config_path)
                or qualification["fit_manifest_sha256"] != digest(fit)
                or qualification["source_sha256"] != source_hashes()):
            raise RuntimeError("qualification result does not match this contract")
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or torch.cuda.device_count() != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", "8")) != 8:
        raise RuntimeError("requires one homogeneous 8-GPU node")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 75 * 1024**3 or not any(n in properties.name for n in ("H100", "H200")):
        raise RuntimeError("requires H100/H200 with at least 75 GiB")
    dist.init_process_group("nccl", device_id=device)
    output = ROOT / "checkpoints/foundir_rain_flux2_precision_fit_v1" / args.run_id
    started = time.monotonic()
    try:
        gpu_names = [None] * world
        dist.all_gather_object(gpu_names, properties.name)
        if len(set(gpu_names)) != 1:
            raise RuntimeError("GPU model must be homogeneous")
        run_source_hashes = source_hashes()
        error = [None]
        if rank == 0:
            try:
                if output.exists():
                    raise RuntimeError("run directory already exists; preserve it and inspect before retry")
                preflight(config_path)
                output.mkdir(parents=True)
                write_json(output / "receipt.json", {"config": config, "mode": args.mode,
                    "config_sha256": old.sha256(config_path), "fit_manifest_sha256": digest(fit),
                    "fit_records": fit, "development_manifest_sha256": digest(development),
                    "source_sha256": run_source_hashes})
            except Exception as exc:
                error[0] = str(exc)
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        seed = config["training"]["seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            TORCH_HOME=str(ROOT / parent["runtime"]["torch_home"]),
            KLEIN_4B_BASE_MODEL_PATH=str(ROOT / parent["model"]["flow_weights"]),
            AE_MODEL_PATH=str(ROOT / parent["model"]["autoencoder_weights"]))
        context, context_ids, _, _ = old.encode_prompt(parent, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        ae = load_ae(parent["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(model, parent["lora"]["target_patterns"], rank=32, alpha=32, dropout=0)
        if len(names) != 50:
            raise RuntimeError("unexpected LoRA targets")
        trainable = promote_lora_to_fp32(model)
        load_lora_checkpoint(model, ROOT / config["initial_lora"])
        if rank == 0:
            print("RAIN_PRECISION_MODEL " + json.dumps({"trainable_parameters": sum(p.numel() for p in trainable),
                "trainable_dtype": "float32", "compute_dtype": "bfloat16", "initial_lora": config["initial_lora"]}), flush=True)
        wrapped = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0, fused=True)
        teacher = old.load_teacher(parent, device)
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval().requires_grad_(False)
        stages = config["modes"][args.mode]
        history = []
        evaluation_args = dict(rank=rank, world=world, device=device, output_dir=output)
        def evaluate_both(step, legacy=False):
            entries = {}
            for split, rows in (("train_fit", fit), ("development", development)):
                entries[split] = evaluate(model, ae, rows, context, context_ids, batched_prc_img,
                    metric, split=split, step=step, legacy=legacy, **evaluation_args)
            if rank == 0:
                history.append(entries)
            return entries
        evaluate_both(0, legacy=True)
        evaluate_both(0)
        step_seconds = []
        sampled_updates_observed = {"A": False, "B": False}
        last_metrics = None
        for step in range(stages["steps"]):
            step_started = time.monotonic()
            order = torch.randperm(32, generator=torch.Generator().manual_seed(seed + step)).tolist()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            sampled_before = [p.detach().flatten()[:4096].clone() for p in trainable]
            for micro in range(4):
                row = fit[order[micro * world + rank]]
                lq_cpu, gt_cpu = old.materialize(row, 512)
                lq, gt = lq_cpu[None].to(device), gt_cpu[None].to(device)
                with torch.no_grad():
                    clean = ae.encode(gt.float()).float()
                    target_teacher = teacher((lq + 1) * .5).clamp(0, 1)
                # no_sync avoids four redundant reductions of the same accumulated gradient.
                with wrapped.no_sync() if micro < 3 else nullcontext():
                    result = restore_endpoint(wrapped, ae, lq, context, context_ids, batched_prc_img)
                    loss, terms = training_losses(result, clean, lq, gt, target_teacher, metric,
                                                   batched_prc_img, parent["training"]["losses"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("nonfinite loss")
                    (loss / 4).backward()
                for name, value in {"total": loss, **terms}.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(value.detach()) / 4
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if not math.isfinite(float(norm)):
                raise RuntimeError("nonfinite gradient")
            optimizer.step()
            assert_fp32_optimizer(optimizer)
            changed = {"A": [0, 0], "B": [0, 0]}
            for index, (parameter, before) in enumerate(zip(trainable, sampled_before)):
                # LoRA insertion registers A then B for each target module.
                values = changed["A" if index % 2 == 0 else "B"]
                after = parameter.detach().flatten()[:4096]
                values[0] += int((after != before).sum()); values[1] += after.numel()
            for part in sampled_updates_observed:
                sampled_updates_observed[part] |= changed[part][0] > 0
            vector = torch.tensor(list(accumulated.values()), device=device, dtype=torch.float64)
            dist.all_reduce(vector); vector /= world
            torch.cuda.synchronize()
            step_seconds.append(time.monotonic() - step_started)
            completed = step + 1
            if rank == 0 and (completed == 1 or completed % 10 == 0 or completed == stages["steps"]):
                print("RAIN_PRECISION_TRAIN " + json.dumps({"step": completed, "max_steps": stages["steps"],
                    "losses": dict(zip(accumulated, vector.tolist())), "grad_norm": float(norm),
                    "sampled_A_changed_fraction": changed["A"][0] / changed["A"][1],
                    "sampled_B_changed_fraction": changed["B"][0] / changed["B"][1],
                    "seconds": step_seconds[-1], "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)
            if completed in stages["milestones"]:
                if rank == 0:
                    path = output / f"step_{completed:06d}.safetensors"
                    save_lora_checkpoint(model, path, {"step": completed, "mode": args.mode,
                        "config_sha256": old.sha256(config_path), "trainable_dtype": "float32"})
                    from safetensors import safe_open
                    with safe_open(path, framework="pt", device="cpu") as handle:
                        if len(handle.keys()) != 100 or any(handle.get_slice(key).get_dtype() != "F32" for key in handle.keys()):
                            raise RuntimeError("saved checkpoint did not preserve FP32 LoRA weights")
                    with path.with_suffix(".optimizer.pt").open("xb") as handle:
                        torch.save({"step": completed, "config_sha256": old.sha256(config_path),
                                    "optimizer": optimizer.state_dict()}, handle)
                dist.barrier()
                last_metrics = evaluate_both(completed)
        if rank == 0:
            if not all(sampled_updates_observed.values()):
                raise RuntimeError("no observed updates in a LoRA factor; inspect gradients before advancing")
            elapsed = time.monotonic() - started
            checks = {split: entry["psnr"] >= 34.2 and entry["lpips"] <= .08
                      for split, entry in last_metrics.items()}
            payload = {"status": "complete", "mode": args.mode, "engineering_pass": True,
                "config_sha256": old.sha256(config_path), "fit_manifest_sha256": fit_sha256,
                "source_sha256": run_source_hashes,
                "steps": stages["steps"], "initial_lora_sha256": config["initial_lora_sha256"],
                "train_fit_meets_numeric_target": checks["train_fit"],
                "development_gate_pass": checks["development"],
                "qualification_is_engineering_only": args.mode == "qualification",
                "fit_is_not_generalization_evidence": True,
                "history": history, "wall_clock_hours": elapsed / 3600,
                "allocated_gpu_hours": elapsed * world / 3600, "gpu_name": properties.name,
                "median_training_step_seconds": float(np.median(step_seconds[1:])),
                "sampled_updates_observed": sampled_updates_observed,
                "peak_allocated_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "official_foundir_test_accessed": False, "sealed_internal_accessed": False,
                "automatic_formal_launch": False}
            write_json(output / "result.json", payload)
            print("RAIN_PRECISION_END " + json.dumps({k: v for k, v in payload.items() if k != "history"}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
