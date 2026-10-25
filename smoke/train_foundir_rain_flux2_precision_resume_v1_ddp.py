#!/usr/bin/env python3
"""True Rain FLUX continuation: restore step-3000 LoRA, Adam and absolute data position."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))
from smoke import train_foundir_rain_flux2_precision_fit_v1_ddp as fit_impl
from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer, promote_lora_to_fp32, restore_endpoint
from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, save_lora_checkpoint

old = fit_impl.old
digest, write_json = fit_impl.digest, fit_impl.write_json
training_losses, evaluate = fit_impl.training_losses, fit_impl.evaluate


from smoke import train_foundir_rain_flux2_precision_full_v1_ddp as full_impl
from lowlight_restoration.rain_precision_resume import RainResumeDataset, restore_adam, validate_adam_payload

CONFIG_SHA256 = "962ed5663e49ca59827f28f37260a75229085f8b49cb9e63ba03748a922638b9"


def source_hashes():
    paths = [Path(__file__), ROOT / "lowlight_restoration/rain_precision_resume.py",
             ROOT / "smoke/run_foundir_rain_flux2_precision_resume_v1_8gpu.sh",
             ROOT / "smoke/start_foundir_rain_flux2_precision_resume_v1_tmux.sh"]
    return {**full_impl.source_hashes(),
            **{str(p.relative_to(ROOT)): old.sha256(p) for p in paths}}


def read_contract(path):
    if old.sha256(path) != CONFIG_SHA256:
        raise RuntimeError("frozen continuation contract changed")
    resume = json.loads(path.read_text())
    if old.sha256(ROOT / resume["parent_config"]) != resume["parent_config_sha256"]:
        raise RuntimeError("parent configuration changed")
    config, parent = full_impl.read_contract(ROOT / resume["parent_config"])
    config["resume"] = resume
    config["initial_lora"] = resume["lora"]
    config["initial_lora_sha256"] = resume["lora_sha256"]
    config["modes"] = {"continuation": {"steps": resume["end_step"], "milestones": resume["milestones"]}}
    return config, parent


load_protocol = full_impl.load_protocol


def audit_resume(config, parent):
    resume = config["resume"]
    for key in ("parent_result", "lora", "optimizer"):
        if old.sha256(ROOT / resume[key]) != resume[key + "_sha256"]:
            raise RuntimeError(f"resume artifact changed: {key}")
    result = json.loads((ROOT / resume["parent_result"]).read_text())
    training, probe, development = load_protocol(config, parent)
    if not (result["status"] == "complete" and result["engineering_pass"]
            and result["mode"] == "formal" and result["steps"] == 3000
            and result["sample_presentations"] == 96000
            and result["config_sha256"] == resume["parent_config_sha256"]
            and result["source_sha256"] == full_impl.source_hashes()
            and result["training_manifest_sha256"] == digest(training)
            and result["fit_manifest_sha256"] == digest(probe)
            and not result["official_foundir_test_accessed"]
            and not result["sealed_internal_accessed"]):
        raise RuntimeError("parent completion/source/data identity mismatch")
    # Reconstruct the ORIGINAL registration order without allocating a 4B model.
    from flux2.model import Flux2, Klein4BParams
    with torch.device("meta"):
        model = Flux2(Klein4BParams())
        model.requires_grad_(False)
        names = inject_lora(model, parent["lora"]["target_patterns"], rank=32, alpha=32, dropout=0)
    ordered = [(name, p.shape) for name, p in model.named_parameters() if p.requires_grad]
    from safetensors import safe_open
    with safe_open(ROOT / resume["lora"], framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if (len(names) != 50 or len(ordered) != 100 or set(handle.keys()) != {n for n, _ in ordered}
                or metadata["step"] != "3000"
                or metadata["config_sha256"] != resume["parent_config_sha256"]):
            raise RuntimeError("LoRA identity/order mismatch")
        for name, shape in ordered:
            tensor = handle.get_tensor(name)
            if tensor.shape != shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("invalid saved LoRA tensor")
    payload = torch.load(ROOT / resume["optimizer"], map_location="cpu", weights_only=True)
    validate_adam_payload(payload, config_sha256=resume["parent_config_sha256"],
                          step=resume["start_step"], shapes=[shape for _, shape in ordered])
    return {"status": "rain_precision_resume_state_pass", "start_step": 3000,
            "end_step": 4000, "adam_parameters": len(ordered), "adam_step": 3000,
            "learning_rate": 1e-5, "first_sample_position": 96000,
            "last_sample_position": 127999, "parameter_order": [name for name, _ in ordered],
            "protected_pixels_accessed": False, "saved_rng_available": False,
            "bitwise_gpu_equivalence_claimed": False}


def preflight(config_path):
    config, parent = read_contract(config_path)
    receipt = full_impl.preflight(ROOT / config["resume"]["parent_config"])
    resume_receipt = audit_resume(config, parent)
    receipt.update(resume_receipt)
    receipt["config_sha256"] = old.sha256(config_path)
    print("RAIN_RESUME_PREFLIGHT " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/foundir_rain_flux2_precision_resume_v1.json")
    parser.add_argument("--mode", choices=("continuation",), default="continuation")
    parser.add_argument("--run-id", default="continue_v1")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.preflight:
        preflight(config_path)
        return
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise RuntimeError("unsafe run ID")
    config, parent = read_contract(config_path)
    training, fit, development = load_protocol(config, parent)
    fit_sha256 = digest(fit)
    training_sha256 = digest(training)
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8 or torch.cuda.device_count() != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", "8")) != 8:
        raise RuntimeError("requires one homogeneous 8-GPU node")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 75 * 1024**3 or not any(n in properties.name for n in ("H100", "H200")):
        raise RuntimeError("requires H100/H200 with at least 75 GiB")
    dist.init_process_group("nccl", device_id=device)
    output = ROOT / "checkpoints/foundir_rain_flux2_precision_resume_v1" / args.run_id
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
                    "fit_records": fit, "training_manifest_sha256": training_sha256,
                    "train_records": len(training), "development_manifest_sha256": digest(development),
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
            print("RAIN_RESUME_MODEL " + json.dumps({"trainable_parameters": sum(p.numel() for p in trainable),
                "trainable_dtype": "float32", "compute_dtype": "bfloat16", "initial_lora": config["initial_lora"]}), flush=True)
        wrapped = DDP(model, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True)
        optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0, fused=True)
        resume = config["resume"]
        saved_optimizer = torch.load(ROOT / resume["optimizer"], map_location="cpu", weights_only=True)
        restore_adam(optimizer, saved_optimizer, config_sha256=resume["parent_config_sha256"],
                     step=resume["start_step"])
        del saved_optimizer
        assert_fp32_optimizer(optimizer)
        if rank == 0:
            print("RAIN_RESUME_STATE_RESTORED " + json.dumps({"lora_step": 3000,
                "adam_step": 3000, "first_sample_position": 96000, "learning_rate": 1e-5,
                "optimizer_state_exactly_restored": True}), flush=True)
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
        dataset = RainResumeDataset(ROOT, training,
            steps=resume["end_step"] - resume["start_step"], start_step=resume["start_step"],
            rank=rank, world=world, seed=seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
            num_workers=2, prefetch_factor=2, pin_memory=True,
            multiprocessing_context="spawn", persistent_workers=False)
        iterator = iter(loader)
        step_seconds = []
        seen_records, seen_groups = set(), set()
        sampled_updates_observed = {"A": False, "B": False}
        last_metrics = None
        for step in range(resume["start_step"], resume["end_step"]):
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            sampled_before = [p.detach().flatten()[:4096].clone() for p in trainable]
            for micro in range(4):
                batch = next(iterator)
                expected_position = step * 32 + micro * world + rank
                if int(batch["position"][0]) != expected_position:
                    raise RuntimeError("data stream ordering changed")
                seen_records.add(batch["record_id"][0])
                seen_groups.add(batch["source_group"][0])
                lq = batch["lq"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
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
            if rank == 0 and (completed == resume["start_step"] + 1 or completed % 10 == 0 or completed == stages["steps"]):
                print("RAIN_RESUME_TRAIN " + json.dumps({"step": completed, "max_steps": stages["steps"],
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
                                    "optimizer": optimizer.state_dict(),
                                    "parameter_names": [n for n, p in model.named_parameters() if p.requires_grad],
                                    "next_sample_position": completed * 32,
                                    "resume_parent_config_sha256": resume["parent_config_sha256"]}, handle)
                dist.barrier()
                last_metrics = evaluate_both(completed)
        seen = [None] * world
        dist.all_gather_object(seen, (seen_records, seen_groups))
        if rank == 0:
            if not all(sampled_updates_observed.values()):
                raise RuntimeError("no observed updates in a LoRA factor; inspect gradients before advancing")
            elapsed = time.monotonic() - started
            checks = {split: entry["psnr"] >= 34.2 and entry["lpips"] <= .08
                      for split, entry in last_metrics.items()}
            payload = {"status": "complete", "mode": args.mode, "engineering_pass": True,
                "config_sha256": old.sha256(config_path), "fit_manifest_sha256": fit_sha256,
                "source_sha256": run_source_hashes,
                "steps": stages["steps"], "start_step": resume["start_step"],
                "additional_steps": resume["end_step"] - resume["start_step"],
                "initial_lora_sha256": config["initial_lora_sha256"],
                "resume_optimizer_sha256": resume["optimizer_sha256"],
                "resume_parent_result_sha256": resume["parent_result_sha256"],
                "optimizer_state_restored": True, "data_position_restored": True,
                "saved_rng_available": False, "bitwise_gpu_equivalence_claimed": False,
                "train_fit_meets_numeric_target": checks["train_fit"],
                "development_gate_pass": checks["development"],
                "qualification_is_engineering_only": args.mode == "qualification",
                "train_probe_is_not_generalization_evidence": True,
                "training_manifest_sha256": training_sha256,
                "training_records": len(training),
                "unique_records_seen": len(set.union(*(x[0] for x in seen))),
                "unique_sources_seen": len(set.union(*(x[1] for x in seen))),
                "sample_presentations": (resume["end_step"] - resume["start_step"]) * 32,
                "cumulative_sample_presentations": resume["end_step"] * 32,
                "unique_counts_scope": "continuation_only",
                "first_sample_position": resume["first_sample_position"],
                "last_sample_position": resume["last_sample_position"],
                "history": history, "wall_clock_hours": elapsed / 3600,
                "allocated_gpu_hours": elapsed * world / 3600, "gpu_name": properties.name,
                "median_training_step_seconds": float(np.median(step_seconds[1:])),
                "sampled_updates_observed": sampled_updates_observed,
                "peak_allocated_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "official_foundir_test_accessed": False, "sealed_internal_accessed": False,
                "automatic_formal_launch": False}
            write_json(output / "result.json", payload)
            print("RAIN_RESUME_END " + json.dumps({k: v for k, v in payload.items() if k != "history"}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
