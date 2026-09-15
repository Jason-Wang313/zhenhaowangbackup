#!/usr/bin/env python3
"""Evaluate predeclared full-scale Z-Image teacher candidates on all Blur development records."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

import torch
import torch.distributed as dist
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from lowlight_restoration.zimage_conditional_teacher import ConditionalTeacher, PROTECTED, fixed_seed, load_settings
from lowlight_restoration.foundir_zimage_paired_flow import sha256, validate_gpu_inventory, verify_model_artifacts
from lowlight_restoration.zimage_repro_batch import install_flash_control, seed_everything
from lowlight_restoration.zimage_padding_compat import build_padding_function
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import aggregate, lpips_value, psnr
from smoke.train_zimage_conditional_teacher import (
    check_rank_weights,
    encode,
    exact_load,
    gather,
    load_full_development,
    triptych,
    validate_admission,
    write_json,
)


def validate_training_result(settings, digest, root):
    path = root / "result.json"
    result = json.loads(path.read_text())
    c = settings["stages"]["full"]
    if (result.get("config_sha256"), result.get("stage"), result.get("steps")) != (digest, "full", c["steps"]):
        raise ValueError("full training result identity changed")
    if any(result.get(k) is not False for k in PROTECTED):
        raise ValueError("full training protected scope changed")
    candidates = result.get("full_development_candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("full candidate selection missing/out of bounds")
    steps = [x["step"] for x in candidates]
    if steps != sorted(set(steps)) or c["steps"] not in steps:
        raise ValueError("full candidate steps changed")
    for candidate in candidates:
        checkpoint = ROOT / candidate["checkpoint"]
        if checkpoint.parent != root or sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise ValueError("full candidate checkpoint identity changed")
    return result, candidates


def candidate_gate(metrics, settings):
    gate = settings["stages"]["full"]["gate"]
    checks = dict(
        development_psnr_gain=metrics["psnr_gain_db"] > gate["development_psnr_gain_db"],
        development_lpips_gain=metrics["lpips_improvement"] > gate["development_lpips_improvement"],
        sow_psnr=metrics["output_psnr"] >= gate["sow_psnr_min"],
        sow_lpips=metrics["output_lpips"] <= gate["sow_lpips_max"],
    )
    return checks, all(checks.values())


def selection_score(entry, settings):
    gate, metrics = settings["stages"]["full"]["gate"], entry["metrics"]
    psnr_deficit = max(0.0, gate["sow_psnr_min"] - metrics["output_psnr"]) / gate["sow_psnr_min"]
    lpips_excess = max(0.0, metrics["output_lpips"] - gate["sow_lpips_max"]) / gate["sow_lpips_max"]
    return max(psnr_deficit, lpips_excess), -metrics["output_psnr"], metrics["output_lpips"], -entry["step"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-authorization", type=Path, required=True)
    args = parser.parse_args()
    settings, base = load_settings(args.config)
    digest = sha256(args.config)
    _, admission = validate_admission(settings, digest, "full", full_path=args.full_authorization)
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise ValueError("full development evaluation requires one eight-GPU instance")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
    hardware = validate_gpu_inventory([p.name for p in props], [p.total_memory // 2**20 for p in props], base)
    dist.init_process_group("nccl", device_id=device)
    seed_everything(settings["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    install_flash_control(True)
    from diffsynth.pipelines import z_image
    build_padding_function(z_image)
    verify_model_artifacts(base, large_hashes=rank == 0)
    dist.barrier()

    root = ROOT / settings["stages"]["full"]["output_dir"]
    training_result, candidates = validate_training_result(settings, digest, root)
    model = ConditionalTeacher(base, settings, device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named) != 477 or sum(p.numel() for _, p in named) != 79626240:
        raise RuntimeError("unexpected full evaluator trainable geometry")
    with torch.no_grad():
        prompt = model.pipe.units[1].process(model.pipe, base["model"]["prompt"], None)["prompt_embeds"]
    rows = load_full_development(settings, base)
    local_items = [encode(model, row, settings["crop_size"]) for row in rows[rank::world]]
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval().requires_grad_(False)
    evaluated = []
    for candidate in candidates:
        checkpoint = ROOT / candidate["checkpoint"]
        exact_load(model, checkpoint)
        check_rank_weights(named, world)
        local = []
        for index, item in enumerate(local_items, 1):
            rid = item["row"]["record_id"]
            with torch.no_grad():
                latent = model.restore(item["lq_latent"], prompt,
                                       fixed_seed(settings["seed"], "full_development", rid),
                                       settings["teacher_steps"], settings["inference_shift"])
                rgb = model.pipe.vae_decoder(latent.to(model.pipe.torch_dtype))
                image = model.pipe.vae_output_to_image(rgb).convert("RGB")
            record = dict(record_id=rid, source_group=item["row"]["source_group"],
                          input_psnr=psnr(item["gt"], item["lq"]), output_psnr=psnr(item["gt"], image),
                          input_lpips=lpips_value(metric, item["gt"], item["lq"], device),
                          output_lpips=lpips_value(metric, item["gt"], image, device),
                          latent_mse=float((latent - item["gt_latent"].float()).square().mean()))
            if any(not math.isfinite(v) for v in record.values() if isinstance(v, (float, int))):
                raise RuntimeError("nonfinite full development metric")
            folder = root / f"step_{candidate['step']:06d}" / "full_development"
            triptych(folder / f"{rid}.png", item, image)
            write_json(folder / f"{rid}.json", dict(record, config_sha256=digest,
                       checkpoint_sha256=candidate["checkpoint_sha256"], triptych_sha256=sha256(folder / f"{rid}.png"),
                       sealed_internal_accessed=False, official_foundir_test_accessed=False,
                       development_gt_optimization=False))
            local.append(record)
            metric.reset()
            if index % 25 == 0:
                print("CONDITIONAL_FULL_EVAL_PROGRESS", candidate["step"], rank, index, len(local_items), flush=True)
        all_rows = sorted(gather(local, world), key=lambda x: x["record_id"])
        if len(all_rows) != len(rows) or {x["record_id"] for x in all_rows} != {x["record_id"] for x in rows}:
            raise RuntimeError("full development record coverage mismatch")
        metrics = aggregate(all_rows)
        checks, passed = candidate_gate(metrics, settings)
        entry = dict(step=candidate["step"], checkpoint=candidate["checkpoint"],
                     checkpoint_sha256=candidate["checkpoint_sha256"], metrics=metrics,
                     checks=checks, automatic_metric_gate_pass=passed)
        evaluated.append(entry)
        if rank == 0:
            write_json(root / f"step_{candidate['step']:06d}" / "full_development" / "metrics.json",
                       dict(entry, rows=all_rows, evaluation_scope="2595_training_archive_source_disjoint_development_records"))
            print("CONDITIONAL_FULL_EVAL", json.dumps(entry, allow_nan=False), flush=True)
    if rank == 0:
        passing = [x for x in evaluated if x["automatic_metric_gate_pass"]]
        selected = (max(passing, key=lambda x: (x["metrics"]["output_psnr"], -x["metrics"]["output_lpips"], x["step"]))
                    if passing else min(evaluated, key=lambda x: selection_score(x, settings)))
        result = dict(status="conditional_teacher_full_development_complete",
                      config_sha256=digest, admission=admission,
                      training_result_sha256=sha256(root / "result.json"), hardware=hardware,
                      training_records=settings["stages"]["full"]["train_records"],
                      training_source_groups=settings["stages"]["full"]["train_source_groups"],
                      development_records=len(rows), development_source_groups=len({x["source_group"] for x in rows}),
                      candidate_selection=settings["stages"]["full"]["candidate_selection"], candidates=evaluated,
                      selected=selected, automatic_metric_gate_pass=bool(passing),
                      summary={"task_metrics": {"Blur": {"psnr": selected["metrics"]["output_psnr"],
                                                         "lpips": selected["metrics"]["output_lpips"]}}},
                      visual_review_required=True, visual_review_complete=False,
                      **{k: False for k in PROTECTED})
        write_json(root / "full_development_result.json", result)
        print("CONDITIONAL_FULL_DEVELOPMENT_COMPLETE", json.dumps(dict(selected=selected,
              automatic_metric_gate_pass=bool(passing))), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
