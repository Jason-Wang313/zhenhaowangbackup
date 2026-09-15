#!/usr/bin/env python3
"""Eight-rank tiny gate for the pure Z-Image deterministic restoration bridge."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/diffsynth-studio")]

import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file

from lowlight_restoration.foundir_zimage_paired_flow import (
    load_aligned_pair, load_frozen_rows, sha256, validate_gpu_inventory,
    verify_model_artifacts, write_once,
)
from lowlight_restoration.zimage_deterministic_bridge import (
    PROTECTED, DeterministicBridgeTeacher, bridge_pair, clean_from_velocity,
    load_settings,
)
from lowlight_restoration.zimage_latent_first import average_gradients
from lowlight_restoration.zimage_padding_compat import build_padding_function
from lowlight_restoration.zimage_repro_batch import (
    install_flash_control, seed_everything, tensor_digest,
)
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import (
    aggregate, lpips_value, psnr,
)


def write_json(path: Path, value: object) -> None:
    write_once(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def gather(value, world: int):
    if world == 1:
        return value
    parts = [None] * world
    dist.all_gather_object(parts, value)
    return [item for part in parts for item in part]


def fixed_seed(seed: int, *parts: object) -> int:
    import hashlib
    raw = "|".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & 0x7FFF_FFFF


@torch.no_grad()
def encode(model, row, size: int):
    lq, gt = load_aligned_pair(row, size)
    pipe = model.pipe
    lq_rgb, gt_rgb = pipe.preprocess_image(lq), pipe.preprocess_image(gt)
    return dict(row=row, lq=lq, gt=gt, lq_rgb=lq_rgb, gt_rgb=gt_rgb,
                lq_latent=pipe.vae_encoder(lq_rgb).float().detach(),
                gt_latent=pipe.vae_encoder(gt_rgb).float().detach())


def charbonnier(a: torch.Tensor, b: torch.Tensor, epsilon: float) -> torch.Tensor:
    return torch.sqrt((a.float() - b.float()).square() + epsilon**2).mean()


def loss_for(model, item, prompt, sigma: float, metric, settings):
    state, target = bridge_pair(item["gt_latent"], item["lq_latent"], sigma)
    velocity = model(state, item["lq_latent"], sigma, prompt)
    clean = clean_from_velocity(state, velocity, sigma)
    rgb = model.pipe.vae_decoder(clean.to(model.pipe.torch_dtype)).float()
    weights = settings["loss"]
    terms = {
        "flow_mse": (velocity - target).square().mean(),
        "clean_latent_charbonnier": charbonnier(clean, item["gt_latent"], weights["charbonnier_epsilon"]),
        "rgb_charbonnier": charbonnier(rgb, item["gt_rgb"], weights["charbonnier_epsilon"]),
        # The metric's normalize=True contract accepts [0, 1]; retain gradients
        # through the frozen decoder while matching the evaluation convention.
        "lpips": metric(((rgb + 1) / 2).clamp(0, 1),
                         ((item["gt_rgb"].float() + 1) / 2).clamp(0, 1)),
    }
    loss = sum(terms[name] * weights[name] for name in terms)
    return loss, terms, velocity, clean, rgb


def optimizer_for(model, settings, scale: float = 1.0):
    projections = list(model.lq_projections.parameters())
    gate = [model.velocity_gate]
    excluded = {id(p) for p in projections + gate}
    lora = [p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
    if not lora or not projections:
        raise RuntimeError("missing LoRA or Layer-1 conditioning parameters")
    return torch.optim.AdamW([
        {"params": lora, "lr": settings["learning_rate"] * scale,
         "base_lr": settings["learning_rate"], "name": "lora"},
        {"params": projections, "lr": settings["condition_learning_rate"] * scale,
         "base_lr": settings["condition_learning_rate"], "name": "lq_multidepth"},
        {"params": gate, "lr": settings["velocity_gate_learning_rate"] * scale,
         "base_lr": settings["velocity_gate_learning_rate"], "name": "velocity_gate"},
    ], weight_decay=settings["weight_decay"])


def parameter_digest(model, world: int) -> str:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    digest = tensor_digest(named)
    if len(set(gather([digest], world))) != 1:
        raise RuntimeError("trainable parameters differ across ranks")
    return digest


def save_checkpoint(model, path: Path, config_digest: str, step: int, digest: str) -> str:
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(state, str(path), metadata={"config_sha256": config_digest, "step": str(step),
                                         "classification": "pure_zimage_layer1_bridge"})
    loaded = load_file(str(path))
    if tensor_digest(loaded.items()) != digest or any(v.dtype != torch.float32 for v in loaded.values()):
        raise RuntimeError("checkpoint roundtrip mismatch")
    return sha256(path)


def save_triptych(path: Path, item, output: Image.Image) -> None:
    size = item["lq"].width
    canvas = Image.new("RGB", (size * 3, size + 28), "white")
    draw = ImageDraw.Draw(canvas)
    rid = item["row"]["record_id"]
    for index, (label, image) in enumerate((("Input", item["lq"]), ("Output", output), ("GT", item["gt"]))):
        canvas.paste(image, (index * size, 28))
        draw.text((index * size + 4, 7), f"{rid} | {label}", fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, "PNG")
    write_once(path, buffer.getvalue())


def validate_rows(train, development) -> None:
    if len(train) != 16 or len(development) != 16:
        raise ValueError("fixed tiny 16/16 coverage changed")
    if len({r["source_group"] for r in train}) != 16 or len({r["source_group"] for r in development}) != 16:
        raise ValueError("tiny source groups changed")
    if {r["source_group"] for r in train} & {r["source_group"] for r in development}:
        raise ValueError("train/development source leakage")
    if any(r.get("category") != "01Blur" or r.get("composition_size") != "1" for r in train + development):
        raise ValueError("non-Blur or mixed degradation record")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    settings, base = load_settings(args.config)
    config_digest = sha256(args.config)
    train, development = load_frozen_rows(base, verify_images=True)
    validate_rows(train, development)
    if args.preflight:
        verify_model_artifacts(base, large_hashes=True)
        for name, memory in (("NVIDIA H100 80GB HBM3", 81920), ("NVIDIA H200", 143000)):
            validate_gpu_inventory([name] * 8, [memory] * 8, base)
        print("DETERMINISTIC_BRIDGE_PREFLIGHT_PASS", config_digest, flush=True)
        return

    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    if (args.cpu_smoke and world != 1) or (not args.cpu_smoke and world != 8):
        raise ValueError("CPU smoke requires one rank; training requires exactly eight")
    device = torch.device("cpu" if args.cpu_smoke else f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.set_num_threads(4)
    if not args.cpu_smoke:
        required = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NCCL_ALGO": "Ring", "NCCL_PROTO": "Simple"}
        if any(os.environ.get(k) != v for k, v in required.items()):
            raise RuntimeError(f"deterministic environment mismatch: {required}")
        torch.cuda.set_device(device)
        if torch.cuda.device_count() != 8:
            raise ValueError("exactly eight visible GPUs required")
        props = [torch.cuda.get_device_properties(i) for i in range(8)]
        hardware = validate_gpu_inventory([p.name for p in props], [p.total_memory // 2**20 for p in props], base)
        dist.init_process_group("nccl", device_id=device)
    else:
        hardware = "CPU_smoke_only"
    seed_everything(settings["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.cpu_smoke:
        install_flash_control(True)
    from diffsynth.pipelines import z_image
    _, repaired, padding = build_padding_function(z_image)
    z_image.model_fn_z_image_turbo = repaired
    verify_model_artifacts(base, large_hashes=rank == 0)
    if world > 1:
        dist.barrier()

    output = (Path(tempfile.mkdtemp(prefix="zimage_bridge_cpu_", dir=ROOT / "logs"))
              if args.cpu_smoke else ROOT / settings["output_dir"])
    if rank == 0 and not args.cpu_smoke:
        output.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    if rank == 0:
        write_json(output / "runtime.json", {
            "status": "pure_zimage_layer1_runtime", "config_sha256": config_digest,
            "source_sha256": sha256(Path(__file__)),
            "module_sha256": sha256(ROOT / "lowlight_restoration/zimage_deterministic_bridge.py"),
            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "hardware": hardware, "world": world, "global_batch": world,
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "padding": padding,
            "scope": "pure Z-Image Layer-1; no independent restoration branch",
        })
    model = DeterministicBridgeTeacher(base, settings, device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named or any(p.dtype != torch.float32 for _, p in named):
        raise RuntimeError("trainable state must exist entirely in FP32")
    parameters = [p for _, p in named]
    with torch.no_grad():
        prompt = model.pipe.units[1].process(model.pipe, base["model"]["prompt"], None)["prompt_embeds"]
    size = 64 if args.cpu_smoke else settings["crop_size"]
    cached = {
        "train": [encode(model, row, size) for row in train[rank::world]][:2 if args.cpu_smoke else None],
        "development": [encode(model, row, size) for row in development[rank::world]][:2 if args.cpu_smoke else None],
    }
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval().requires_grad_(False)

    # This is an executable architecture smoke, not a quality run.
    if args.cpu_smoke:
        item = cached["train"][0]
        restored = model.restore(item["lq_latent"], prompt, steps=2, shift=settings["inference_shift"])
        if not torch.equal(restored, item["lq_latent"]):
            raise RuntimeError("zero-gate solver is not exact latent identity")
        loss, terms, _, _, rgb = loss_for(model, item, prompt, .5, metric, settings)
        loss.backward()
        if model.velocity_gate.grad is None or not torch.isfinite(model.velocity_gate.grad).all():
            raise RuntimeError("velocity gate has no finite first-update gradient")
        print("DETERMINISTIC_BRIDGE_REAL_CPU_SMOKE_PASS", output.relative_to(ROOT),
              {k: float(v.detach()) for k, v in terms.items()}, flush=True)
        return

    snapshots = []

    @torch.no_grad()
    def evaluate(step: int, checkpoint_digest: str) -> None:
        for role, items in cached.items():
            local = []
            for item in items:
                latent = model.restore(item["lq_latent"], prompt, settings["teacher_steps"], settings["inference_shift"])
                if step == 0 and not torch.equal(latent, item["lq_latent"]):
                    raise RuntimeError("step-0 solver lost exact LQ-latent identity")
                rgb = model.pipe.vae_decoder(latent.to(model.pipe.torch_dtype))
                image = model.pipe.vae_output_to_image(rgb).convert("RGB")
                rid = item["row"]["record_id"]
                row = {
                    "record_id": rid, "source_group": item["row"]["source_group"],
                    "input_psnr": psnr(item["gt"], item["lq"]),
                    "input_lpips": lpips_value(metric, item["gt"], item["lq"], device),
                    "output_psnr": psnr(item["gt"], image),
                    "output_lpips": lpips_value(metric, item["gt"], image, device),
                    "latent_mse": float((latent - item["gt_latent"]).square().mean()),
                    "identity_latent_max_abs": float((latent - item["lq_latent"]).abs().max()),
                }
                if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
                    raise RuntimeError("non-finite evaluation metric")
                folder = output / f"step_{step:06d}" / role
                save_triptych(folder / f"{rid}.png", item, image)
                write_json(folder / f"{rid}.json", {**row, "step": step,
                           "config_sha256": config_digest, "checkpoint_digest": checkpoint_digest,
                           "triptych_sha256": sha256(folder / f"{rid}.png")})
                local.append(row)
                metric.reset()
            rows = sorted(gather(local, world), key=lambda row: row["record_id"])
            expected = train if role == "train" else development
            if len(rows) != 16 or {r["record_id"] for r in rows} != {r["record_id"] for r in expected}:
                raise RuntimeError("distributed evaluation coverage mismatch")
            metrics = aggregate(rows)
            metrics["latent_mse"] = sum(r["latent_mse"] for r in rows) / len(rows)
            payload = {"step": step, "role": role, "metrics": metrics, "rows": rows}
            if rank == 0:
                snapshots.append(payload)
                write_json(output / f"step_{step:06d}" / role / "metrics.json", payload)
                print("DETERMINISTIC_BRIDGE_EVAL", step, role, json.dumps(metrics), flush=True)
        dist.barrier()

    initial_digest = parameter_digest(model, world)
    evaluate(0, initial_digest)
    optimizer = optimizer_for(model, settings)
    local_train = cached["train"]
    start = time.monotonic()
    checkpoints = []
    for step in range(1, settings["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        lr_scale = min(1.0, step / settings["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale
        item = local_train[(step - 1) % len(local_train)]
        generator = torch.Generator(device=device).manual_seed(fixed_seed(settings["seed"], "sigma", rank, step))
        sigma = float(torch.rand((), generator=generator, device=device))
        loss, terms, _, _, _ = loss_for(model, item, prompt, sigma, metric, settings)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Layer-1 loss")
        loss.backward()
        average_gradients(parameters, world)
        norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        if any(optimizer.state[p][key].dtype != torch.float32 for p in parameters
               for key in ("exp_avg", "exp_avg_sq")):
            raise RuntimeError("AdamW state is not FP32")
        values = torch.tensor([float(loss.detach()), *(float(terms[k].detach()) for k in
                              ("flow_mse", "clean_latent_charbonnier", "rgb_charbonnier", "lpips"))], device=device)
        dist.all_reduce(values)
        values /= world
        metric.reset()
        if rank == 0:
            row = dict(step=step, loss=float(values[0]), flow_mse=float(values[1]),
                       clean_latent_charbonnier=float(values[2]), rgb_charbonnier=float(values[3]),
                       lpips=float(values[4]), sigma=sigma, grad_norm=float(norm), lr_scale=lr_scale,
                       elapsed_seconds=time.monotonic() - start)
            with (output / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 20 == 0:
                print("DETERMINISTIC_BRIDGE_TRAIN", f"{step}/{settings['steps']}", json.dumps(row), flush=True)
        if step in settings["snapshot_steps"]:
            digest = parameter_digest(model, world)
            path = output / f"step_{step:06d}.safetensors"
            if rank == 0:
                checkpoints.append({"step": step, "path": str(path.relative_to(ROOT)),
                                    "sha256": save_checkpoint(model, path, config_digest, step, digest)})
            dist.barrier()
            evaluate(step, digest)
    if rank == 0:
        final = next(x["metrics"] for x in snapshots if x["step"] == settings["steps"] and x["role"] == "train")
        baseline = next(x["metrics"] for x in snapshots if x["step"] == 0 and x["role"] == "train")
        gate = settings["gate"]
        passed = (final["output_psnr"] >= gate["train_psnr_min"] and
                  final["output_lpips"] <= gate["train_lpips_max"] and
                  final["output_psnr"] > baseline["input_psnr"] and
                  final["output_lpips"] < baseline["input_lpips"])
        result = {
            "status": "zimage_deterministic_bridge_tiny_complete", "config_sha256": config_digest,
            "classification": "pure_zimage_layer1_teacher_not_independent_restoration_branch",
            "steps": settings["steps"], "checkpoints": checkpoints, "snapshots": snapshots,
            "automatic_metric_gate_pass": passed, "visual_review_required": True,
            **{key: False for key in PROTECTED},
        }
        write_json(output / "result.json", result)
        print("DETERMINISTIC_BRIDGE_TINY_COMPLETE", json.dumps({"metric_gate": passed, "metrics": final}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
