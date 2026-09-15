#!/usr/bin/env python3
"""Native conditional Z-Image teacher: tiny learnability then diverse pilot."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from PIL import Image, ImageDraw

from lowlight_restoration.zimage_conditional_teacher import (
    ConditionalTeacher, PROTECTED, fixed_seed, load_settings, noise_flow_pair, stage_gate,
)
from lowlight_restoration.foundir_zimage_paired_flow import (
    load_frozen_rows, load_aligned_pair, sha256, verify_model_artifacts, validate_gpu_inventory, write_once,
)
from lowlight_restoration.zimage_repro_batch import seed_everything, install_flash_control, tensor_digest
from lowlight_restoration.zimage_padding_compat import build_padding_function
from lowlight_restoration.zimage_latent_first import average_gradients
from smoke.evaluate_foundir_blur_only_zimage_teacher_paired_flow_ddp import psnr, lpips_value, aggregate
from smoke.prepare_zimage_conditional_pilot import freeze
from smoke.prepare_zimage_conditional_pilot import select_groups
from smoke.build_foundir_blur_only_sanity_manifest import frozen_row

PILOT_MANIFEST_SHA256 = "19309079bd2a64e253527ff1b637fc3bd2a17522cde8c47fbd3a60283315e11f"
SCALE_AUTHORIZATION_SHA256 = "21c6cc9b8f20e37fde16fad5799e4fa701118c4730ba5defc4fca6d1e6365460"
FULL_AUTHORIZATION_SHA256 = "56257204db0239315301e0b45db1469a7dcb7e3aea8d13eeb1805847f3ca4eb8"


def write_json(path, obj):
    write_once(path, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def gather(rows, world):
    if world == 1:
        return rows
    parts = [None] * world
    dist.all_gather_object(parts, rows)
    return [r for part in parts for r in part]


def load_rows(s, base, stage):
    tiny, visual = load_frozen_rows(base, verify_images=True)
    if stage == "tiny":
        return tiny, tiny, visual
    if stage == "full":
        source = ROOT / base["data"]["source_manifest"]
        if sha256(source) != base["data"]["source_manifest_sha256"]:
            raise ValueError("full source manifest changed")
        with source.open(newline="") as f:
            source_rows = [r for r in csv.DictReader(f) if r["category"] == "01Blur"
                           and r["composition_size"] == "1" and r["split"] in ("train", "validation")]
        train_raw = [r for r in source_rows if r["split"] == "train"]
        dev_raw = [r for r in source_rows if r["split"] == "validation"]
        c = s["stages"]["full"]
        if (len(train_raw), len({r["source_group"] for r in train_raw}), len(dev_raw),
                len({r["source_group"] for r in dev_raw})) != (
                    c["train_records"], c["train_source_groups"], c["full_development_records"],
                    c["full_development_source_groups"]):
            raise ValueError("full Blur coverage changed")
        if sha256(ROOT / s["pilot_manifest"]) != PILOT_MANIFEST_SHA256:
            raise ValueError("pilot manifest changed before full scaling")
        with (ROOT / s["pilot_manifest"]).open(newline="") as f:
            pilot_dev = [r for r in csv.DictReader(f) if r["role"] == "pilot_development"]
        monitor = select_groups(dev_raw, pilot_dev, c["monitor_records"], "full_monitor", s["seed"])
        train = [frozen_row(r, role="full_train", crop_size=s["crop_size"], crop_seed=s["seed"])
                 for r in train_raw]
        if {r["source_group"] for r in train} & {r["source_group"] for r in monitor}:
            raise ValueError("full train/development leakage")
        return train, tiny, monitor
    meta = json.loads((ROOT / s["pilot_manifest_metadata"]).read_text())
    if meta["manifest_sha256"] != PILOT_MANIFEST_SHA256 or sha256(ROOT / s["pilot_manifest"]) != PILOT_MANIFEST_SHA256:
        raise ValueError("pilot manifest hash mismatch")
    with (ROOT / s["pilot_manifest"]).open(newline="") as f:
        rows = list(csv.DictReader(f))
    train = [r for r in rows if r["role"] == "pilot_train"]
    dev = [r for r in rows if r["role"] == "pilot_development"]
    if len(train) != 2048 or len(dev) != 128:
        raise ValueError("pilot coverage changed")
    if {r["source_group"] for r in train} & {r["source_group"] for r in dev}:
        raise ValueError("source leakage")
    if any(r["split"] != split or r["category"] != "01Blur" or r["composition_size"] != "1"
           for split, selected in (("train", train), ("validation", dev)) for r in selected):
        raise ValueError("protected/non-Blur row")
    return train, tiny, dev


def load_full_development(s, base):
    source = ROOT / base["data"]["source_manifest"]
    if sha256(source) != base["data"]["source_manifest_sha256"]:
        raise ValueError("full source manifest changed")
    with source.open(newline="") as f:
        rows = [frozen_row(r, role="full_development", crop_size=s["crop_size"], crop_seed=s["seed"])
                for r in csv.DictReader(f) if r["category"] == "01Blur" and r["composition_size"] == "1"
                and r["split"] == "validation"]
    c = s["stages"]["full"]
    if len(rows) != c["full_development_records"] or len({r["source_group"] for r in rows}) != c["full_development_source_groups"]:
        raise ValueError("full development coverage changed")
    return rows


def select_full_candidates(snapshots, final_step):
    rows = [x for x in snapshots if x["role"] == "development" and x["step"] > 0]
    if not rows or rows[-1]["step"] != final_step:
        raise ValueError("full monitor curve incomplete")
    best_psnr = max(rows, key=lambda x: (x["metrics"]["output_psnr"], x["step"]))["step"]
    best_lpips = min(rows, key=lambda x: (x["metrics"]["output_lpips"], -x["step"]))["step"]
    return sorted({best_psnr, best_lpips, final_step})


def validate_pilot_pair(row):
    for field in ("lq", "gt"):
        path = (ROOT / row[field + "_path"]).resolve()
        if not path.is_relative_to(ROOT / "data/foundir/MillionIRData/Train/01Blur") or sha256(path) != row[field + "_sha256"]:
            raise ValueError("pilot image path/hash mismatch")
    load_aligned_pair(row, 512)


def validate_review(s, digest, review_path):
    if review_path is None:
        raise ValueError("pilot requires the completed tiny metric and visual review artifact")
    review = json.loads(review_path.read_text())
    root = ROOT / s["stages"]["tiny"]["output_dir"]
    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    expected = dict(config_sha256=digest, result_sha256=sha256(result_path),
                    checkpoint_sha256=result["final_checkpoint_sha256"], visual_pass=True)
    if any(review.get(k) != v for k, v in expected.items()) or not review.get("reviewed_images") or not review.get("rationale"):
        raise ValueError("missing/unbound tiny visual review")
    if result["automatic_metric_gate_pass"] is not True:
        raise ValueError("tiny metric gate failed")
    path = ROOT / result["final_checkpoint"]
    if sha256(path) != result["final_checkpoint_sha256"]:
        raise ValueError("tiny checkpoint changed")
    return path


def exact_load(model, path):
    state = load_file(str(path), device="cpu")
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != set(named) or any(v.dtype != torch.float32 for v in state.values()):
        raise ValueError("checkpoint trainable state/precision mismatch")
    with torch.no_grad():
        for n, p in named.items():
            if p.shape != state[n].shape:
                raise ValueError("checkpoint geometry mismatch")
            p.copy_(state[n].to(p.device))
    if tensor_digest(named.items()) != tensor_digest(state.items()):
        raise ValueError("checkpoint reload mismatch")


def validate_admission(s, digest, stage, review_path=None, scale_path=None, full_path=None):
    """A prospective user exception is not a passing tiny visual review."""
    if sum(x is not None for x in (review_path, scale_path, full_path)) > 1:
        raise ValueError("choose exactly one admission artifact")
    if stage == "full":
        if full_path is None:
            raise ValueError("full scaling requires explicit authorization")
        if sha256(full_path) != FULL_AUTHORIZATION_SHA256:
            raise ValueError("full authorization changed")
        authorization = json.loads(full_path.read_text())
        c, parent = s["stages"]["full"], s["parent_pilot"]
        result_path, checkpoint = ROOT / parent["result"], ROOT / parent["checkpoint"]
        result = json.loads(result_path.read_text())
        expected = dict(config_sha256=digest, parent_result_sha256=sha256(result_path),
                        parent_checkpoint_sha256=sha256(checkpoint), required_world_size=8,
                        train_records=c["train_records"], train_source_groups=c["train_source_groups"],
                        additional_optimizer_steps=c["steps"], global_batch=c["global_batch"],
                        snapshot_steps=c["snapshot_steps"], monitor_source_groups=c["monitor_source_groups"],
                        full_development_records=c["full_development_records"],
                        full_development_source_groups=c["full_development_source_groups"],
                        sow_psnr_min=c["gate"]["sow_psnr_min"], sow_lpips_max=c["gate"]["sow_lpips_max"],
                        candidate_selection=c["candidate_selection"])
        if any(authorization.get(k) != v for k, v in expected.items()):
            raise ValueError("full authorization config/parent/budget mismatch")
        if parent["result_sha256"] != expected["parent_result_sha256"] or parent["checkpoint_sha256"] != expected["parent_checkpoint_sha256"]:
            raise ValueError("full parent identity changed")
        if result["stage"] != "pilot" or result["steps"] != 2000 or result["automatic_metric_gate_pass"] is not False:
            raise ValueError("unexpected full parent result")
        if any(result.get(k) is not False for k in PROTECTED):
            raise ValueError("full parent protected scope changed")
        return checkpoint, dict(mode=authorization["decision"], authorization_sha256=sha256(full_path),
                                authorization=authorization, initial_checkpoint=parent["checkpoint"],
                                initial_checkpoint_sha256=parent["checkpoint_sha256"], optimizer_reset=True)
    if stage != "pilot":
        if review_path is not None or scale_path is not None or full_path is not None:
            raise ValueError("admission artifacts apply to pilot/full only")
        return None, dict(mode="fresh_tiny")
    if scale_path is None:
        path = validate_review(s, digest, review_path)
        return path, dict(mode="tiny_metric_and_visual_pass", review_sha256=sha256(review_path))
    if sha256(scale_path) != SCALE_AUTHORIZATION_SHA256:
        raise ValueError("scale authorization changed")
    authorization = json.loads(scale_path.read_text())
    result_path = ROOT / s["stages"]["tiny"]["output_dir"] / "result.json"
    result = json.loads(result_path.read_text())
    c = s["stages"]["pilot"]
    expected = dict(config_sha256=digest, result_sha256=sha256(result_path),
                    checkpoint_sha256=result["final_checkpoint_sha256"],
                    pilot_manifest_sha256=PILOT_MANIFEST_SHA256,
                    pilot_steps=c["steps"], pilot_global_batch=c["global_batch"],
                    train_source_groups=c["train_source_groups"],
                    development_source_groups=c["development_source_groups"],
                    snapshot_steps=c["snapshot_steps"])
    if any(authorization.get(k) != v for k, v in expected.items()):
        raise ValueError("scale authorization config/result/protocol mismatch")
    if result["config_sha256"] != digest or result["stage"] != "tiny" or result["steps"] != 1200:
        raise ValueError("unexpected tiny initialization")
    if result["automatic_metric_gate_pass"] is not False or any(result.get(k) is not False for k in PROTECTED):
        raise ValueError("tiny failure/protected scope changed")
    path = ROOT / s["stages"]["tiny"]["output_dir"] / "step_001200.safetensors"
    if result["final_checkpoint"] != str(path.relative_to(ROOT)) or sha256(path) != authorization["checkpoint_sha256"]:
        raise ValueError("scale initialization checkpoint changed")
    return path, dict(mode=authorization["decision"], authorization_sha256=sha256(scale_path),
                      authorization=authorization, initial_checkpoint=str(path.relative_to(ROOT)),
                      initial_checkpoint_sha256=authorization["checkpoint_sha256"], optimizer_reset=True)


@torch.no_grad()
def encode(model, row, crop_size, transform=0, cropped_pair=None):
    lq, gt = load_aligned_pair(row, crop_size) if cropped_pair is None else cropped_pair
    if transform & 1:
        lq, gt = lq.transpose(Image.Transpose.FLIP_LEFT_RIGHT), gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    for _ in range(transform // 2):
        lq, gt = lq.transpose(Image.Transpose.ROTATE_90), gt.transpose(Image.Transpose.ROTATE_90)
    pipe = model.pipe
    return dict(row=row, lq=lq, gt=gt,
                lq_latent=pipe.vae_encoder(pipe.preprocess_image(lq)).detach(),
                gt_latent=pipe.vae_encoder(pipe.preprocess_image(gt)).detach())


def sample_loss(model, item, prompt, seed, sigma=None):
    gt = item["gt_latent"]
    gen = torch.Generator(device=gt.device).manual_seed(seed)
    noise = torch.randn(gt.shape, generator=gen, device=gt.device, dtype=torch.float32)
    sigma = float(torch.rand((), generator=gen, device=gt.device)) if sigma is None else sigma
    state, target = noise_flow_pair(gt, noise, sigma)
    prediction = model(state, item["lq_latent"], sigma, prompt)
    return (prediction.float() - target).square().mean(), prediction


def optimizer_for(model, s, warmup_scale=1.0):
    lora = [p for n, p in model.named_parameters() if p.requires_grad and n != "lq_projection.weight"]
    return torch.optim.AdamW([
        {"params": lora, "lr": s["learning_rate"] * warmup_scale, "base_lr": s["learning_rate"]},
        {"params": [model.lq_projection.weight], "lr": s["condition_learning_rate"] * warmup_scale,
         "base_lr": s["condition_learning_rate"]},
    ], weight_decay=s["weight_decay"])


def check_rank_weights(named, world):
    digest = tensor_digest(named)
    if len(set(gather([digest], world))) != 1:
        raise RuntimeError("data-parallel parameter mismatch")
    return digest


def rehearsal(model, item, prompt, s, world, native, root, rank):
    """Native equivalence, repeated synchronized updates, actual LQ sensitivity."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    initial = {n: p.detach().cpu().clone() for n, p in named}
    initial_digest = check_rank_weights(named, world)
    # The exact native comparison is meaningful before any condition training.
    generator = torch.Generator(device=item["lq_latent"].device).manual_seed(s["seed"])
    state = torch.randn(item["lq_latent"].shape, generator=generator, device=item["lq_latent"].device, dtype=torch.bfloat16)
    rows = []
    for which in ("native", "conditional_zero"):
        model.zero_grad(set_to_none=True)
        prediction = (native(dit=model.pipe.dit, latents=state, timestep=state.new_tensor([500]),
                             prompt_embeds=prompt, use_gradient_checkpointing=True)
                      if which == "native" else model(state, item["lq_latent"], .5, prompt))
        prediction.float().square().mean().backward()
        rows.append(dict(name=which, output=tensor_digest([("v", prediction)]),
                         lora_gradient=tensor_digest([(n, p.grad) for n, p in named if n != "lq_projection.weight"])))
    write_json(root / f"native_equivalence_rank_{rank}.json", rows)
    if rows[0]["output"] != rows[1]["output"] or rows[0]["lora_gradient"] != rows[1]["lora_gradient"]:
        raise RuntimeError("zero-condition native forward/backward mismatch")
    references = None
    for repeat in range(2):
        with torch.no_grad():
            for n, p in named:
                p.copy_(initial[n].to(p.device))
        optimizer = optimizer_for(model, s)
        updates = []
        for index, sigma in enumerate((.5, .75)):
            optimizer.zero_grad(set_to_none=True)
            loss, prediction = sample_loss(model, item, prompt, fixed_seed(s["seed"], "rehearsal", rank, index), sigma)
            loss.backward()
            average_gradients(params, world)
            norm = torch.nn.utils.clip_grad_norm_(params, s["max_grad_norm"], error_if_nonfinite=True)
            grad_digest = tensor_digest([(n, p.grad) for n, p in named])
            optimizer.step()
            if any(optimizer.state[p][key].dtype != torch.float32 for p in params for key in ("exp_avg", "exp_avg_sq")):
                raise RuntimeError("non-FP32 AdamW state")
            updates.append(dict(loss=float(loss.detach()), output=tensor_digest([("v", prediction)]),
                                gradient=grad_digest, weights=check_rank_weights(named, world), grad_norm=float(norm)))
        if repeat and references != updates:
            raise RuntimeError("conditional teacher two-update repeat differs")
        references = updates
        del optimizer
    with torch.no_grad():
        a = model(state, item["lq_latent"], .5, prompt, grad=False)
        b = model(state, torch.zeros_like(item["lq_latent"]), .5, prompt, grad=False)
        sensitivity = float((a.float() - b.float()).abs().max())
    if sensitivity <= 0 or not math.isfinite(sensitivity):
        raise RuntimeError("learned LQ path does not change model output")
    trial_path = root / "rehearsal_weights.safetensors"
    if rank == 0:
        save_file({n: p.detach().cpu().contiguous() for n, p in named}, str(trial_path))
    if world > 1:
        dist.barrier()
    digest_before = tensor_digest(named)
    with torch.no_grad():
        for _, p in named:
            p.zero_()
    exact_load(model, trial_path)
    if tensor_digest(named) != digest_before:
        raise RuntimeError("rehearsal checkpoint reload failed")
    with torch.no_grad():
        for n, p in named:
            p.copy_(initial[n].to(p.device))
    model.zero_grad(set_to_none=True)
    if check_rank_weights(named, world) != initial_digest:
        raise RuntimeError("initial state not restored")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("frozen parameter received gradient")
    result = dict(status="pass", native_forward_backward_exact=True, repeated_updates_exact=True,
                  updates=references, lq_sensitivity_absmax=sensitivity, reset_digest=initial_digest,
                  checkpoint_sha256=sha256(trial_path))
    write_json(root / f"rehearsal_rank_{rank}.json", result)
    print("CONDITIONAL_REHEARSAL_PASS", rank, flush=True)


def triptych(path, item, image):
    size = item["lq"].width
    canvas = Image.new("RGB", (3 * size, size + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, picture) in enumerate((("Input", item["lq"]), ("Conditional teacher", image), ("GT", item["gt"]))):
        canvas.paste(picture, (i * size, 28))
        draw.text((i * size + 4, 7), item["row"]["record_id"] + " | " + label, fill="black")
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    write_once(path, buffer.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("tiny", "pilot", "full"), default="tiny")
    parser.add_argument("--tiny-review", type=Path)
    parser.add_argument("--scale-authorization", type=Path)
    parser.add_argument("--full-authorization", type=Path)
    parser.add_argument("--admission-check", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--cpu-inference-smoke", action="store_true")
    args = parser.parse_args()
    args.cpu_smoke = args.cpu_smoke or args.cpu_inference_smoke
    s, base = load_settings(args.config)
    digest = sha256(args.config)
    c = s["stages"][args.stage]
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    checkpoint_path, admission = validate_admission(s, digest, args.stage, args.tiny_review,
                                                    args.scale_authorization, args.full_authorization)
    if args.admission_check:
        print("CONDITIONAL_ADMISSION_PASS", json.dumps(admission), flush=True)
        return
    if args.preflight:
        if args.stage != "full":
            freeze(args.config)
    train, train_eval, dev = load_rows(s, base, args.stage)
    if args.preflight:
        verify_model_artifacts(base, large_hashes=True)
        for name, mem in (("NVIDIA H100 80GB HBM3", 81920), ("NVIDIA H200", 143000)):
            validate_gpu_inventory([name] * 8, [mem] * 8, base)
        if args.stage == "full":
            full_dev = load_full_development(s, base)
            checked = 0
            for row in train + full_dev:
                for field in ("lq_path", "gt_path"):
                    path = (ROOT / row[field]).resolve()
                    if not path.is_relative_to(ROOT / "data/foundir/MillionIRData/Train/01Blur") or not path.is_file():
                        raise ValueError("full image path missing/outside Blur training archive")
                checked += 1
            print("CONDITIONAL_FULL_PATH_PREFLIGHT", checked, "pairs", flush=True)
            print("CONDITIONAL_PREFLIGHT_PASS", digest, "full=105057+2595 both_hardware_profiles", flush=True)
        else:
            # Validate every selected pilot image before presenting its launcher.
            pilot, _, pilot_dev = load_rows(s, base, "pilot")
            with ThreadPoolExecutor(max_workers=8) as pool:
                for count, _ in enumerate(pool.map(validate_pilot_pair, pilot + pilot_dev), 1):
                    if count % 128 == 0 or count == len(pilot) + len(pilot_dev):
                        print("CONDITIONAL_IMAGE_PREFLIGHT", f"{count}/{len(pilot) + len(pilot_dev)}", flush=True)
            print("CONDITIONAL_PREFLIGHT_PASS", digest, "pilot=2048+128 both_hardware_profiles", flush=True)
        return
    if (args.cpu_smoke and world != 1) or (not args.cpu_smoke and world != 8):
        raise ValueError("CPU smoke requires one rank; training requires eight")
    if args.cpu_smoke and args.stage != "tiny":
        raise ValueError("CPU smoke starts fresh Base only")
    device = torch.device("cpu" if args.cpu_smoke else f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.set_num_threads(4)
    if not args.cpu_smoke:
        for k, v in dict(CUBLAS_WORKSPACE_CONFIG=":4096:8", NCCL_ALGO="Ring", NCCL_PROTO="Simple").items():
            if os.environ.get(k) != v:
                raise RuntimeError(f"required deterministic runtime {k}={v}")
        torch.cuda.set_device(device)
        props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
        hardware = validate_gpu_inventory([p.name for p in props], [p.total_memory // 2**20 for p in props], base)
        dist.init_process_group("nccl", device_id=device)
    else:
        hardware = "CPU_smoke_only"
    seed_everything(s["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not args.cpu_smoke:
        install_flash_control(True)
    from diffsynth.pipelines import z_image
    _, native, padding = build_padding_function(z_image)
    verify_model_artifacts(base, large_hashes=rank == 0)
    if world > 1:
        dist.barrier()
    root = Path(tempfile.mkdtemp(prefix="zimage_conditional_cpu_", dir=ROOT / "logs")) if args.cpu_smoke else ROOT / c["output_dir"]
    if not args.cpu_smoke and rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    if rank == 0:
        write_json(root / "runtime.json", dict(config_sha256=digest, hardware=hardware, world=world,
                   admission=admission,
                   global_batch=c["global_batch"], microbatch=1, accumulation=c["global_batch"] // 8,
                   torch=str(torch.__version__), cuda=torch.version.cuda, padding=padding,
                   source_sha256=sha256(Path(__file__)), helper_sha256=sha256(ROOT / "lowlight_restoration/zimage_conditional_teacher.py"),
                   git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    model = ConditionalTeacher(base, s, device)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named) != 477 or sum(p.numel() for _, p in named) != 79626240 or any(p.dtype != torch.float32 for _, p in named):
        raise RuntimeError("unexpected trainable state")
    params = [p for _, p in named]
    with torch.no_grad():
        prompt = model.pipe.units[1].process(model.pipe, base["model"]["prompt"], None)["prompt_embeds"]
    size = 64 if args.cpu_smoke else s["crop_size"]
    probe_item = encode(model, train[rank % len(train)], size)
    if args.cpu_inference_smoke:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval().requires_grad_(False)
        with torch.no_grad():
            a = model.restore(probe_item["lq_latent"], prompt, s["seed"], steps=2)
            b = model.restore(probe_item["lq_latent"], prompt, s["seed"], steps=2)
            if not torch.equal(a, b):
                raise RuntimeError("CPU solver repeat mismatch")
            rgb = model.pipe.vae_decoder(a.to(model.pipe.torch_dtype))
            if rgb.shape != (1, 3, size, size) or not torch.isfinite(rgb).all():
                raise RuntimeError("CPU decoder geometry/nonfinite")
            image = model.pipe.vae_output_to_image(rgb).convert("RGB")
        values = dict(output_psnr=psnr(probe_item["gt"], image), output_lpips=lpips_value(metric, probe_item["gt"], image, device))
        if not all(math.isfinite(v) for v in values.values()):
            raise RuntimeError("CPU metric nonfinite")
        triptych(root / "inference_smoke.png", probe_item, image)
        write_json(root / "inference_smoke.json", dict(status="pass", deterministic_solver_repeat=True,
                   **values, triptych_sha256=sha256(root / "inference_smoke.png"), quality_claim=False))
        print("CONDITIONAL_REAL_CPU_INFERENCE_PASS", root.relative_to(ROOT), flush=True)
        return
    rehearsal(model, probe_item, prompt, s, world, native, root, rank)
    if args.cpu_smoke:
        print("CONDITIONAL_REAL_CPU_SMOKE_PASS", root.relative_to(ROOT), flush=True)
        return
    if args.stage in ("pilot", "full"):
        exact_load(model, checkpoint_path)
        check_rank_weights(named, world)
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval().requires_grad_(False)
    eval_items = {"train": [encode(model, r, size) for r in train_eval[rank::world]],
                  "development": [encode(model, r, size) for r in dev[rank::world]]}
    snapshots = []

    @torch.no_grad()
    def probe_loss():
        values = []
        for item in eval_items["train"]:
            for sigma in (.1, .5, .9):
                seed = fixed_seed(s["seed"], "probe", item["row"]["record_id"], sigma)
                loss, _ = sample_loss(model, item, prompt, seed, sigma)
                values.append(float(loss))
        values = gather(values, world)
        return sum(values) / len(values)

    @torch.no_grad()
    def evaluate(step, checkpoint_digest):
        for role, items in eval_items.items():
            local = []
            for item in items:
                rid = item["row"]["record_id"]
                latent = model.restore(item["lq_latent"], prompt, fixed_seed(s["seed"], "evaluation", rid), s["teacher_steps"], s["inference_shift"])
                rgb = model.pipe.vae_decoder(latent.to(model.pipe.torch_dtype))
                image = model.pipe.vae_output_to_image(rgb).convert("RGB")
                row = dict(record_id=rid, source_group=item["row"]["source_group"],
                           input_psnr=psnr(item["gt"], item["lq"]), output_psnr=psnr(item["gt"], image),
                           input_lpips=lpips_value(metric, item["gt"], item["lq"], device),
                           output_lpips=lpips_value(metric, item["gt"], image, device),
                           latent_mse=float((latent - item["gt_latent"].float()).square().mean()))
                if any(not math.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
                    raise RuntimeError("nonfinite evaluation")
                folder = root / f"step_{step:06d}" / role
                triptych(folder / f"{rid}.png", item, image)
                write_json(folder / f"{rid}.json", dict(row, config_sha256=digest, checkpoint_digest=checkpoint_digest,
                           triptych_sha256=sha256(folder / f"{rid}.png")))
                local.append(row)
                metric.reset()
            rows = sorted(gather(local, world), key=lambda r: r["record_id"])
            expected = train_eval if role == "train" else dev
            if len(rows) != len(expected) or {r["record_id"] for r in rows} != {r["record_id"] for r in expected}:
                raise RuntimeError("evaluation record coverage mismatch")
            entry = dict(step=step, role=role, metrics=aggregate(rows), rows=rows)
            snapshots.append(entry)
            if rank == 0:
                write_json(root / f"step_{step:06d}" / role / "metrics.json", entry)
                print("CONDITIONAL_EVAL", step, role, json.dumps(entry["metrics"]), flush=True)

    initial_probe = probe_loss()
    evaluate(0, check_rank_weights(named, world))
    optimizer = optimizer_for(model, s)
    local_rows = train[rank::world]
    cache = {}
    crop_cache = {}
    permutations = {}
    accumulation = c["global_batch"] // world
    start = time.monotonic()
    for step in range(1, c["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        scale = min(1.0, step / s["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale
        losses, presented = [], []
        for micro in range(accumulation):
            index = (step - 1) * accumulation + micro
            epoch, offset = divmod(index, len(local_rows))
            if epoch not in permutations:
                order = list(range(len(local_rows)))
                random.Random(fixed_seed(s["seed"], "order", rank, epoch)).shuffle(order)
                permutations = {epoch: order}
            row = local_rows[permutations[epoch][offset]]
            transform = fixed_seed(s["seed"], "augment", row["record_id"], epoch) % 8 if args.stage in ("pilot", "full") else 0
            key = (row["record_id"], transform)
            if key not in cache:
                # Decode a full-resolution PNG pair only once per rank. Cache
                # the aligned 512px RGB crop, then encode its exact D4 variants.
                # Pilot: 256 crop pairs/rank, about 384 MiB host RGB storage.
                if row["record_id"] not in crop_cache:
                    crop_cache[row["record_id"]] = load_aligned_pair(row, size)
                item = encode(model, row, size, transform, crop_cache[row["record_id"]])
                cache[key] = {k: item[k].cpu() for k in ("lq_latent", "gt_latent")}
            item = {k: v.to(device) for k, v in cache[key].items()}
            loss, _ = sample_loss(model, item, prompt, fixed_seed(s["seed"], "train", rank, step, micro))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            presented.append(row["record_id"])
        average_gradients(params, world)
        norm = torch.nn.utils.clip_grad_norm_(params, s["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        all_losses = gather(losses, world)
        all_presented = gather(presented, world)
        if rank == 0:
            record = dict(step=step, loss=sum(all_losses) / len(all_losses), grad_norm=float(norm),
                          lr_scale=scale, records=all_presented, elapsed_seconds=time.monotonic() - start)
            with (root / "training.jsonl").open("a") as f:
                f.write(json.dumps(record, allow_nan=False) + "\n")
            if step == 1 or step % 20 == 0:
                print("CONDITIONAL_TRAIN", f"{step}/{c['steps']}", json.dumps(record), flush=True)
        if step in c["snapshot_steps"]:
            parameter_digest = check_rank_weights(named, world)
            path = root / f"step_{step:06d}.safetensors"
            if rank == 0:
                save_file({n: p.detach().cpu().contiguous() for n, p in named}, str(path),
                          metadata=dict(config_sha256=digest, step=str(step), stage=args.stage))
                if tensor_digest(load_file(str(path)).items()) != parameter_digest:
                    raise RuntimeError("saved checkpoint differs")
            dist.barrier()
            evaluate(step, parameter_digest)
    final_probe = probe_loss()
    if rank == 0:
        metrics = next(x["metrics"] for x in snapshots if x["step"] == c["steps"] and x["role"] == ("train" if args.stage == "tiny" else "development"))
        gate = stage_gate(args.stage, metrics, initial_probe, final_probe, s)
        candidates = None
        if args.stage == "full":
            candidates = []
            for step in select_full_candidates(snapshots, c["steps"]):
                candidate_path = root / f"step_{step:06d}.safetensors"
                candidates.append(dict(step=step, checkpoint=str(candidate_path.relative_to(ROOT)),
                                       checkpoint_sha256=sha256(candidate_path)))
        result = dict(status="conditional_teacher_stage_complete", stage=args.stage, config_sha256=digest,
                      admission=admission,
                      final_checkpoint=str(path.relative_to(ROOT)), final_checkpoint_sha256=sha256(path),
                      steps=c["steps"], snapshots=snapshots, initial_probe=initial_probe, final_probe=final_probe,
                      full_development_candidates=candidates, **gate, **{k: False for k in PROTECTED})
        write_json(root / "result.json", result)
        print("CONDITIONAL_STAGE_COMPLETE", args.stage, json.dumps(gate), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
