#!/usr/bin/env python3
"""Frozen 50-step original-LoRA vs static-bank replay; training LQ only."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import uuid
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))
from smoke import existing_flux_replay_contract_v1 as c


def save_tensors(output, prefix, tensors):
    from safetensors.torch import save_file
    name = f"tensors/{prefix}_{uuid.uuid4().hex}.safetensors"
    p = c.path(output, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    # An interrupted orphan is preserved, never mistaken for a committed record.
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(p))
    with p.open("rb") as f:
        os.fsync(f.fileno())
    directory = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return name, c.sha(p)


def hardware(world, *, require_idle):
    import torch
    if world not in (4, 8) or torch.cuda.device_count() != world:
        raise ValueError("expose exactly 4 or 8 CUDA GPUs")
    props = [torch.cuda.get_device_properties(i) for i in range(world)]
    if len({p.name for p in props}) != 1 or any(
            not any(f in p.name for f in ("H100", "H200")) or p.total_memory < 75 * 1024**3 for p in props):
        raise ValueError("homogeneous H100/H200, >=75 GiB per device required")
    if require_idle:
        visible_ids = {str(p.uuid).lower().removeprefix("gpu-") for p in props}
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader,nounits"], text=True)
        for line in apps.splitlines():
            gpu, pid = [x.strip() for x in line.split(",")]
            if gpu.lower().removeprefix("gpu-") in visible_ids and int(pid) != os.getpid():
                raise ValueError(f"visible GPU has another compute process {pid}; do not terminate it")
        for i in range(world):
            free, total = torch.cuda.mem_get_info(i)
            if total - free > 2 * 1024**3:
                raise ValueError(f"GPU {i} is occupied; no automatic termination")
    import transformers
    return {"gpu_name": props[0].name, "world": world, "torch": str(torch.__version__),
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "transformers": transformers.__version__,
            "nvidia_driver": Path("/proc/driver/nvidia/version").read_text().strip()}


def prompt_info(output, receipt):
    info = c.read(output / "prompts.json")
    if info["receipt_sha256"] != c.sha(output / "receipt.json"):
        raise ValueError("prompt/receipt mismatch")
    if info["prompts"] != {t: receipt["experts"][t]["prompt"] for t in receipt["config"]["tasks"]}:
        raise ValueError("prompt text changed")
    if set(info["tensors"]) != set(receipt["config"]["tasks"]):
        raise ValueError("prompt set incomplete")
    for row in info["tensors"].values():
        if c.sha(c.path(output, row["path"])) != row["sha256"]:
            raise ValueError("prompt tensor changed")
    return info


def prompts(output, receipt):
    import torch
    if (output / "prompts.json").exists():
        prompt_info(output, receipt)
        print("EXPERT_REPLAY_PROMPTS_REUSED", flush=True)
        return
    from flux2.text_encoder import Qwen3Embedder
    from flux2.sampling import batched_prc_txt
    from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
    torch.cuda.set_device(0)
    encoder = Qwen3Embedder.__new__(Qwen3Embedder)
    torch.nn.Module.__init__(encoder)
    model_path = c.path(ROOT, "models/qwen3-4b-fp8")
    encoder.model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="cuda:0",
        quantization_config=FineGrainedFP8Config(dequantize=True), local_files_only=True).eval().requires_grad_(False)
    encoder.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    encoder.max_length = 512
    entries, texts = {}, {}
    for task in receipt["config"]["tasks"]:
        text = receipt["experts"][task]["prompt"]
        marker = output / f"prompt_{task}.json"
        identity = {"prompt": text, "receipt_sha256": c.sha(output / "receipt.json")}
        if marker.exists():
            row = c.read(marker)
            c.validate_record(row, identity, output)
            name, file_hash = next(iter(row["files"].items()))
        else:
            with torch.inference_mode():
                encoded = encoder(["", text]).to(torch.bfloat16)
                ctx, ids = batched_prc_txt(encoded)
            if (ctx.shape != (2, 512, 7680) or ids.shape != (2, 512, 4)
                    or ids.dtype != torch.int64 or not torch.isfinite(ctx).all()):
                raise ValueError("unexpected prompt tensors")
            name, file_hash = save_tensors(output, f"prompt_{task}", {"cfg_context": ctx, "cfg_ids": ids})
            c.write_new(marker, {"identity": identity, "files": {name: file_hash}})
        entries[task] = {"path": name, "sha256": file_hash}
        texts[task] = text
        print(f"EXPERT_REPLAY_PROMPT_COMPLETE task={task}", flush=True)
    c.write_new(output / "prompts.json", {"receipt_sha256": c.sha(output / "receipt.json"),
        "tensors": entries, "prompts": texts, "encoding": "offline_FP8_checkpoint_dequantized_BF16",
        "historical_prompt_tensor_equality_claimed": False})


def lq_tensor(row, crop, device):
    import torch
    import numpy as np
    from PIL import Image
    if row["split"] != "train" or "gt_path" in row:
        raise ValueError("replay reads only frozen training LQ")
    p = c.path(ROOT, row["lq_path"])
    if c.sha(p) != row["lq_sha256"]:
        raise ValueError("training LQ hash mismatch")
    x, y = row["crop_xy"]
    with Image.open(p) as im:
        if im.size != (int(row["width"]), int(row["height"])):
            raise ValueError("training LQ dimensions changed")
        array = np.asarray(im.convert("RGB").crop((x, y, x + crop, y + crop))).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float().div(127.5).sub(1).to(device)


def quantize(task, lq, raw, alpha):
    from lowlight_restoration.scalar_residual import quantized_scalar_residual
    u8, first = quantized_scalar_residual(lq, raw, 1.0)
    if task == "haze":
        if alpha != 1:
            raise ValueError("Haze original protocol has no residual blend")
        return u8
    return quantized_scalar_residual(lq, first, alpha)[0]


def identity_for(task, row, receipt_sha, prompts_sha, rank):
    return {"key": f"{task}_{row['record_id']}", "task": task, "row": row,
            "receipt_sha256": receipt_sha, "prompts_sha256": prompts_sha, "rank": rank}


def run(output, receipt, world):
    import torch
    from safetensors.torch import load_file
    from flux2.util import load_flow_model, load_ae
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.lora import inject_lora, load_lora_checkpoint, LoRALinear
    from lowlight_restoration.existing_flux_experts import install_replay_bank, FLOW_CONTRACT
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch

    rank = int(os.environ["RANK"])
    if (int(os.environ["WORLD_SIZE"]) != world or int(os.environ["LOCAL_WORLD_SIZE"]) != world
            or int(os.environ["LOCAL_RANK"]) != rank):
        raise ValueError("single-node torchrun only")
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    task, rows = c.assigned(receipt["config"]["tasks"], receipt["rows"], rank, world)
    receipt_sha = c.sha(output / "receipt.json")
    prompt_data = prompt_info(output, receipt)
    prompt_sha = c.sha(output / "prompts.json")
    identities = {r["record_id"]: identity_for(task, r, receipt_sha, prompt_sha, rank) for r in rows}
    # Validate committed progress before skipping it. Never trust mere file existence.
    pending = []
    for r in rows:
        key = identities[r["record_id"]]["key"]
        original_path, bank_path = output / "original" / f"{key}.json", output / "bank" / f"{key}.json"
        if original_path.exists():
            c.validate_record(c.read(original_path), identities[r["record_id"]], output)
        if bank_path.exists():
            info = c.read(bank_path)
            c.validate_record(info, identities[r["record_id"]], output)
            if not original_path.exists() or info["original_receipt_sha256"] != c.sha(original_path):
                raise ValueError("bank result has no matching original")
        else:
            pending.append(r)
    if not pending:
        print(f"EXPERT_REPLAY_RANK_REUSED rank={rank} task={task}", flush=True)
        return
    cfg = receipt["config"]
    task_spec = receipt["experts"][task]
    alpha = task_spec["inference"].get("frozen_alpha", 1.)
    prompt = load_file(str(c.path(output, prompt_data["tensors"][task]["path"])), device=str(device))
    ae = load_ae("flux.2-klein-base-4b", device=device).eval().requires_grad_(False)
    model = load_flow_model("flux.2-klein-base-4b", device=device).eval().requires_grad_(False)
    spec = c.read(ROOT / "configs/foundir_blur_only_flux2_full_v1.json")["lora"]
    names = inject_lora(model, spec["target_patterns"], spec["rank"], spec["alpha"], spec["dropout"])
    if set(names) != set(expected_modules()):
        raise ValueError("original FLUX LoRA installation sites changed")
    load_lora_checkpoint(model, c.path(ROOT, task_spec["checkpoint"]["path"]))
    model.eval().requires_grad_(False)

    def restore(row):
        lq = lq_tensor(row, cfg["crop_size"], device)
        raw = restore_batch(model, ae, lq, prompt["cfg_context"], prompt["cfg_ids"], batched_prc_img,
                            seeds=[row["inference_seed"]], steps=cfg["steps"], reference_time=cfg["reference_time"])
        if not torch.isfinite(raw).all():
            raise ValueError("nonfinite decoded output")
        return raw, quantize(task, lq, raw, alpha)

    for row in pending:
        ident = identities[row["record_id"]]
        marker = output / "original" / f"{ident['key']}.json"
        if marker.exists():
            continue
        start = time.monotonic()
        raw, image = restore(row)
        name, file_hash = save_tensors(output, f"original_{ident['key']}", {"decoded": raw, "uint8": image})
        c.write_new(marker, {"identity": ident, "files": {name: file_hash}, "tensor": name,
            "seconds": time.monotonic() - start, "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3})
        print(f"EXPERT_REPLAY_ORIGINAL rank={rank} key={ident['key']}", flush=True)
        del raw, image
    # Remove only our original diagnostic wrappers; reuse identical frozen base tensors.
    for name in names:
        module = model.get_submodule(name)
        if not isinstance(module, LoRALinear):
            raise ValueError("unexpected wrapper before bank installation")
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, module.base)
    states = [(t, load_file(str(c.path(ROOT, receipt["experts"][t]["checkpoint"]["path"])), device="cpu"),
               32., FLOW_CONTRACT) for t in cfg["tasks"]]
    install_replay_bank(model, modules=expected_modules(), expert_states=states, expert_index=cfg["tasks"].index(task))
    del states
    gc.collect()
    torch.cuda.empty_cache()
    model.eval().requires_grad_(False)
    for row in pending:
        ident = identities[row["record_id"]]
        marker = output / "bank" / f"{ident['key']}.json"
        start = time.monotonic()
        original_path = output / "original" / f"{ident['key']}.json"
        original = c.read(original_path)
        c.validate_record(original, ident, output)
        tensors = load_file(str(c.path(output, original["tensor"])), device="cpu")
        raw, image = restore(row)
        raw_cpu, image_cpu = raw.cpu(), image.cpu()
        if raw_cpu.shape != tensors["decoded"].shape or image_cpu.shape != tensors["uint8"].shape:
            raise ValueError("saved tensor shape mismatch")
        difference = float((raw_cpu - tensors["decoded"]).abs().max())
        equal = torch.equal(image_cpu, tensors["uint8"])
        passed = difference <= cfg["rgb_max_abs_tolerance"] and equal
        name, file_hash = save_tensors(output, f"bank_{ident['key']}", {"decoded": raw_cpu, "uint8": image_cpu})
        c.write_new(marker, {"identity": ident, "files": {name: file_hash}, "tensor": name,
            "original_receipt_sha256": c.sha(original_path), "max_abs_rgb": difference,
            "uint8_equal": equal, "pass": passed, "seconds": time.monotonic() - start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3})
        print(f"EXPERT_REPLAY_BANK rank={rank} key={ident['key']} max_abs={difference} uint8_equal={equal}", flush=True)
        del raw, image, raw_cpu, image_cpu, tensors


def verify(output, receipt):
    import torch
    from safetensors.torch import load_file
    records, expected = [], []
    receipt_sha, prompts_sha = c.sha(output / "receipt.json"), c.sha(output / "prompts.json")
    prompt_info(output, receipt)
    for rank in range(receipt["world"]):
        task, rows = c.assigned(receipt["config"]["tasks"], receipt["rows"], rank, receipt["world"])
        for row in rows:
            ident = identity_for(task, row, receipt_sha, prompts_sha, rank)
            key = ident["key"]
            original_path, bank_path = output / "original" / f"{key}.json", output / "bank" / f"{key}.json"
            original, bank = c.read(original_path), c.read(bank_path)
            c.validate_record(original, ident, output); c.validate_record(bank, ident, output)
            if bank["original_receipt_sha256"] != c.sha(original_path):
                raise ValueError("original receipt changed")
            a, b = [load_file(str(c.path(output, info["tensor"])), device="cpu") for info in (original, bank)]
            for tensor in (a, b):
                if (set(tensor) != {"decoded", "uint8"} or tensor["decoded"].shape != (1, 3, 512, 512)
                        or tensor["decoded"].dtype != torch.float32 or tensor["uint8"].dtype != torch.uint8
                        or tensor["uint8"].shape != tensor["decoded"].shape
                        or not torch.isfinite(tensor["decoded"]).all()):
                    raise ValueError("invalid saved output tensors")
            difference = float((a["decoded"] - b["decoded"]).abs().max())
            equal = torch.equal(a["uint8"], b["uint8"])
            if difference != bank["max_abs_rgb"] or equal != bank["uint8_equal"]:
                raise ValueError("reported comparison does not match saved tensors")
            records.append(bank); expected.append(key)
    actual = {p.stem for p in (output / "bank").glob("*.json")}
    if actual != set(expected) or {p.stem for p in (output / "original").glob("*.json")} != set(expected):
        raise ValueError("extra/missing replay sidecars")
    passed = c.validate_summary(records, expected, receipt["config"]["rgb_max_abs_tolerance"])
    result = {"status": "full_rollout_static_selection_equivalence_pass" if passed else "full_rollout_equivalence_failed",
              "pass": passed, "receipt_sha256": receipt_sha, "prompts_sha256": prompts_sha,
              "records": len(records), "by_task": {t: {"records": sum(r["identity"]["task"] == t for r in records),
                  "max_abs_rgb": max(r["max_abs_rgb"] for r in records if r["identity"]["task"] == t),
                  "uint8_equal_count": sum(r["uint8_equal"] for r in records if r["identity"]["task"] == t)}
                  for t in receipt["config"]["tasks"]}, "comparisons": records,
              "gt_pixels_read": False, "training": False, "scientific_quality_gate_claimed": False,
              "rain_adapted": False, "dynamic_or_soft_router_qualified": False}
    c.write_new(output / "result.json", result)
    print(f"EXPERT_REPLAY_RESULT_VERIFIED pass={passed} records={len(records)}", flush=True)
    return 0 if passed else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["preflight", "hardware", "prepare", "prompts", "run", "verify"])
    p.add_argument("--run-id", default="replay8_v1")
    p.add_argument("--gpus", type=int, choices=[4, 8], default=8)
    p.add_argument("--asset-root", type=Path)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    if args.phase == "preflight":
        receipt = c.make_receipt(ROOT, args.asset_root or ROOT, args.gpus)
        result = {"status": "EXPERT_REPLAY_CPU_PREFLIGHT_PASS", "world": args.gpus,
                  "selection_digest": receipt["selection_digest"], "sources": receipt["sources"],
                  "records": {t: len(v) for t, v in receipt["rows"].items()},
                  "selected_record_ids": {t: [r["record_id"] for r in v] for t,v in receipt["rows"].items()},
                  "pixel_decoding": False, "gpu_used": False,
                  "asset_root": str((args.asset_root or ROOT).resolve())}
        if args.report:
            c.write_new(args.report, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    cfg = c.read(ROOT / c.CONFIG)
    if ROOT.resolve() != Path(cfg["active_root"]).resolve():
        raise ValueError("GPU execution is allowed only in the migrated workspace")
    if args.asset_root or args.report:
        raise ValueError("asset-root/report overrides are CPU-preflight-only")
    if args.phase == "hardware":
        print("EXPERT_REPLAY_HARDWARE_PASS " + json.dumps(hardware(args.gpus, require_idle=True)), flush=True)
        return 0
    output = c.run_path(ROOT, args.run_id)
    if args.phase == "prepare":
        import shutil
        if shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise ValueError("replay needs at least 5 GiB free; do not delete historical artifacts")
        receipt = c.make_receipt(ROOT, ROOT, args.gpus)
        c.write_new(output / "receipt.json", receipt)
        c.write_new(output / "runtime.json", hardware(args.gpus, require_idle=True))
        print("EXPERT_REPLAY_PREPARE_PASS", flush=True)
        return 0
    receipt = c.load_receipt(ROOT, output, args.gpus)
    if args.phase in ("prompts", "run"):
        if hardware(args.gpus, require_idle=False) != c.read(output / "runtime.json"):
            raise ValueError("runtime/GPU family changed; do not mix execution environments")
        os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT, "models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors")),
                          AE_MODEL_PATH=str(c.path(ROOT, "models/flux2-klein-base-4b/ae.safetensors")),
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.phase == "prompts":
        prompts(output, receipt)
    elif args.phase == "run":
        run(output, receipt, args.gpus)
    else:
        return verify(output, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
