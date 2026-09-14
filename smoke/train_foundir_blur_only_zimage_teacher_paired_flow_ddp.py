#!/usr/bin/env python3
"""Train the corrected Z-Image Base teacher on a paired LQ-to-GT latent flow."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/diffsynth-studio"))

from lowlight_restoration.foundir_zimage_paired_flow import (  # noqa: E402
    load_aligned_pair,
    load_config,
    load_frozen_rows,
    model_paths,
    paired_flow_state,
    paired_flow_target,
    sha256,
    validate_gpu_inventory,
    verify_model_artifacts,
    write_once,
)
from diffsynth.core import ModelConfig  # noqa: E402
from diffsynth.diffusion import DiffusionTrainingModule  # noqa: E402
from diffsynth.pipelines.z_image import ZImagePipeline  # noqa: E402


class FixedBlurDataset(Dataset):
    def __init__(self, rows, crop_size: int, total_items: int, prompt: str):
        self.rows = list(rows)
        self.crop_size = crop_size
        self.total_items = total_items
        self.prompt = prompt

    def __len__(self):
        return self.total_items

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        lq, gt = load_aligned_pair(row, self.crop_size)
        return {"record_id": row["record_id"], "lq": lq, "gt": gt, "prompt": self.prompt}


class ZImagePairedFlowTeacherModule(DiffusionTrainingModule):
    def __init__(self, config, device):
        super().__init__()
        paths = model_paths(config)
        self.pipe = ZImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=[
                ModelConfig(path=[str(path) for path in paths["transformer"]]),
                ModelConfig(path=[str(path) for path in paths["text_encoder"]]),
                ModelConfig(path=str(paths["vae"])),
            ],
            tokenizer_config=ModelConfig(path=str(paths["tokenizer"])),
            enable_npu_patch=False,
        )
        if self.pipe.controlnet is not None:
            raise RuntimeError("paired-flow v2 unexpectedly loaded a ControlNet")
        lora = config["lora"]
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=None,
            lora_base_model="dit",
            lora_target_modules=",".join(lora["target_modules"]),
            lora_rank=int(lora["rank"]),
            task="sft",
        )
        self.scheduler_shift = float(config["training"]["scheduler_shift"])
        self.pipe.scheduler.set_timesteps(1000, training=True, shift=self.scheduler_shift)

    def _encode_lq(self, image):
        self.pipe.load_models_to_device(["vae_encoder"])
        with torch.no_grad():
            return self.pipe.vae_encoder(self.pipe.preprocess_image(image))

    def forward(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["gt"],
            "height": data["gt"].height,
            "width": data["gt"].width,
            "cfg_scale": 1.0,
            "rand_device": self.pipe.device,
            "controlnet_inputs": None,
            "use_gradient_checkpointing": True,
            "use_gradient_checkpointing_offload": False,
        }
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega
            )

        gt_latent = inputs_shared["input_latents"].detach()
        lq_latent = self._encode_lq(data["lq"]).detach()
        timestep_id = torch.randint(0, len(self.pipe.scheduler.timesteps), (1,))
        sigma = self.pipe.scheduler.sigmas[timestep_id].to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
        inputs_shared["latents"] = paired_flow_state(gt_latent, lq_latent, sigma)
        target = paired_flow_target(gt_latent, lq_latent)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        prediction = self.pipe.model_fn(
            **models,
            **inputs_shared,
            **inputs_posi,
            timestep=timestep,
        )
        loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        return loss * self.pipe.scheduler.training_weight(timestep)


def save_lora_checkpoint(accelerator: Accelerator, model, path: Path, config_sha256: str, step: int) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        state = {
            name.removeprefix("pipe.dit."): parameter.detach().cpu().contiguous()
            for name, parameter in unwrapped.named_parameters()
            if parameter.requires_grad
        }
        if not state or any("lora_" not in name for name in state):
            raise RuntimeError("checkpoint export contains non-LoRA or no trainable parameters")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f"refusing to overwrite checkpoint: {path}")
        save_file(state, path, metadata={"config_sha256": config_sha256, "step": str(step)})
        print(f"ZIMAGE_PAIRED_FLOW_CHECKPOINT step={step} path={path} sha256={sha256(path)}", flush=True)
    accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    if accelerator.num_processes != int(config["distributed"]["required_world_size"]):
        raise RuntimeError("torchrun world size does not match the frozen eight-GPU contract")
    verify_model_artifacts(config, large_hashes=accelerator.is_main_process)
    accelerator.wait_for_everyone()
    train_rows, _ = load_frozen_rows(config, verify_images=True)
    names = [torch.cuda.get_device_properties(index).name for index in range(torch.cuda.device_count())]
    memory = [torch.cuda.get_device_properties(index).total_memory // (1024 * 1024) for index in range(torch.cuda.device_count())]
    hardware = validate_gpu_inventory(names, memory, config)
    seed = int(config["training"]["seed"])
    set_seed(seed, device_specific=True)
    random.seed(seed + accelerator.process_index)
    np.random.seed(seed + accelerator.process_index)

    output = ROOT / str(config["runtime"]["output_dir"])
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
        existing = [path for path in output.iterdir() if path.name not in {"launch_metadata.json", "preflight.json"}]
        if existing:
            raise RuntimeError("paired-flow output directory is nonempty; inspect it instead of duplicating the run")
    accelerator.wait_for_everyone()

    max_steps = int(config["training"]["max_steps"])
    world = int(config["distributed"]["required_world_size"])
    dataset = FixedBlurDataset(
        train_rows,
        int(config["data"]["crop_size"]),
        max_steps * world,
        str(config["model"]["prompt"]),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda batch: batch[0])
    model = ZImagePairedFlowTeacherModule(config, accelerator.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count <= 0:
        raise RuntimeError("Z-Image paired-flow teacher has no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    config_digest = sha256(config_path)
    save_steps = set(int(step) for step in config["training"]["save_steps"])
    started = time.time()
    metrics_path = output / "training_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    for step, data in enumerate(dataloader, start=1):
        if step > max_steps:
            break
        with accelerator.accumulate(model):
            loss = model(data)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, float(config["training"]["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        reduced = accelerator.reduce(loss.detach().float(), reduction="mean")
        if accelerator.is_main_process:
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"step": step, "loss": float(reduced), "elapsed_seconds": time.time() - started}) + "\n")
        if step == 1 or step % 10 == 0:
            accelerator.print(f"ZIMAGE_PAIRED_FLOW_TRAIN step={step}/{max_steps} loss={float(reduced):.8f}")
        if step in save_steps:
            save_lora_checkpoint(accelerator, model, output / f"step_{step:06d}.safetensors", config_digest, step)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        summary = {
            "schema_version": 2,
            "status": "zimage_blur_teacher_paired_flow_tiny_overfit_training_complete",
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": config_digest,
            "hardware_profile": hardware,
            "world_size": world,
            "global_batch_size": int(config["training"]["global_batch_size"]),
            "steps": max_steps,
            "trainable_parameters": trainable_count,
            "conditioning": config["implementation"]["conditioning"],
            "scheduler_shift": float(config["training"]["scheduler_shift"]),
            "controlnet_loaded": False,
            "wall_seconds": time.time() - started,
            "sealed_internal_accessed": False,
            "official_foundir_test_accessed": False,
        }
        write_once(output / "training_result.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode())
        print("ZIMAGE_PAIRED_FLOW_TRAINING_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
