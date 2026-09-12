#!/usr/bin/env python3
"""Evaluate five Blur-only FLUX.2 checkpoints on all 2,595 development records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))

from lowlight_restoration.lora import inject_lora, load_lora_checkpoint  # noqa: E402
from smoke.train_foundir_blur_only_flux2_overfit_ddp import (  # noqa: E402
    encode_prompt,
    lpips_value,
    psnr,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*values: object) -> str:
    return hashlib.sha256(":".join(map(str, values)).encode()).hexdigest()


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def write_once(path: Path, payload: Mapping[str, object] | str) -> None:
    if isinstance(payload, str):
        content = payload.encode()
    else:
        content = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to overwrite differing artifact: {path}")
        return
    with path.open("xb") as handle:
        handle.write(content)


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def validate_config(config_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    config = load_json(config_path)
    if (
        config.get("schema_version") != 1
        or config.get("status") != "advisor_authorized_full_blur_development_evaluation"
        or config.get("authorization", {}).get("hold_id")
        != "blur_only_full_training_completion-1c1a1897c39b4044"
        or config.get("authorization", {}).get("hold_sha256")
        != "2bb5e855e9936a0e0c8561efda0908af8574e528a9c52f9e493158a429fbf31b"
    ):
        raise RuntimeError("Blur full-development advisor authorization changed")
    scope = config["scope"]
    if scope != {
        "category": "01Blur",
        "split": "validation",
        "qwen_degradation_conditioning": False,
        "mixed_degradations": False,
        "pixel_refiner": False,
        "sealed_internal_accessed": False,
        "official_foundir_test_accessed": False,
    }:
        raise RuntimeError("Blur full-development scope changed")
    parent_path = workspace_path(config["parent_training_config"])
    if sha256(parent_path) != config["parent_training_config_sha256"]:
        raise RuntimeError("parent Blur-only training config changed")
    parent = load_json(parent_path)
    bundle_path = workspace_path(config["candidate_bundle"])
    if sha256(bundle_path) != config["candidate_bundle_sha256"]:
        raise RuntimeError("candidate checkpoint bundle changed")
    bundle = load_json(bundle_path)
    if bundle.get("checkpoints") != config.get("checkpoints"):
        raise RuntimeError("config and candidate checkpoint bundle disagree")
    checkpoints = config["checkpoints"]
    if [int(item["step"]) for item in checkpoints] != [2000, 4000, 6000, 8000, 10000]:
        raise RuntimeError("all five frozen Blur-only checkpoints are required")
    for item in checkpoints:
        checkpoint = workspace_path(item["path"])
        if not checkpoint.is_file() or sha256(checkpoint) != item["sha256"]:
            raise RuntimeError(f"candidate checkpoint changed: {checkpoint}")
        metadata = load_json(checkpoint.with_suffix(checkpoint.suffix + ".json"))
        if (
            int(metadata.get("step", -1)) != int(item["step"])
            or metadata.get("config_sha256") != config["parent_training_config_sha256"]
        ):
            raise RuntimeError(f"candidate checkpoint metadata changed: {checkpoint}")
    data = config["data"]
    inference = config["inference"]
    metrics = config["metrics"]
    distributed = config["distributed"]
    if (
        int(data.get("expected_records", -1)) != 2595
        or int(data.get("expected_source_groups", -1)) != 165
        or int(data.get("crop_size", -1)) != 512
        or data.get("crop_policy")
        != "one_deterministic_aligned_512_crop_per_record_shared_across_checkpoints"
        or inference.get("schedule") != "official_flux2_shifted_schedule"
        or int(inference.get("num_steps", -1)) != 50
        or float(inference.get("guidance", -1)) != 1.0
        or inference.get("quantization")
        != "round_clip_uint8_png_equivalent_before_metrics"
        or metrics.get("names") != ["psnr", "lpips"]
        or metrics.get("bootstrap_unit") != "source_group_equal_weight"
        or int(metrics.get("bootstrap_resamples", -1)) != 5000
        or int(distributed.get("required_world_size", -1)) != 8
        or distributed.get("allowed_gpu_name_fragments") != ["H100", "H200"]
        or distributed.get("evaluation_batch_size_per_rank") != {"H100": 2, "H200": 4}
    ):
        raise RuntimeError("Blur full-development evaluation contract changed")
    if (
        parent["model"]["name"] != "flux.2-klein-base-4b"
        or parent["scope"]["category"] != "01Blur"
        or parent["scope"]["qwen_degradation_conditioning"] is not False
        or parent["scope"]["mixed_degradations"] is not False
        or parent["scope"]["pixel_refiner"] is not False
    ):
        raise RuntimeError("parent model is no longer the authorized Blur-only FLUX.2 model")
    return config, parent


def load_development_rows(config: Mapping[str, object]) -> list[dict[str, str]]:
    data = config["data"]
    manifest = workspace_path(data["source_manifest"])
    if sha256(manifest) != data["source_manifest_sha256"]:
        raise RuntimeError("FoundIR pair manifest changed")
    with manifest.open(newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    isolated = [
        row for row in all_rows
        if row["category"] == "01Blur" and row["composition_size"] == "1"
    ]
    rows = sorted(
        (dict(row) for row in isolated if row["split"] == "validation"),
        key=lambda row: row["record_id"],
    )
    train_groups = {row["source_group"] for row in isolated if row["split"] == "train"}
    groups = {row["source_group"] for row in rows}
    if (
        len(rows) != int(data["expected_records"])
        or len(groups) != int(data["expected_source_groups"])
        or len({row["record_id"] for row in rows}) != len(rows)
        or groups & train_groups
    ):
        raise RuntimeError("isolated Blur development cardinality or source isolation changed")
    for row in rows:
        if min(int(row["width"]), int(row["height"])) < int(data["crop_size"]):
            raise RuntimeError(f"undersized isolated Blur development record: {row['record_id']}")
        if "/" in row["record_id"] or ".." in row["record_id"]:
            raise RuntimeError(f"unsafe record ID: {row['record_id']}")
    return rows


class RankSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world: int) -> None:
        self.values = range(rank, length, world)

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)


class BlurDevelopmentDataset(Dataset[dict[str, object]]):
    def __init__(self, rows: Sequence[Mapping[str, str]], *, crop_size: int, seed: int) -> None:
        self.rows = rows
        self.crop_size = crop_size
        self.seed = seed
        self._verified: set[str] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        arrays = []
        for field, hash_field in (("lq_path", "lq_sha256"), ("gt_path", "gt_sha256")):
            path = workspace_path(row[field])
            identity = f"{field}:{row['record_id']}"
            if identity not in self._verified:
                if sha256(path) != row[hash_field]:
                    raise RuntimeError(f"Blur development image changed: {identity}")
                self._verified.add(identity)
            with Image.open(path) as handle:
                image = handle.convert("RGB")
                if image.size != (int(row["width"]), int(row["height"])):
                    raise RuntimeError(f"Blur development dimensions changed: {row['record_id']}")
                arrays.append(np.asarray(image, dtype=np.uint8).copy())
        digest = bytes.fromhex(stable_digest(self.seed, "full_development_crop", row["record_id"]))
        height, width = arrays[1].shape[:2]
        y = int.from_bytes(digest[:8], "big") % (height - self.crop_size + 1)
        x = int.from_bytes(digest[8:16], "big") % (width - self.crop_size + 1)

        def tensor(array: np.ndarray) -> torch.Tensor:
            crop = array[y : y + self.crop_size, x : x + self.crop_size]
            return torch.from_numpy(crop.copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0)

        return {
            "index": index,
            "record_id": row["record_id"],
            "source_group": row["source_group"],
            "crop_xy": [x, y],
            "lq": tensor(arrays[0]),
            "gt": tensor(arrays[1]),
        }


def collate(items: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "index": [int(item["index"]) for item in items],
        "record_id": [str(item["record_id"]) for item in items],
        "source_group": [str(item["source_group"]) for item in items],
        "crop_xy": [list(item["crop_xy"]) for item in items],
        "lq": torch.stack([item["lq"] for item in items]),
        "gt": torch.stack([item["gt"] for item in items]),
    }


def expand_cfg_condition(
    cfg_context: torch.Tensor, cfg_ids: torch.Tensor, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg_context.shape[0] != 2 or cfg_ids.shape[0] != 2 or batch < 1:
        raise RuntimeError("CFG context must contain exactly empty and Blur prompt rows")
    context = torch.cat(
        [cfg_context[0:1].expand(batch, *cfg_context.shape[1:]),
         cfg_context[1:2].expand(batch, *cfg_context.shape[1:])]
    ).contiguous()
    ids = torch.cat(
        [cfg_ids[0:1].expand(batch, *cfg_ids.shape[1:]),
         cfg_ids[1:2].expand(batch, *cfg_ids.shape[1:])]
    ).contiguous()
    return context, ids


def deterministic_noise(reference: torch.Tensor, seeds: Sequence[int]) -> torch.Tensor:
    values = []
    for index, seed in enumerate(seeds):
        generator = torch.Generator(device=reference.device).manual_seed(int(seed))
        values.append(
            torch.randn(
                reference[index].shape,
                generator=generator,
                device=reference.device,
                dtype=torch.bfloat16,
            )
        )
    return torch.stack(values)


def restore_batch(
    model,
    autoencoder,
    lq: torch.Tensor,
    cfg_context: torch.Tensor,
    cfg_ids: torch.Tensor,
    pack_image,
    *,
    seeds: Sequence[int],
    steps: int,
    reference_time: int,
) -> torch.Tensor:
    from flux2.sampling import denoise_cfg, get_schedule, scatter_ids

    with torch.inference_mode():
        reference = autoencoder.encode(lq).to(torch.bfloat16)
        reference_tokens, reference_ids = pack_image(reference)
        reference_ids = reference_ids.clone()
        reference_ids[..., 0] = reference_time
        noise = deterministic_noise(reference, seeds)
        tokens, ids = pack_image(noise)
        context, context_ids = expand_cfg_condition(cfg_context, cfg_ids, lq.shape[0])
        schedule = get_schedule(steps, tokens.shape[1])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored = denoise_cfg(
                model,
                tokens,
                ids,
                context,
                context_ids,
                timesteps=schedule,
                guidance=1.0,
                img_cond_seq=reference_tokens,
                img_cond_seq_ids=reference_ids,
            )
        latent = torch.cat(scatter_ids(restored, ids)).squeeze(2)
        return autoencoder.decode(latent).float().clamp(-1, 1)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def verify_sidecar(
    path: Path,
    *,
    row: Mapping[str, str],
    step: int,
    checkpoint_sha: str,
    config_sha: str,
) -> dict[str, object]:
    payload = load_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("config_sha256") != config_sha
        or int(payload.get("checkpoint_step", -1)) != step
        or payload.get("checkpoint_sha256") != checkpoint_sha
        or payload.get("split") != "validation"
        or payload.get("record_id") != row["record_id"]
        or payload.get("source_group") != row["source_group"]
        or payload.get("category") != "01Blur"
        or payload.get("sealed_internal_accessed") is not False
        or payload.get("protected_foundir_test_accessed") is not False
        or not isinstance(payload.get("crop_xy"), list)
        or len(payload["crop_xy"]) != 2
        or not isinstance(payload.get("restored_uint8_sha256"), str)
        or len(payload["restored_uint8_sha256"]) != 64
    ):
        raise RuntimeError(f"Blur development sidecar identity changed: {path}")
    if not all(
        finite(payload[group][metric])
        for group in ("input", "restored")
        for metric in ("psnr", "lpips")
    ):
        raise RuntimeError(f"Blur development sidecar has invalid metrics: {path}")
    return payload


def source_group_bootstrap(
    rows: Sequence[Mapping[str, object]], *, resamples: int, confidence: float, seed: int
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["source_group"])].append(row)
    groups = sorted(grouped)
    values = np.asarray(
        [[
            np.mean([item["restored"]["psnr"] for item in grouped[group]]),
            np.mean([item["restored"]["lpips"] for item in grouped[group]]),
        ] for group in groups],
        dtype=np.float64,
    )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(groups), size=(resamples, len(groups)))
    draws = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail], axis=0)
    return {
        "unit": "source_group_equal_weight",
        "source_groups": len(groups),
        "resamples": resamples,
        "confidence": confidence,
        "group_balanced_mean": {"psnr": float(values[:, 0].mean()), "lpips": float(values[:, 1].mean())},
        "lower": {"psnr": float(lower[0]), "lpips": float(lower[1])},
        "upper": {"psnr": float(upper[0]), "lpips": float(upper[1])},
    }


def aggregate_checkpoint(
    rows: Sequence[Mapping[str, object]],
    *,
    step: int,
    checkpoint_path: str,
    checkpoint_sha: str,
    config: Mapping[str, object],
    config_sha: str,
) -> dict[str, object]:
    restored_psnr = np.asarray([row["restored"]["psnr"] for row in rows], dtype=np.float64)
    restored_lpips = np.asarray([row["restored"]["lpips"] for row in rows], dtype=np.float64)
    input_psnr = np.asarray([row["input"]["psnr"] for row in rows], dtype=np.float64)
    input_lpips = np.asarray([row["input"]["lpips"] for row in rows], dtype=np.float64)
    threshold = config["metrics"]["sow_stage2"]
    metrics = {
        "records": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "input_psnr": float(input_psnr.mean()),
        "input_lpips": float(input_lpips.mean()),
        "psnr": float(restored_psnr.mean()),
        "lpips": float(restored_lpips.mean()),
        "psnr_gain_db": float((restored_psnr - input_psnr).mean()),
        "lpips_improvement": float((input_lpips - restored_lpips).mean()),
        "psnr_improved_fraction": float((restored_psnr > input_psnr).mean()),
        "lpips_improved_fraction": float((restored_lpips < input_lpips).mean()),
    }
    bootstrap = source_group_bootstrap(
        rows,
        resamples=int(config["metrics"]["bootstrap_resamples"]),
        confidence=float(config["metrics"]["bootstrap_confidence"]),
        seed=int(stable_digest(config["metrics"]["bootstrap_seed"], step)[:16], 16),
    )
    count = int(config["metrics"]["worst_records_per_checkpoint"])
    return {
        "schema_version": 1,
        "status": "checkpoint_full_development_complete",
        "config_sha256": config_sha,
        "checkpoint_step": step,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "split": "validation",
        "metrics": metrics,
        "sow_stage2": threshold,
        "sow_checks": {
            "psnr": metrics["psnr"] >= float(threshold["psnr_minimum"]),
            "lpips": metrics["lpips"] <= float(threshold["lpips_maximum"]),
        },
        "source_group_bootstrap": bootstrap,
        "worst_record_ids": {
            "psnr": [row["record_id"] for row in sorted(rows, key=lambda item: item["restored"]["psnr"])[:count]],
            "lpips": [row["record_id"] for row in sorted(rows, key=lambda item: -item["restored"]["lpips"])[:count]],
        },
        "sealed_internal_accessed": False,
        "protected_foundir_test_accessed": False,
    }


def diagnose_overfitting(curve: Sequence[Mapping[str, object]]) -> dict[str, object]:
    final = curve[-1]
    earlier = curve[:-1]
    best_psnr = max(curve, key=lambda item: item["metrics"]["psnr"])
    best_lpips = min(curve, key=lambda item: item["metrics"]["lpips"])
    dominated = [
        item for item in earlier
        if item["metrics"]["psnr"] >= final["metrics"]["psnr"]
        and item["metrics"]["lpips"] <= final["metrics"]["lpips"]
        and (
            item["metrics"]["psnr"] > final["metrics"]["psnr"]
            or item["metrics"]["lpips"] < final["metrics"]["lpips"]
        )
    ]
    return {
        "definition": "late_overfitting_if_an_earlier_checkpoint_pareto_dominates_step_10000",
        "late_overfitting_detected": bool(dominated),
        "any_metric_overfitting_signal": (
            int(best_psnr["checkpoint_step"]) < int(final["checkpoint_step"])
            or int(best_lpips["checkpoint_step"]) < int(final["checkpoint_step"])
        ),
        "psnr_overfitting_signal": int(best_psnr["checkpoint_step"]) < int(final["checkpoint_step"]),
        "lpips_overfitting_signal": int(best_lpips["checkpoint_step"]) < int(final["checkpoint_step"]),
        "earlier_steps_dominating_final": [int(item["checkpoint_step"]) for item in dominated],
        "best_psnr_step": int(best_psnr["checkpoint_step"]),
        "best_psnr": float(best_psnr["metrics"]["psnr"]),
        "best_lpips_step": int(best_lpips["checkpoint_step"]),
        "best_lpips": float(best_lpips["metrics"]["lpips"]),
        "final_psnr_gap_from_best": float(final["metrics"]["psnr"] - best_psnr["metrics"]["psnr"]),
        "final_lpips_gap_from_best": float(final["metrics"]["lpips"] - best_lpips["metrics"]["lpips"]),
    }


def render_advisor_report(result: Mapping[str, object]) -> str:
    lines = [
        "# FLUX.2 Blur-only 全量 development 结果",
        "",
        "> DIAGNOSTIC ONLY / 非五类主线 final result。未访问 sealed internal 或 official FoundIR test。",
        "",
        f"- 数据：全部 {result['records']} 张 source-group-disjoint isolated Blur validation images（{result['source_groups']} 个 source groups）；每张固定一个 512×512 aligned crop。",
        "- 模型：FLUX.2 Klein Base 4B；Blur-only；DiT LoRA rank/alpha 32/32；无 Qwen degradation conditioning、无 mixed degradation、无 pixel refiner。",
        "- 推理：official shifted schedule，50 steps，guidance 1.0；所有 checkpoints 共用相同 crop 和 per-record seed。",
        f"- Evaluation config：`{result['config']}` (`{result['config_sha256']}`)。",
        f"- Candidate checkpoint bundle：`{result['candidate_bundle']}` (`{result['candidate_bundle_sha256']}`)。",
        "",
        "| Checkpoint | SHA-256 | PSNR | LPIPS | SOW PSNR >=25.9 | SOW LPIPS <=0.18 |",
        "|---:|---|---:|---:|:---:|:---:|",
    ]
    for item in result["checkpoint_curve"]:
        lines.append(
            f"| `{item['checkpoint']}` | `{item['checkpoint_sha256']}` | {item['metrics']['psnr']:.4f} | {item['metrics']['lpips']:.5f} | "
            f"{'pass' if item['sow_checks']['psnr'] else 'fail'} | {'pass' if item['sow_checks']['lpips'] else 'fail'} |"
        )
    diagnosis = result["overfitting_diagnosis"]
    lines.extend([
        "",
        f"- 明确 Pareto overfitting：{'是' if diagnosis['late_overfitting_detected'] else '否'}；定义为存在 PSNR、LPIPS 同时优于 step 10000 的 earlier checkpoint。",
        f"- 任一指标出现后期退化：{'是' if diagnosis['any_metric_overfitting_signal'] else '否'}。",
        f"- PSNR 最佳：step {diagnosis['best_psnr_step']}；LPIPS 最佳：step {diagnosis['best_lpips_step']}。",
        "- 请求 advisor 判断：根据全量 curve 和已发送的视觉结果，下一步是继续 Blur-only 调整，还是先审计 hallucination/training-inference code。",
    ])
    return "\n".join(lines) + "\n"


def run_preflight(config_path: Path) -> None:
    config, _ = validate_config(config_path)
    rows = load_development_rows(config)
    visual_root = workspace_path("checkpoints/foundir_blur_only_flux2_full_v1/fixed_validation/training_triptychs")
    for item in config["checkpoints"]:
        directory = visual_root / f"step_{int(item['step']):06d}"
        if not (directory / "metrics.json").is_file() or len(list(directory.glob("*.png"))) != 16:
            raise RuntimeError(f"fixed visual evidence is incomplete: {directory}")
    print(json.dumps({
        "status": "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_PREFLIGHT_PASS",
        "records": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "checkpoints": [item["step"] for item in config["checkpoints"]],
        "sealed_internal_accessed": False,
        "protected_foundir_test_accessed": False,
    }, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config_path = workspace_path(args.config)
    if args.preflight:
        run_preflight(config_path)
        return 0
    config, parent = validate_config(config_path)
    rows = load_development_rows(config)
    config_sha = sha256(config_path)
    output = workspace_path(config["runtime"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend=str(config["distributed"]["backend"]), device_id=device)
    started = time.monotonic()
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        if world != 8 or torch.cuda.device_count() != 8:
            raise RuntimeError("full Blur development evaluation requires exactly eight visible GPUs")
        properties = torch.cuda.get_device_properties(device)
        profiles = [name for name in ("H100", "H200") if name in properties.name]
        if len(profiles) != 1 or properties.total_memory < int(config["distributed"]["minimum_memory_gib"]) * 1024**3:
            raise RuntimeError(f"unsupported GPU for Blur development evaluation: {properties.name}")
        profile = profiles[0]
        gathered: list[object | None] = [None] * world
        dist.all_gather_object(gathered, profile)
        if set(gathered) != {profile}:
            raise RuntimeError(f"mixed GPU profiles are forbidden: {gathered}")
        batch_size = int(config["distributed"]["evaluation_batch_size_per_rank"][profile])
        os.environ.update(
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TORCH_HOME=str(workspace_path(config["runtime"]["torch_home"])),
            KLEIN_4B_BASE_MODEL_PATH=str(workspace_path(parent["model"]["flow_weights"])),
            AE_MODEL_PATH=str(workspace_path(parent["model"]["autoencoder_weights"])),
            QWEN3_MODEL_PATH=str(workspace_path(parent["model"]["text_encoder"])),
        )
        _, _, cfg_context, cfg_ids = encode_prompt(parent, device)
        from flux2.sampling import batched_prc_img
        from flux2.util import load_ae, load_flow_model
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        autoencoder = load_ae(parent["model"]["name"], device=device).eval().requires_grad_(False)
        model = load_flow_model(parent["model"]["name"], device=device).eval().requires_grad_(False)
        names = inject_lora(
            model,
            list(parent["lora"]["target_patterns"]),
            rank=int(parent["lora"]["rank"]),
            alpha=float(parent["lora"]["alpha"]),
            dropout=float(parent["lora"]["dropout"]),
        )
        if len(names) != int(parent["lora"]["expected_module_count"]):
            raise RuntimeError("FLUX.2 Blur-only LoRA target count changed")
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
        dataset = BlurDevelopmentDataset(
            rows,
            crop_size=int(config["data"]["crop_size"]),
            seed=int(config["data"]["crop_seed"]),
        )
        workers = int(config["distributed"]["data_loader_workers_per_rank"])
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=RankSampler(len(rows), rank, world),
            num_workers=workers,
            prefetch_factor=2 if workers else None,
            persistent_workers=bool(workers),
            pin_memory=True,
            collate_fn=collate,
            drop_last=False,
        )
        aggregates = []
        for candidate in config["checkpoints"]:
            step = int(candidate["step"])
            step_dir = output / f"step_{step:06d}"
            sidecar_dir = step_dir / "per_record"
            aggregate_path = step_dir / "metrics.json"
            if aggregate_path.exists():
                aggregate = load_json(aggregate_path)
                if (
                    aggregate.get("config_sha256") != config_sha
                    or aggregate.get("checkpoint_sha256") != candidate["sha256"]
                    or int(aggregate.get("metrics", {}).get("records", -1)) != len(rows)
                ):
                    raise RuntimeError(f"completed checkpoint aggregate identity changed: {aggregate_path}")
                aggregates.append(aggregate)
                dist.barrier()
                continue
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            load_lora_checkpoint(model, workspace_path(candidate["path"]))
            model.eval()
            for batch_index, batch in enumerate(loader):
                missing_positions = []
                for position, index in enumerate(batch["index"]):
                    sidecar = sidecar_dir / f"{batch['record_id'][position]}.json"
                    if sidecar.exists():
                        verify_sidecar(
                            sidecar,
                            row=rows[index],
                            step=step,
                            checkpoint_sha=str(candidate["sha256"]),
                            config_sha=config_sha,
                        )
                    else:
                        missing_positions.append(position)
                if not missing_positions:
                    continue
                selection = torch.tensor(missing_positions, dtype=torch.long)
                lq = batch["lq"].index_select(0, selection).to(device, non_blocking=True)
                gt = batch["gt"].index_select(0, selection).to(device, non_blocking=True)
                record_ids = [batch["record_id"][position] for position in missing_positions]
                seeds = [
                    int(stable_digest(config["inference"]["seed"], "blur_full_development", record_id)[:16], 16)
                    for record_id in record_ids
                ]
                restored = restore_batch(
                    model,
                    autoencoder,
                    lq,
                    cfg_context,
                    cfg_ids,
                    batched_prc_img,
                    seeds=seeds,
                    steps=int(config["inference"]["num_steps"]),
                    reference_time=int(config["inference"]["reference_time_coordinate"]),
                )
                restored_u8 = restored.detach().cpu().clamp(-1, 1).add(1).mul(127.5).round().byte()
                restored_q = restored_u8.float().div_(127.5).sub_(1.0).to(device)
                for local, position in enumerate(missing_positions):
                    index = batch["index"][position]
                    input_metrics = {
                        "psnr": psnr(gt[local : local + 1], lq[local : local + 1]),
                        "lpips": lpips_value(metric, gt[local : local + 1], lq[local : local + 1]),
                    }
                    metric.reset()
                    restored_metrics = {
                        "psnr": psnr(gt[local : local + 1], restored_q[local : local + 1]),
                        "lpips": lpips_value(metric, gt[local : local + 1], restored_q[local : local + 1]),
                    }
                    metric.reset()
                    image_bytes = restored_u8[local].permute(1, 2, 0).contiguous().numpy().tobytes()
                    payload = {
                        "schema_version": 1,
                        "status": "complete",
                        "config_sha256": config_sha,
                        "checkpoint_step": step,
                        "checkpoint_sha256": candidate["sha256"],
                        "split": "validation",
                        "record_id": rows[index]["record_id"],
                        "source_group": rows[index]["source_group"],
                        "category": "01Blur",
                        "crop_xy": batch["crop_xy"][position],
                        "inference_seed": seeds[local],
                        "input": input_metrics,
                        "restored": restored_metrics,
                        "restored_uint8_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "sealed_internal_accessed": False,
                        "protected_foundir_test_accessed": False,
                    }
                    write_once(sidecar_dir / f"{rows[index]['record_id']}.json", payload)
                if rank == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
                    print(
                        f"FOUNDIR_BLUR_FULL_DEVELOPMENT step={step} rank0_batch={batch_index+1}/{len(loader)} "
                        f"batch_size={batch_size}",
                        flush=True,
                    )
            dist.barrier()
            if rank == 0:
                expected = [sidecar_dir / f"{row['record_id']}.json" for row in rows]
                if set(sidecar_dir.glob("*.json")) != set(expected):
                    raise RuntimeError(f"checkpoint {step} sidecar set is incomplete or contaminated")
                result_rows = [
                    verify_sidecar(
                        path,
                        row=row,
                        step=step,
                        checkpoint_sha=str(candidate["sha256"]),
                        config_sha=config_sha,
                    )
                    for row, path in zip(rows, expected, strict=True)
                ]
                aggregate = aggregate_checkpoint(
                    result_rows,
                    step=step,
                    checkpoint_path=str(candidate["path"]),
                    checkpoint_sha=str(candidate["sha256"]),
                    config=config,
                    config_sha=config_sha,
                )
                write_once(aggregate_path, aggregate)
                print(
                    "FOUNDIR_BLUR_FULL_DEVELOPMENT_CHECKPOINT_COMPLETE "
                    + json.dumps({
                        "step": step,
                        "psnr": aggregate["metrics"]["psnr"],
                        "lpips": aggregate["metrics"]["lpips"],
                        "sow_checks": aggregate["sow_checks"],
                    }, sort_keys=True),
                    flush=True,
                )
            dist.barrier()
            aggregates.append(load_json(aggregate_path))

        if rank == 0:
            input_pairs = {
                (round(item["metrics"]["input_psnr"], 12), round(item["metrics"]["input_lpips"], 12))
                for item in aggregates
            }
            if len(input_pairs) != 1:
                raise RuntimeError("input baseline changed across checkpoints")
            final = aggregates[-1]
            result = {
                "schema_version": 1,
                "status": "blur_only_full_development_evaluation_complete",
                "automatic_gate_pass": all(final["sow_checks"].values()),
                "config": config_path.relative_to(ROOT).as_posix(),
                "config_sha256": config_sha,
                "checkpoint": final["checkpoint"],
                "checkpoint_sha256": final["checkpoint_sha256"],
                "candidate_bundle": config["candidate_bundle"],
                "candidate_bundle_sha256": config["candidate_bundle_sha256"],
                "model": "FLUX.2 Klein Base 4B isolated-Blur DiT LoRA rank/alpha 32/32",
                "split": "source-group-disjoint development validation",
                "records": len(rows),
                "source_groups": len({row["source_group"] for row in rows}),
                "hardware_profile": profile,
                "gpu_name": properties.name,
                "batch_size_per_rank": batch_size,
                "checkpoint_curve": aggregates,
                "overfitting_diagnosis": diagnose_overfitting(aggregates),
                "summary": {
                    "reported_checkpoint_step": int(final["checkpoint_step"]),
                    "task_metrics": {"blur": {
                        "psnr": final["metrics"]["psnr"],
                        "lpips": final["metrics"]["lpips"],
                    }},
                },
                "qwen_degradation_conditioning": False,
                "mixed_degradations": False,
                "pixel_refiner": False,
                "sealed_internal_accessed": False,
                "protected_foundir_test_accessed": False,
                "wall_seconds": time.monotonic() - started,
            }
            write_once(output / "result.json", result)
            write_once(output / "advisor_report.md", render_advisor_report(result))
            print(
                "FOUNDIR_BLUR_ONLY_FULL_DEVELOPMENT_COMPLETE "
                + json.dumps({
                    "final_psnr": final["metrics"]["psnr"],
                    "final_lpips": final["metrics"]["lpips"],
                    **result["overfitting_diagnosis"],
                }, sort_keys=True),
                flush=True,
            )
        dist.barrier()
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
