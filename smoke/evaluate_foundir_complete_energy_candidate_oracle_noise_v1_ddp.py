#!/usr/bin/env python3
"""GT-only feasibility audit for a complete strong-residual candidate pool."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.proposal_residual_basis import ProposalResidualBasisRestormer  # noqa: E402
from lowlight_restoration.task_native_restormer import (  # noqa: E402
    TASK_NATIVE_RESTORMER_SPECS,
    load_task_native_restormer,
)
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    atomic_json,
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
)
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import (  # noqa: E402
    build_lpips,
    lpips_values,
    run_frozen,
)

CONFIG_PATH = ROOT / "configs/foundir_complete_energy_candidate_oracle_noise_v1.json"


def deployable_candidates(proposal: Tensor, fidelity: Tensor, perceptual: Tensor) -> dict[str, Tensor]:
    return {
        "proposal": proposal,
        "fidelity": fidelity,
        "perceptual": perceptual,
        "under_fidelity_0.5": (proposal + .5 * (fidelity - proposal)).clamp(0, 1),
        "over_fidelity_1.25": (proposal + 1.25 * (fidelity - proposal)).clamp(0, 1),
        "fidelity_0.75_perceptual_0.25": (.75 * fidelity + .25 * perceptual).clamp(0, 1),
        "fidelity_0.5_perceptual_0.5": (.5 * fidelity + .5 * perceptual).clamp(0, 1),
    }


def patch_oracle(candidates: Tensor, target: Tensor, patch: int) -> tuple[Tensor, Tensor]:
    """Select the minimum-MSE candidate independently on each patch."""
    if candidates.ndim != 5 or candidates.shape[0] != target.shape[0]:
        raise ValueError("candidate tensor must be [B,K,3,H,W]")
    batch, count, _, height, width = candidates.shape
    if height % patch or width % patch:
        raise ValueError("oracle patch size must divide candidate dimensions")
    error = (candidates - target[:, None]).square().mean(2)
    pooled = F.avg_pool2d(error.flatten(0, 1)[:, None], patch, patch)
    pooled = pooled.reshape(batch, count, *pooled.shape[-2:])
    winner = pooled.argmin(1)
    weights = F.one_hot(winner, num_classes=count).permute(0, 3, 1, 2).to(candidates.dtype)
    weights = weights.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
    selected = (candidates * weights[:, :, None]).sum(1)
    return selected, winner


def load_basis(config: dict, base: dict, device: torch.device) -> ProposalResidualBasisRestormer:
    checkpoint = ROOT / config["residual_checkpoint"]
    if file_sha256(checkpoint) != config["residual_checkpoint_sha256"]:
        raise RuntimeError("complete-candidate audit residual checkpoint identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["step"]) != int(config["residual_checkpoint_step"]):
        raise RuntimeError("complete-candidate audit residual step changed")
    model = ProposalResidualBasisRestormer(
        residual_bound=float(base["model"]["residual_bound"]),
        initial_output_scale=float(base["model"]["initial_output_scale"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def load_negative_models(config: dict, device: torch.device) -> dict[str, torch.nn.Module]:
    result = {}
    specification = config["contrastive_negative_models"]
    directory = ROOT / specification["checkpoint_dir"]
    for role in ("under_restored", "wrong_expert", "detail_damaged"):
        name = specification[role]
        model_spec = TASK_NATIVE_RESTORMER_SPECS[name]
        if file_sha256(directory / model_spec.filename) != model_spec.sha256:
            raise RuntimeError(f"complete-candidate audit {role} identity changed")
        model, _ = load_task_native_restormer(name, directory)
        result[role] = model.to(device).eval().requires_grad_(False)
    return result


def infer_deployable(proposal_model, basis, observation: Tensor) -> dict[str, Tensor]:
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        proposal = run_frozen(proposal_model, observation).add(1).mul(.5)
        fidelity, perceptual = basis(observation, proposal)
    return deployable_candidates(proposal.float(), fidelity.float(), perceptual.float())


def scalar_metrics(lpips_network, candidates: dict[str, Tensor], target: Tensor) -> dict[str, dict[str, float]]:
    labels, values = list(candidates), tuple(candidates.values())
    perceptual = lpips_values(
        lpips_network, tuple(value.mul(2).sub(1) for value in values), target.mul(2).sub(1)
    )[0]
    result = {}
    for index, label in enumerate(labels):
        mse = (values[index] - target).square().mean()
        result[label] = {
            "psnr": float(-10 * torch.log10(mse.clamp_min(1e-12))),
            "lpips": float(perceptual[index]),
        }
    return result


def summarize(records: list[dict], config: dict) -> dict:
    labels = config["deployable_candidates"]
    fixed = {}
    for label in labels:
        fixed[label] = {
            metric: sum(row["fixed"][label][metric] for row in records) / len(records)
            for metric in ("psnr", "lpips")
        }
    fidelity = fixed["fidelity"]
    for value in fixed.values():
        value["psnr_gain_over_fidelity_db"] = value["psnr"] - fidelity["psnr"]
        value["lpips_change_from_fidelity"] = value["lpips"] - fidelity["lpips"]
    patch_results = {}
    gates = config["feasibility_gate"]
    for patch in config["patch_sizes"]:
        key = str(patch)
        psnr = sum(row["oracle"][key]["psnr"] for row in records) / len(records)
        lpips = sum(row["oracle"][key]["lpips"] for row in records) / len(records)
        counts = [sum(row["oracle"][key]["win_counts"][i] for row in records) for i in range(len(labels))]
        total = sum(counts)
        fractions = {label: counts[i] / total for i, label in enumerate(labels)}
        stable = sum(row["oracle"][key]["stable"] for row in records)
        compared = sum(row["oracle"][key]["compared"] for row in records)
        candidates_over_one_percent = sum(value >= .01 for value in fractions.values())
        result = {
            "psnr": psnr, "lpips": lpips,
            "psnr_gain_over_fidelity_db": psnr - fidelity["psnr"],
            "lpips_change_from_fidelity": lpips - fidelity["lpips"],
            "win_fractions": fractions,
            "candidates_with_at_least_one_percent_wins": candidates_over_one_percent,
            "horizontal_flip_winner_stability": stable / compared,
        }
        result["feasibility_pass"] = (
            result["psnr_gain_over_fidelity_db"] >= gates["minimum_psnr_gain_over_fidelity_db"]
            and result["lpips"] <= gates["maximum_lpips"]
            and candidates_over_one_percent >= gates["minimum_candidates_with_one_percent_wins"]
            and result["horizontal_flip_winner_stability"] >= gates["minimum_flip_stability"][key]
        )
        patch_results[key] = result
    negative = {}
    for label in ("under_restored", "wrong_expert", "detail_damaged", "repeat_proposal"):
        psnr = sum(row["negative"][label]["psnr"] for row in records) / len(records)
        lpips = sum(row["negative"][label]["lpips"] for row in records) / len(records)
        negative[label] = {
            "psnr": psnr, "lpips": lpips,
            "psnr_difference_from_fidelity_db": psnr - fidelity["psnr"],
            "lpips_difference_from_fidelity": lpips - fidelity["lpips"],
        }
    return {
        "fixed_candidate_means": fixed,
        "patch_oracles": patch_results,
        "contrastive_negative_means": negative,
        "any_patch_protocol_feasibility_pass": any(v["feasibility_pass"] for v in patch_results.values()),
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    config = json.loads(CONFIG_PATH.read_text())
    if world != int(config["hardware"]["world_size"]):
        raise RuntimeError("complete-candidate audit requires exactly eight local GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    names = [None] * world
    dist.all_gather_object(names, torch.cuda.get_device_name(local_rank))
    families = [f for f in config["hardware"]["supported_families"] if all(f in str(n).upper() for n in names)]
    if len(families) != 1:
        raise RuntimeError(f"complete-candidate audit requires homogeneous H100/H200, got {names}")
    base = json.loads((ROOT / config["base_config"]).read_text())
    if file_sha256(ROOT / base["data"]["manifest"]) != base["data"]["manifest_sha256"]:
        raise RuntimeError("complete-candidate audit manifest identity changed")
    proposal_model = load_frozen_proposal(base, device)
    basis = load_basis(config, base, device)
    negative_models = load_negative_models(config, device)
    lpips_network = build_lpips(device)
    rows = load_task_rows(base)
    validation = fixed_validation_rows(
        rows["validation"], int(base["data"]["validation_subset_records"]),
        int(base["data"]["validation_crop_seed"]),
    )
    labels = config["deployable_candidates"]
    local = []
    for index in range(rank, len(validation), world):
        row = validation[index]
        lq, gt, _, _ = load_audit_crop(
            row, crop_size=int(base["data"]["validation_crop_size"]),
            seed=int(base["data"]["validation_crop_seed"]),
        )
        observation = lq[None].to(device).add(1).mul(.5)
        target = gt[None].to(device).add(1).mul(.5)
        candidates = infer_deployable(proposal_model, basis, observation)
        if list(candidates) != labels:
            raise RuntimeError("complete-candidate label order changed")
        stacked = torch.stack(tuple(candidates.values()), dim=1)
        fixed = scalar_metrics(lpips_network, candidates, target)
        oracle = {}
        flipped_observation, flipped_target = observation.flip(-1), target.flip(-1)
        flipped_candidates = infer_deployable(proposal_model, basis, flipped_observation)
        flipped_stacked = torch.stack(tuple(flipped_candidates.values()), dim=1)
        for patch in config["patch_sizes"]:
            output, winner = patch_oracle(stacked, target, int(patch))
            _, flipped_winner = patch_oracle(flipped_stacked, flipped_target, int(patch))
            aligned = flipped_winner.flip(-1)
            values = scalar_metrics(lpips_network, {"oracle": output}, target)["oracle"]
            values.update({
                "win_counts": torch.bincount(winner.flatten(), minlength=len(labels)).cpu().tolist(),
                "stable": int((winner == aligned).sum()),
                "compared": winner.numel(),
            })
            oracle[str(patch)] = values
        with torch.inference_mode():
            negative = {}
            for label, model in negative_models.items():
                negative[label] = run_frozen(model, observation).add(1).mul(.5)
            negative["repeat_proposal"] = run_frozen(proposal_model, candidates["proposal"]).add(1).mul(.5)
        negative_metrics = scalar_metrics(lpips_network, negative, target)
        local.append({
            "index": index, "record_id": row["record_id"], "fixed": fixed,
            "oracle": oracle, "negative": negative_metrics,
        })
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        records = sorted((row for shard in gathered if shard for row in shard), key=lambda row: row["index"])
        if len(records) != int(base["data"]["validation_subset_records"]):
            raise RuntimeError("complete-candidate audit validation cardinality changed")
        summary = summarize(records, config)
        result = {
            "schema_version": 1,
            "status": "complete_energy_candidate_gt_oracle_audit_complete",
            "warning": "GT-only feasibility upper bound; not deployable model performance",
            "records": len(records), "gpu_names": names,
            "residual_checkpoint": config["residual_checkpoint"],
            "residual_checkpoint_sha256": config["residual_checkpoint_sha256"],
            "deployable_candidates": labels,
            "patch_sizes": config["patch_sizes"],
            "feasibility_gate": config["feasibility_gate"],
            **summary, **config["access"],
        }
        output = ROOT / config["output"]
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing audit: {output}")
        atomic_json(output, result)
        print("FOUNDIR_COMPLETE_ENERGY_CANDIDATE_ORACLE_COMPLETE " + json.dumps(summary, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
