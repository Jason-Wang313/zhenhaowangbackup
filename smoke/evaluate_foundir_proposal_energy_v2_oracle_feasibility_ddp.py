#!/usr/bin/env python3
"""GT-only development oracle for proposal/direct residual feasibility.

This diagnostic never produces a deployable result.  It asks whether the
already learned direct residual contains a combination that can clear the
frozen +.30 dB / -.005 LPIPS development gate.  Targets are used only to
select output-space mixtures on the fixed source-disjoint validation subset.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.proposal_relative_energy_v2 import EnergyAndDirectControlsV2  # noqa: E402
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
)
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import (  # noqa: E402
    CONFIG_PATH,
    build_lpips,
    descriptor_for_row,
    lpips_values,
    run_frozen,
)

DEFAULT_CHECKPOINT = ROOT / "checkpoints/foundir_proposal_relative_energy_noise_v2/noise_energy_v2_formal1/step_005118.pt"
DEFAULT_CHECKPOINT_SHA256 = "b9444ff2c3d4dedfbad2022c4d3dbb74b193fd6e64d2dcf52cb33275fa438bd1"
ALPHAS = tuple(index / 40 for index in range(41))
PATCH_SIZES = (8, 16, 32, 64)
PSNR_GATE = 0.30
LPIPS_GATE = 0.005


def patch_alpha_candidate(proposal: Tensor, direct: Tensor, target: Tensor, patch: int, smooth: bool) -> Tensor:
    residual = direct - proposal
    desired = target - proposal
    numerator = F.avg_pool2d((residual * desired).sum(1, keepdim=True), patch, patch)
    denominator = F.avg_pool2d(residual.square().sum(1, keepdim=True), patch, patch).clamp_min(1e-12)
    alpha = (numerator / denominator).clamp(0, 1)
    if smooth:
        alpha = F.interpolate(alpha, size=proposal.shape[-2:], mode="bilinear", align_corners=False)
    else:
        alpha = alpha.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
    return (proposal + alpha * residual).clamp(-1, 1)


def metric_vectors(lpips_network: torch.nn.Module, candidates: list[Tensor], target: Tensor) -> tuple[list[float], list[float]]:
    psnr: list[float] = []
    perceptual: list[float] = []
    for begin in range(0, len(candidates), 8):
        batch = torch.cat(candidates[begin:begin + 8], dim=0)
        repeated = target.expand(batch.shape[0], -1, -1, -1)
        mse = (batch.add(1).mul(.5) - repeated.add(1).mul(.5)).square().flatten(1).mean(1)
        psnr.extend(float(value) for value in (-10 * torch.log10(mse.clamp_min(1e-12))).cpu())
        values = lpips_values(lpips_network, tuple(batch[index:index + 1] for index in range(batch.shape[0])), target)[0]
        perceptual.extend(float(value) for value in values.cpu())
    return psnr, perceptual


def summarize(records: list[dict], labels: list[str]) -> dict:
    psnr = torch.tensor([[row["psnr"][label] for label in labels] for row in records], dtype=torch.float64)
    lpips = torch.tensor([[row["lpips"][label] for label in labels] for row in records], dtype=torch.float64)
    baseline_psnr = psnr[:, 0]
    baseline_lpips = lpips[:, 0]
    fixed = {
        label: {
            "psnr": float(psnr[:, index].mean()),
            "lpips": float(lpips[:, index].mean()),
            "psnr_gain_db": float((psnr[:, index] - baseline_psnr).mean()),
            "lpips_reduction": float((baseline_lpips - lpips[:, index]).mean()),
        }
        for index, label in enumerate(labels)
    }

    frontier = []
    for index in range(101):
        psnr_weight = index / 100
        utility = (
            psnr_weight * (psnr - baseline_psnr[:, None]) / PSNR_GATE
            + (1 - psnr_weight) * (baseline_lpips[:, None] - lpips) / LPIPS_GATE
        )
        selected = utility.argmax(1)
        rows = torch.arange(len(records))
        mean_psnr = float(psnr[rows, selected].mean())
        mean_lpips = float(lpips[rows, selected].mean())
        counts = Counter(labels[value] for value in selected.tolist())
        frontier.append({
            "psnr_weight": psnr_weight,
            "psnr": mean_psnr,
            "lpips": mean_lpips,
            "psnr_gain_db": mean_psnr - fixed["proposal"]["psnr"],
            "lpips_reduction": fixed["proposal"]["lpips"] - mean_lpips,
            "selected_candidate_counts": dict(sorted(counts.items())),
        })
    for row in frontier:
        row["simultaneous_gate_pass"] = (
            row["psnr_gain_db"] >= PSNR_GATE and row["lpips_reduction"] >= LPIPS_GATE
        )
        row["minimum_normalized_gate_margin"] = min(
            (row["psnr_gain_db"] - PSNR_GATE) / PSNR_GATE,
            (row["lpips_reduction"] - LPIPS_GATE) / LPIPS_GATE,
        )
    best = max(frontier, key=lambda row: row["minimum_normalized_gate_margin"])
    return {
        "fixed_candidate_means": fixed,
        "oracle_pareto_sweep": frontier,
        "best_balanced_oracle": best,
        "any_oracle_simultaneous_gate_pass": any(row["simultaneous_gate_pass"] for row in frontier),
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise RuntimeError("oracle feasibility audit requires exactly eight local GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    config = json.loads(CONFIG_PATH.read_text())
    if file_sha256(ROOT / config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise RuntimeError("V2 manifest identity changed")

    checkpoint = Path(os.environ.get("PROPOSAL_ENERGY_V2_ORACLE_CHECKPOINT", DEFAULT_CHECKPOINT))
    expected_sha = os.environ.get("PROPOSAL_ENERGY_V2_ORACLE_CHECKPOINT_SHA256", DEFAULT_CHECKPOINT_SHA256)
    if file_sha256(checkpoint) != expected_sha:
        raise RuntimeError("oracle checkpoint identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    controls = EnergyAndDirectControlsV2()
    controls.load_state_dict(payload["model"], strict=True)
    direct_model = controls.direct_model.to(device).eval().requires_grad_(False)
    proposal_model = load_frozen_proposal(config, device)
    lpips_network = build_lpips(device)

    rows = load_task_rows(config)
    validation = fixed_validation_rows(
        rows["validation"],
        int(config["data"]["validation_subset_records"]),
        int(config["data"]["validation_crop_seed"]),
    )
    labels = ["proposal"] + [f"global_alpha_{alpha:.3f}" for alpha in ALPHAS[1:]]
    labels += [f"patch_{patch}_{kind}" for patch in PATCH_SIZES for kind in ("hard", "smooth")]
    local: list[dict] = []
    for index in range(rank, len(validation), world):
        row = validation[index]
        lq, gt, _, _ = load_audit_crop(
            row,
            crop_size=int(config["data"]["validation_crop_size"]),
            seed=int(config["data"]["validation_crop_seed"]),
        )
        observation_01 = lq[None].to(device).add(1).mul(.5)
        target = gt[None].to(device)
        observation = observation_01.mul(2).sub(1)
        descriptor = descriptor_for_row(row, config, device)[None]
        proposal = run_frozen(proposal_model, observation_01)
        with torch.inference_mode():
            direct = direct_model(observation, proposal, descriptor)
            candidates = [proposal]
            candidates.extend((proposal + alpha * (direct - proposal)).clamp(-1, 1) for alpha in ALPHAS[1:])
            candidates.extend(
                patch_alpha_candidate(proposal, direct, target, patch, smooth)
                for patch in PATCH_SIZES
                for smooth in (False, True)
            )
            psnr, perceptual = metric_vectors(lpips_network, candidates, target)
        local.append({
            "index": index,
            "record_id": row["record_id"],
            "psnr": dict(zip(labels, psnr, strict=True)),
            "lpips": dict(zip(labels, perceptual, strict=True)),
        })

    gathered: list[list[dict] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        records = sorted((row for shard in gathered if shard for row in shard), key=lambda row: row["index"])
        if len(records) != 256:
            raise RuntimeError(f"oracle validation cardinality changed: {len(records)}")
        summary = summarize(records, labels)
        proposal = summary["fixed_candidate_means"]["proposal"]
        expected = payload.get("validation", {}).get("proposal")
        if expected is None or abs(proposal["psnr"] - expected["psnr"]) > 2e-4 or abs(proposal["lpips"] - expected["lpips"]) > 2e-5:
            raise RuntimeError(f"oracle protocol failed to reproduce checkpoint proposal metrics: {proposal} vs {expected}")
        result = {
            "schema_version": 1,
            "status": "complete_gt_oracle_feasibility_diagnostic",
            "warning": "GT-selecting oracle upper bound only; not a deployable inference result",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": expected_sha,
            "checkpoint_step": int(payload["step"]),
            "records": len(records),
            "candidate_labels": labels,
            "gate": {"psnr_gain_db": PSNR_GATE, "lpips_reduction": LPIPS_GATE},
            **summary,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "internal_test_accessed": False,
            "frozen_660_accessed": False,
        }
        output = Path(os.environ.get(
            "PROPOSAL_ENERGY_V2_ORACLE_OUTPUT",
            ROOT / "reports/foundir_proposal_energy_v2_direct_residual_oracle_step005118.json",
        ))
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text() != encoded:
            raise RuntimeError(f"refusing to overwrite differing oracle result: {output}")
        output.write_text(encoded)
        print("FOUNDIR_PROPOSAL_ENERGY_V2_ORACLE_COMPLETE " + json.dumps(result["best_balanced_oracle"], sort_keys=True))
        print(f"ORACLE_SIMULTANEOUS_GATE_PASS={result['any_oracle_simultaneous_gate_pass']}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
