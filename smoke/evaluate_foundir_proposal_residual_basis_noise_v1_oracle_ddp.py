#!/usr/bin/env python3
"""GT-only amplitude/local feasibility audit for the frozen residual basis.

This is an upper-bound diagnostic, not a deployable selector.  It asks whether
the final fidelity residual direction already contains a candidate that can
clear the frozen +.30 dB / -.005 LPIPS development gate.
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
from lowlight_restoration.proposal_residual_basis import ProposalResidualBasisRestormer  # noqa: E402
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
)
from smoke.train_foundir_proposal_residual_basis_noise_v1_ddp import (  # noqa: E402
    CONFIG_PATH,
    build_lpips,
    lpips_values,
    run_frozen,
)

CHECKPOINT = ROOT / "checkpoints/foundir_proposal_residual_basis_noise_v1/formal1/step_008530.pt"
CHECKPOINT_SHA256 = "9ebc6b73e627e94f35ef52f511696215285c7efb9f800936d0b3d873b8aac95a"
OUTPUT = ROOT / "reports/foundir_proposal_residual_basis_noise_v1_oracle_step008530.json"
ALPHAS = tuple(index / 40 for index in range(81))  # 0.000 ... 2.000
PATCH_SIZES = (8, 16, 32, 64)
PSNR_GATE = .30
LPIPS_GATE = .005
PSNR_TOLERANCE = .001
LPIPS_TOLERANCE = .00001


def patch_alpha_candidate(proposal: Tensor, fidelity: Tensor, target: Tensor, patch: int, smooth: bool) -> Tensor:
    """Return the per-patch least-squares amplitude, bounded to [0, 2]."""
    residual = fidelity - proposal
    desired = target - proposal
    numerator = F.avg_pool2d((residual * desired).sum(1, keepdim=True), patch, patch)
    denominator = F.avg_pool2d(residual.square().sum(1, keepdim=True), patch, patch).clamp_min(1e-12)
    alpha = (numerator / denominator).clamp(0, 2)
    if smooth:
        alpha = F.interpolate(alpha, size=proposal.shape[-2:], mode="bilinear", align_corners=False)
    else:
        alpha = alpha.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
    return (proposal + alpha * residual).clamp(0, 1)


def metric_vectors(network: torch.nn.Module, candidates: list[Tensor], target: Tensor) -> tuple[list[float], list[float]]:
    psnr: list[float] = []
    perceptual: list[float] = []
    for begin in range(0, len(candidates), 8):
        batch = torch.cat(candidates[begin:begin + 8], dim=0)
        repeated = target.expand(batch.shape[0], -1, -1, -1)
        mse = (batch - repeated).square().flatten(1).mean(1)
        psnr.extend(float(value) for value in (-10 * torch.log10(mse.clamp_min(1e-12))).cpu())
        values = lpips_values(
            network,
            tuple(batch[index:index + 1].mul(2).sub(1) for index in range(batch.shape[0])),
            target.mul(2).sub(1),
        )[0]
        perceptual.extend(float(value) for value in values.cpu())
    return psnr, perceptual


def summarize(records: list[dict], labels: list[str]) -> dict:
    psnr = torch.tensor([[row["psnr"][label] for label in labels] for row in records], dtype=torch.float64)
    lpips = torch.tensor([[row["lpips"][label] for label in labels] for row in records], dtype=torch.float64)
    baseline_psnr, baseline_lpips = psnr[:, 0], lpips[:, 0]
    fixed = {}
    for index, label in enumerate(labels):
        fixed[label] = {
            "psnr": float(psnr[:, index].mean()),
            "lpips": float(lpips[:, index].mean()),
            "psnr_gain_db": float((psnr[:, index] - baseline_psnr).mean()),
            "lpips_reduction": float((baseline_lpips - lpips[:, index]).mean()),
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
        row = {
            "psnr_weight": psnr_weight,
            "psnr": mean_psnr,
            "lpips": mean_lpips,
            "psnr_gain_db": mean_psnr - fixed["proposal"]["psnr"],
            "lpips_reduction": fixed["proposal"]["lpips"] - mean_lpips,
            "selected_candidate_counts": dict(sorted(Counter(labels[value] for value in selected.tolist()).items())),
        }
        row["simultaneous_gate_pass"] = (
            row["psnr_gain_db"] >= PSNR_GATE and row["lpips_reduction"] >= LPIPS_GATE
        )
        row["minimum_normalized_gate_margin"] = min(
            (row["psnr_gain_db"] - PSNR_GATE) / PSNR_GATE,
            (row["lpips_reduction"] - LPIPS_GATE) / LPIPS_GATE,
        )
        frontier.append(row)
    best = max(frontier, key=lambda row: row["minimum_normalized_gate_margin"])
    passing_fixed = {
        label: value for label, value in fixed.items()
        if value["psnr_gain_db"] >= PSNR_GATE and value["lpips_reduction"] >= LPIPS_GATE
    }
    return {
        "fixed_candidate_means": fixed,
        "fixed_candidates_passing_joint_gate": passing_fixed,
        "oracle_pareto_sweep": frontier,
        "best_balanced_oracle": best,
        "any_fixed_candidate_simultaneous_gate_pass": bool(passing_fixed),
        "any_oracle_simultaneous_gate_pass": any(row["simultaneous_gate_pass"] for row in frontier),
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 8:
        raise RuntimeError("residual-basis oracle requires exactly eight local GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    config = json.loads(CONFIG_PATH.read_text())
    if file_sha256(ROOT / config["data"]["manifest"]) != config["data"]["manifest_sha256"]:
        raise RuntimeError("residual-basis manifest identity changed")
    checkpoint = Path(os.environ.get("PROPOSAL_RESIDUAL_BASIS_ORACLE_CHECKPOINT", CHECKPOINT))
    expected_sha = os.environ.get("PROPOSAL_RESIDUAL_BASIS_ORACLE_CHECKPOINT_SHA256", CHECKPOINT_SHA256)
    if file_sha256(checkpoint) != expected_sha:
        raise RuntimeError("residual-basis oracle checkpoint identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(payload["step"]) != 8530:
        raise RuntimeError("residual-basis oracle requires frozen step 8,530")

    model = ProposalResidualBasisRestormer(
        residual_bound=float(config["model"]["residual_bound"]),
        initial_output_scale=float(config["model"]["initial_output_scale"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    proposal_model = load_frozen_proposal(config, device)
    lpips_network = build_lpips(device)
    rows = load_task_rows(config)
    validation = fixed_validation_rows(
        rows["validation"], int(config["data"]["validation_subset_records"]),
        int(config["data"]["validation_crop_seed"]),
    )

    labels = ["proposal"] + [f"global_alpha_{alpha:.3f}" for alpha in ALPHAS[1:]]
    labels += [f"patch_{patch}_{kind}" for patch in PATCH_SIZES for kind in ("hard", "smooth")]
    local: list[dict] = []
    for index in range(rank, len(validation), world):
        row = validation[index]
        lq, gt, _, _ = load_audit_crop(
            row, crop_size=int(config["data"]["validation_crop_size"]),
            seed=int(config["data"]["validation_crop_seed"]),
        )
        observation = lq[None].to(device).add(1).mul(.5)
        target = gt[None].to(device).add(1).mul(.5)
        proposal = run_frozen(proposal_model, observation).add(1).mul(.5)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            fidelity, _ = model(observation, proposal)
        fidelity = fidelity.float()
        with torch.no_grad():
            candidates = [proposal]
            candidates.extend((proposal + alpha * (fidelity - proposal)).clamp(0, 1) for alpha in ALPHAS[1:])
            candidates.extend(
                patch_alpha_candidate(proposal, fidelity, target, patch, smooth)
                for patch in PATCH_SIZES for smooth in (False, True)
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
        if len(records) != int(config["data"]["validation_subset_records"]):
            raise RuntimeError(f"oracle validation cardinality changed: {len(records)}")
        summary = summarize(records, labels)
        expected = payload["validation"]
        rerun_proposal = summary["fixed_candidate_means"]["proposal"]
        rerun_fidelity = summary["fixed_candidate_means"]["global_alpha_1.000"]
        differences = {
            "proposal_psnr": abs(rerun_proposal["psnr"] - expected["proposal"]["psnr"]),
            "proposal_lpips": abs(rerun_proposal["lpips"] - expected["proposal"]["lpips"]),
            "fidelity_psnr": abs(rerun_fidelity["psnr"] - expected["fidelity"]["psnr"]),
            "fidelity_lpips": abs(rerun_fidelity["lpips"] - expected["fidelity"]["lpips"]),
        }
        if (
            differences["proposal_psnr"] > PSNR_TOLERANCE
            or differences["proposal_lpips"] > LPIPS_TOLERANCE
            or differences["fidelity_psnr"] > PSNR_TOLERANCE
            or differences["fidelity_lpips"] > LPIPS_TOLERANCE
        ):
            raise RuntimeError(f"oracle failed to reproduce checkpoint protocol: {differences}")
        result = {
            "schema_version": 1,
            "status": "complete_residual_basis_gt_oracle_feasibility_diagnostic",
            "warning": "GT-selecting upper bound only; not a deployable inference result",
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": expected_sha,
            "checkpoint_step": int(payload["step"]),
            "records": len(records),
            "candidate_labels": labels,
            "alpha_range": [ALPHAS[0], ALPHAS[-1]],
            "alpha_step": ALPHAS[1] - ALPHAS[0],
            "patch_sizes": list(PATCH_SIZES),
            "gate": {"psnr_gain_db": PSNR_GATE, "lpips_reduction": LPIPS_GATE},
            "protocol_reproduction": {
                "checkpoint": {"proposal": expected["proposal"], "fidelity": expected["fidelity"]},
                "oracle_rerun": {"proposal": rerun_proposal, "fidelity": rerun_fidelity},
                "absolute_difference": differences,
                "tolerance": {"psnr": PSNR_TOLERANCE, "lpips": LPIPS_TOLERANCE},
                "pass": True,
            },
            **summary,
            "official_foundir_test_accessed": False,
            "sealed_internal_accessed": False,
            "internal_test_accessed": False,
            "frozen_660_accessed": False,
        }
        output = Path(os.environ.get("PROPOSAL_RESIDUAL_BASIS_ORACLE_OUTPUT", OUTPUT))
        output.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if output.exists() and output.read_text() != encoded:
            raise RuntimeError(f"refusing to overwrite differing oracle result: {output}")
        output.write_text(encoded)
        print("FOUNDIR_PROPOSAL_RESIDUAL_BASIS_ORACLE_COMPLETE " + json.dumps(result["best_balanced_oracle"], sort_keys=True))
        print(f"FIXED_CANDIDATE_GATE_PASS={result['any_fixed_candidate_simultaneous_gate_pass']}")
        print(f"ORACLE_GATE_PASS={result['any_oracle_simultaneous_gate_pass']}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
