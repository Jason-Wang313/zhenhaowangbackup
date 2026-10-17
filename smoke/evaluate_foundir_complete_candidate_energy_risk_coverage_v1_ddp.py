#!/usr/bin/env python3
"""Cross-source calibration/evaluation audit for complete candidate energy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.complete_candidate_energy import (  # noqa: E402
    CompleteCandidateEnergy,
    compose_selected_candidates,
    select_candidates_fail_closed,
)
from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from smoke.train_foundir_complete_candidate_energy_noise_v1_ddp import (  # noqa: E402
    frozen_deployable,
    load_residual_basis,
    severity_from_rows,
)
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    atomic_json,
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
)
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import build_lpips, lpips_values  # noqa: E402

CONFIG_PATH = ROOT / "configs/foundir_complete_candidate_energy_risk_coverage_v1.json"


def split_source_groups(rows: list[dict[str, str]], seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups = sorted(
        {row["source_group"] for row in rows},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    calibration_groups = set(groups[::2])
    calibration = [row for row in rows if row["source_group"] in calibration_groups]
    evaluation = [row for row in rows if row["source_group"] not in calibration_groups]
    if not calibration or not evaluation or {r["source_group"] for r in calibration} & {r["source_group"] for r in evaluation}:
        raise RuntimeError("risk-coverage source split failed")
    return calibration, evaluation


def load_energy(audit: dict, energy: dict, device: torch.device) -> CompleteCandidateEnergy:
    path = ROOT / audit["checkpoint"]
    if file_sha256(path) != audit["checkpoint_sha256"]:
        raise RuntimeError("risk-coverage checkpoint identity changed")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["step"]) != int(audit["checkpoint_step"]):
        raise RuntimeError("risk-coverage checkpoint step changed")
    model = CompleteCandidateEnergy(
        condition_dim=int(energy["model"]["condition_dim"]),
        widths=tuple(int(value) for value in energy["model"]["widths"]),
        blocks=int(energy["model"]["blocks"]),
        initial_log_variance=float(energy["model"]["initial_log_variance"]),
    )
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def confidence_maps(risks: torch.Tensor, energy: dict) -> tuple[torch.Tensor, torch.Tensor]:
    safe = int(energy["safe_index"])
    mse, perceptual, log_variance = risks.float().unbind(2)
    uncertainty = torch.exp(.5 * log_variance.clamp(-8, 8)) * float(energy["inference"]["uncertainty_scale"])
    conservative_mse = mse + uncertainty
    conservative_perceptual = perceptual + uncertainty
    allowed = conservative_perceptual <= float(energy["inference"]["maximum_relative_perceptual_risk"])
    allowed[:, safe] = False
    alternative = conservative_mse.masked_fill(~allowed, float("inf"))
    best_cost, best = alternative.min(1)
    margin = conservative_mse[:, safe] - best_cost
    margin = torch.where(torch.isfinite(margin), margin, torch.full_like(margin, -float("inf")))
    return margin, best


def local_record(model, proposal_model, basis, row, energy, device) -> dict:
    lq, gt, _, _ = load_audit_crop(
        row, crop_size=int(energy["data"]["validation_crop_size"]),
        seed=int(energy["data"]["validation_crop_seed"]),
    )
    observation = lq[None].to(device).add(1).mul(.5)
    target = gt[None].to(device).add(1).mul(.5)
    proposal, candidates = frozen_deployable(proposal_model, basis, observation)
    severity = severity_from_rows([row], device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        risks = model(observation, proposal, candidates, severity).float()
    margin, best = confidence_maps(risks, energy)
    patch = int(energy["model"]["patch_size"])
    local_mse = F.avg_pool2d(
        (candidates - target[:, None]).square().mean(2).flatten(0, 1)[:, None], patch, patch
    ).reshape(candidates.shape[1], *margin.shape[-2:])
    return {
        "record_id": row["record_id"], "source_group": row["source_group"],
        "margin": margin[0].cpu(), "best": best[0].cpu(), "local_mse": local_mse.cpu(),
    }


def threshold_metrics(records: list[dict], threshold: float, safe: int) -> dict[str, float]:
    psnr, accepted, total = [], 0, 0
    for record in records:
        use = record["margin"] > threshold
        selected = torch.where(use, record["best"], torch.full_like(record["best"], safe))
        mse = record["local_mse"].gather(0, selected[None]).mean()
        psnr.append(float(-10 * torch.log10(mse.clamp_min(1e-12))))
        accepted += int(use.sum())
        total += use.numel()
    return {"psnr": sum(psnr) / len(psnr), "accepted_fraction": accepted / total}


def exact_evaluation(model, proposal_model, basis, rows, energy, threshold, lpips, device, rank, world):
    sums = torch.zeros(2, 3, dtype=torch.float64, device=device)
    acceptance = torch.zeros(2, dtype=torch.float64, device=device)
    for index in range(rank, len(rows), world):
        row = rows[index]
        lq, gt, _, _ = load_audit_crop(
            row, crop_size=int(energy["data"]["validation_crop_size"]),
            seed=int(energy["data"]["validation_crop_seed"]),
        )
        observation = lq[None].to(device).add(1).mul(.5)
        target = gt[None].to(device).add(1).mul(.5)
        proposal, candidates = frozen_deployable(proposal_model, basis, observation)
        severity = severity_from_rows([row], device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            risks = model(observation, proposal, candidates, severity).float()
        selected, accepted = select_candidates_fail_closed(
            risks, safe_index=int(energy["safe_index"]),
            uncertainty_scale=float(energy["inference"]["uncertainty_scale"]),
            minimum_mse_advantage=threshold,
            maximum_relative_perceptual_risk=float(energy["inference"]["maximum_relative_perceptual_risk"]),
        )
        output = compose_selected_candidates(candidates, selected, patch_size=int(energy["model"]["patch_size"]))
        arms = (candidates[:, int(energy["safe_index"])], output)
        perceptual = lpips_values(lpips, tuple(v.mul(2).sub(1) for v in arms), target.mul(2).sub(1))[0]
        for arm, value in enumerate(arms):
            mse = (value - target).square().mean()
            sums[arm, 0] += (-10 * torch.log10(mse.clamp_min(1e-12))).double()
            sums[arm, 1] += perceptual[arm].double(); sums[arm, 2] += 1
        acceptance[0] += accepted.sum(); acceptance[1] += selected.numel()
    dist.all_reduce(sums); dist.all_reduce(acceptance)
    if rank:
        return None
    names, result = ("fidelity", "calibrated"), {}
    for i, name in enumerate(names):
        result[name] = {"records": int(sums[i, 2]), "psnr": float(sums[i, 0]/sums[i, 2]), "lpips": float(sums[i, 1]/sums[i, 2])}
    result["accepted_fraction"] = float(acceptance[0]/acceptance[1])
    result["psnr_gain_over_fidelity_db"] = result["calibrated"]["psnr"] - result["fidelity"]["psnr"]
    result["lpips_change_from_fidelity"] = result["calibrated"]["lpips"] - result["fidelity"]["lpips"]
    return result


def main() -> None:
    dist.init_process_group("nccl")
    rank, world, local_rank = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank); device = torch.device("cuda", local_rank)
    audit = json.loads(CONFIG_PATH.read_text())
    energy = json.loads((ROOT / audit["energy_config"]).read_text())
    base = json.loads((ROOT / energy["base_config"]).read_text())
    energy["data"], energy["frozen_proposal"] = base["data"], base["frozen_proposal"]
    if world != 8:
        raise RuntimeError("risk-coverage audit requires eight GPUs")
    rows = load_task_rows(energy)
    fixed = fixed_validation_rows(rows["validation"], int(energy["data"]["validation_subset_records"]), int(energy["data"]["validation_crop_seed"]))
    calibration, evaluation = split_source_groups(fixed, int(audit["split_seed"]))
    model = load_energy(audit, energy, device)
    proposal_model = load_frozen_proposal(energy, device)
    basis = load_residual_basis(energy, base, device)
    local = [local_record(model, proposal_model, basis, calibration[i], energy, device) for i in range(rank, len(calibration), world)]
    gathered = [None] * world; dist.all_gather_object(gathered, local)
    chosen = float("inf"); curve = None
    if rank == 0:
        records = [r for shard in gathered for r in shard]
        positive = torch.cat([r["margin"].flatten() for r in records]); positive = positive[torch.isfinite(positive) & (positive > 0)]
        thresholds = [0.0]
        if positive.numel():
            thresholds += [float(torch.quantile(positive, float(q))) for q in audit["threshold_quantiles"]]
            thresholds.append(float(positive.max()) + 1e-6)
        thresholds = sorted(set(thresholds))
        curve = [{"threshold": t, **threshold_metrics(records, t, int(energy["safe_index"]))} for t in thresholds]
        fidelity_psnr = threshold_metrics(records, float("inf"), int(energy["safe_index"]))["psnr"]
        eligible = [x for x in curve if x["accepted_fraction"] >= float(audit["minimum_calibration_acceptance"]) and x["psnr"] > fidelity_psnr]
        if eligible:
            chosen = max(eligible, key=lambda x: x["psnr"])["threshold"]
    selected = [chosen]; dist.broadcast_object_list(selected, src=0); chosen = selected[0]
    lpips = build_lpips(device)
    result = exact_evaluation(model, proposal_model, basis, evaluation, energy, chosen, lpips, device, rank, world)
    if rank == 0:
        payload = {
            "schema_version": 1, "status": "complete_candidate_energy_risk_coverage_audit_complete",
            "warning": "threshold selected only on source-disjoint calibration groups",
            "calibration_records": len(calibration), "evaluation_records": len(evaluation),
            "calibration_source_groups": len({r['source_group'] for r in calibration}),
            "evaluation_source_groups": len({r['source_group'] for r in evaluation}),
            "selected_threshold": None if math.isinf(chosen) else chosen,
            "calibration_curve": curve, "evaluation": result,
            "scorer_has_deployable_signal": result["psnr_gain_over_fidelity_db"] > 0 and result["lpips_change_from_fidelity"] <= 0,
            **audit["access"],
        }
        atomic_json(ROOT / audit["output"], payload)
        print("FOUNDIR_COMPLETE_ENERGY_RISK_COVERAGE_COMPLETE " + json.dumps(payload, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
