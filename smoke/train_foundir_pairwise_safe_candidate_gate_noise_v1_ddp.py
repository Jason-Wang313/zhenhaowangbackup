#!/usr/bin/env python3
"""Train explicit pairwise safe-versus-candidate decisions for Noise restoration."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.complete_candidate_energy import tie_aware_gap_ranking  # noqa: E402
from lowlight_restoration.flux_prompt_cache import file_sha256  # noqa: E402
from lowlight_restoration.pairwise_safe_candidate_gate import (  # noqa: E402
    PairwiseSafeCandidateGate,
    balanced_binary_loss,
    compose_pairwise_selection,
    pairwise_targets,
    parameter_count,
    select_pairwise_candidates,
)
from smoke.evaluate_foundir_complete_candidate_energy_risk_coverage_v1_ddp import split_source_groups  # noqa: E402
from smoke.train_foundir_complete_candidate_energy_noise_v1_ddp import (  # noqa: E402
    frozen_deployable,
    load_residual_basis,
    severity_from_rows,
    spatial_lpips_maps,
)
from smoke.train_foundir_proposal_relative_energy_noise_v1_ddp import (  # noqa: E402
    atomic_json,
    fixed_validation_rows,
    load_audit_crop,
    load_frozen_proposal,
    load_task_rows,
    load_training_patch,
    sample_global_batch,
)
from smoke.train_foundir_proposal_relative_energy_noise_v2_ddp import build_lpips, lpips_values  # noqa: E402

CONFIG_PATH = ROOT / "configs/foundir_pairwise_safe_candidate_gate_noise_v1.json"


def learning_rate(config: dict, step: int) -> float:
    schedule = config["schedule"]
    peak, minimum = float(schedule["learning_rate"]), float(schedule["minimum_learning_rate"])
    warmup, horizon = int(schedule["warmup_steps"]), int(schedule["primary_steps"])
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, horizon - warmup)
    return minimum + .5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def crop_for_step(config: dict, step: int) -> int:
    selected = 0
    for start, size in config["schedule"]["progressive_crops"]:
        if step > int(start):
            selected = int(size)
    if not selected or selected % int(config["model"]["patch_size"]):
        raise RuntimeError("pairwise gate crop geometry changed")
    return selected


def alternative_candidates(deployable: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if deployable.shape[1] != 7:
        raise RuntimeError("deployable candidate count changed")
    return deployable[:, 1], torch.cat((deployable[:, :1], deployable[:, 2:]), dim=1)


def objective(outputs, fidelity, alternatives, target, spatial_lpips, config):
    candidates = torch.cat((fidelity[:, None], alternatives), dim=1)
    with torch.no_grad():
        maps = spatial_lpips_maps(spatial_lpips, candidates, target)
        delta_mse, delta_perceptual = pairwise_targets(
            fidelity, alternatives, target, maps[:, 0], maps[:, 1:],
            patch_size=int(config["model"]["patch_size"]),
        )
    gain_logit, harm_logit, predicted_delta = outputs.float().unbind(2)
    weights = config["loss"]
    gain_boundary = delta_mse + float(weights["gain_margin"])
    gain_mask = gain_boundary.abs() > float(weights["gain_tie"])
    gain_label = gain_boundary < 0
    harm_boundary = delta_perceptual - float(weights["harm_margin"])
    harm_mask = harm_boundary.abs() > float(weights["harm_tie"])
    harm_label = harm_boundary > 0
    gain_loss = balanced_binary_loss(gain_logit, gain_label, gain_mask)
    harm_loss = balanced_binary_loss(harm_logit, harm_label, harm_mask)
    regression = (F.smooth_l1_loss(predicted_delta, delta_mse, reduction="none") * (delta_mse.abs() + .1)).mean()
    ranking = tie_aware_gap_ranking(predicted_delta, delta_mse, tie_epsilon=float(weights["gain_tie"]))
    loss = (
        float(weights["gain_classification_weight"]) * gain_loss
        + float(weights["harm_classification_weight"]) * harm_loss
        + float(weights["delta_regression_weight"]) * regression
        + float(weights["ranking_weight"]) * ranking
    )
    return loss, {
        "gain_bce": gain_loss, "harm_bce": harm_loss, "delta_regression": regression,
        "ranking": ranking, "gain_positive_fraction": gain_label.float().mean(),
        "harm_positive_fraction": harm_label.float().mean(),
    }


def calibration_record(model, proposal_model, basis, row, config, spatial_lpips, device):
    lq, gt, _, _ = load_audit_crop(row, crop_size=int(config["data"]["validation_crop_size"]), seed=int(config["data"]["validation_crop_seed"]))
    observation = lq[None].to(device).add(1).mul(.5); target = gt[None].to(device).add(1).mul(.5)
    proposal, deployable = frozen_deployable(proposal_model, basis, observation)
    fidelity, alternatives = alternative_candidates(deployable)
    severity = severity_from_rows([row], device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(observation, proposal, fidelity, alternatives, severity).float()
    gain, harm, delta = outputs[0].unbind(1)
    patch = int(config["model"]["patch_size"])
    candidates = torch.cat((fidelity[:, None], alternatives), dim=1)
    local_mse = F.avg_pool2d((candidates-target[:, None]).square().mean(2).flatten(0,1)[:,None], patch, patch).reshape(7,*gain.shape[-2:])
    maps = spatial_lpips_maps(spatial_lpips, candidates, target)[0]
    local_perceptual = F.avg_pool2d(maps[:,None], patch, patch)[:,0]
    return {
        "gain": torch.sigmoid(gain).cpu(), "harm": torch.sigmoid(harm).cpu(),
        "delta": delta.cpu(), "local_mse": local_mse.cpu(), "local_perceptual": local_perceptual.cpu(),
    }


def policy_metrics(records, gain_threshold, harm_threshold):
    psnr, perception, accepted, total = [], [], 0, 0
    for record in records:
        eligible = (record["gain"] >= gain_threshold) & (record["harm"] <= harm_threshold) & (record["delta"] < 0)
        cost = record["delta"].masked_fill(~eligible, float("inf")); best_cost, best = cost.min(0)
        use = torch.isfinite(best_cost); index = torch.where(use, best + 1, torch.zeros_like(best))
        mse = record["local_mse"].gather(0, index[None]).mean()
        perceptual = record["local_perceptual"].gather(0, index[None]).mean()
        psnr.append(float(-10*torch.log10(mse.clamp_min(1e-12)))); perception.append(float(perceptual))
        accepted += int(use.sum()); total += use.numel()
    return {"psnr": sum(psnr)/len(psnr), "perceptual_proxy": sum(perception)/len(perception), "accepted_fraction": accepted/total}


def select_calibration_policy(records, config):
    baseline = policy_metrics(records, 1.1, -1)
    eligible = []
    for gain in config["calibration"]["gain_probabilities"]:
        for harm in config["calibration"]["maximum_harm_probabilities"]:
            result = policy_metrics(records, float(gain), float(harm))
            result.update({"gain_probability": float(gain), "maximum_harm_probability": float(harm)})
            if (
                result["accepted_fraction"] >= float(config["calibration"]["minimum_acceptance"])
                and result["psnr"] > baseline["psnr"]
                and result["perceptual_proxy"] <= baseline["perceptual_proxy"] * float(config["calibration"]["maximum_perceptual_proxy_ratio"])
            ):
                eligible.append(result)
    return (max(eligible, key=lambda value: value["psnr"]) if eligible else None), baseline


def exact_evaluation(model, proposal_model, basis, rows, config, policy, lpips, device, rank, world):
    sums = torch.zeros(2,3,dtype=torch.float64,device=device); coverage=torch.zeros(2,dtype=torch.float64,device=device)
    for index in range(rank,len(rows),world):
        row=rows[index]; lq,gt,_,_=load_audit_crop(row,crop_size=int(config["data"]["validation_crop_size"]),seed=int(config["data"]["validation_crop_seed"]))
        observation=lq[None].to(device).add(1).mul(.5); target=gt[None].to(device).add(1).mul(.5)
        proposal,deployable=frozen_deployable(proposal_model,basis,observation); fidelity,alternatives=alternative_candidates(deployable)
        severity=severity_from_rows([row],device)
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16): outputs=model(observation,proposal,fidelity,alternatives,severity).float()
        if policy is None:
            selected=torch.zeros_like(outputs[:,0,0],dtype=torch.long); accepted=torch.zeros_like(selected,dtype=torch.bool)
        else:
            selected,accepted=select_pairwise_candidates(outputs,gain_probability=policy["gain_probability"],maximum_harm_probability=policy["maximum_harm_probability"])
        restored=compose_pairwise_selection(fidelity,alternatives,selected,accepted,patch_size=int(config["model"]["patch_size"]))
        arms=(fidelity,restored); perceptual=lpips_values(lpips,tuple(v.mul(2).sub(1) for v in arms),target.mul(2).sub(1))[0]
        for arm,value in enumerate(arms):
            mse=(value-target).square().mean(); sums[arm,0]+=(-10*torch.log10(mse.clamp_min(1e-12))).double(); sums[arm,1]+=perceptual[arm].double(); sums[arm,2]+=1
        coverage[0]+=accepted.sum(); coverage[1]+=accepted.numel()
    dist.all_reduce(sums); dist.all_reduce(coverage)
    if rank:return None
    result={}
    for i,name in enumerate(("fidelity","pairwise")): result[name]={"records":int(sums[i,2]),"psnr":float(sums[i,0]/sums[i,2]),"lpips":float(sums[i,1]/sums[i,2])}
    result["accepted_fraction"]=float(coverage[0]/coverage[1]); result["psnr_gain_over_fidelity_db"]=result["pairwise"]["psnr"]-result["fidelity"]["psnr"]; result["lpips_change_from_fidelity"]=result["pairwise"]["lpips"]-result["fidelity"]["lpips"]
    result["joint_gate_pass"]=result["pairwise"]["psnr"]>=float(config["gate"]["minimum_psnr"]) and result["pairwise"]["lpips"]<=float(config["gate"]["maximum_lpips"])
    return result


def validate(model,proposal_model,basis,calibration,evaluation,config,spatial_lpips,global_lpips,device,rank,world):
    model.eval(); local=[calibration_record(model,proposal_model,basis,calibration[i],config,spatial_lpips,device) for i in range(rank,len(calibration),world)]
    gathered=[None]*world; dist.all_gather_object(gathered,local); policy=None; baseline=None
    if rank==0: policy,baseline=select_calibration_policy([r for shard in gathered for r in shard],config)
    holder=[policy]; dist.broadcast_object_list(holder,src=0); policy=holder[0]
    result=exact_evaluation(model,proposal_model,basis,evaluation,config,policy,global_lpips,device,rank,world); model.train()
    if rank==0:return {"calibration_policy":policy,"calibration_fidelity":baseline,"evaluation":result}
    return None


def save_checkpoint(path,model,optimizer,**metadata):
    payload={"schema_version":1,"status":"foundir_pairwise_safe_candidate_gate_noise_v1_checkpoint","model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":optimizer.state_dict(),**metadata,"internal_test_accessed":False,"official_foundir_test_accessed":False,"sealed_internal_accessed":False,"frozen_660_accessed":False}
    temporary=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temporary); os.replace(temporary,path)


def main():
    dist.init_process_group("nccl"); rank,world,local_rank=dist.get_rank(),dist.get_world_size(),int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank); device=torch.device("cuda",local_rank)
    config=json.loads(CONFIG_PATH.read_text()); parent=json.loads((ROOT/config["base_config"]).read_text()); base=json.loads((ROOT/parent["base_config"]).read_text()); config["data"],config["frozen_proposal"]=base["data"],base["frozen_proposal"]
    if world!=8:raise RuntimeError("pairwise gate requires eight GPUs")
    names=[None]*world; dist.all_gather_object(names,torch.cuda.get_device_name(local_rank)); families=[f for f in config["hardware"]["supported_families"] if all(f in str(n).upper() for n in names)]
    if len(families)!=1:raise RuntimeError(f"requires homogeneous H100/H200, got {names}")
    profile=config["hardware"]["profiles"][families[0]]; micro,accumulation=int(profile["micro_batch_per_gpu"]),int(profile["gradient_accumulation"])
    if micro*accumulation*world!=int(config["hardware"]["global_batch"]):raise RuntimeError("global batch changed")
    if file_sha256(ROOT/config["data"]["manifest"])!=config["data"]["manifest_sha256"]:raise RuntimeError("manifest changed")
    rows=load_task_rows(config); fixed=fixed_validation_rows(rows["validation"],int(config["data"]["validation_subset_records"]),int(config["data"]["validation_crop_seed"])); calibration,evaluation=split_source_groups(fixed,int(config["calibration"]["split_seed"]))
    grouped={}
    for row in rows["train"]:grouped.setdefault(row["source_group"],[]).append(row)
    proposal_model=load_frozen_proposal(config,device); basis=load_residual_basis(parent,base,device)
    model=PairwiseSafeCandidateGate(alternative_count=len(config["candidate_names"]),condition_dim=int(config["model"]["condition_dim"]),widths=tuple(config["model"]["widths"]),blocks=int(config["model"]["blocks"])).to(device); distributed=DDP(model,device_ids=[local_rank],gradient_as_bucket_view=False)
    spatial_lpips=build_lpips(device); spatial_lpips.spatial=True; global_lpips=build_lpips(device)
    optimizer=torch.optim.AdamW(distributed.parameters(),lr=float(config["schedule"]["learning_rate"]),betas=(.9,.999),weight_decay=float(config["schedule"]["weight_decay"]))
    run_id=os.environ.get("PAIRWISE_SAFE_GATE_RUN_ID","qualification_v1"); total=int(os.environ.get("PAIRWISE_SAFE_GATE_TOTAL_STEPS","25")); allowed={int(v) for v in config["schedule"]["milestones"]}
    if total not in allowed:raise RuntimeError("total steps must be frozen milestone")
    output=ROOT/config["runtime"]["output_root"]/run_id; output.mkdir(parents=True,exist_ok=True); config_sha=file_sha256(CONFIG_PATH); hardware={"gpu_family":families[0],"gpu_names":names,"world_size":world,"micro_batch_per_gpu":micro,"gradient_accumulation":accumulation,"global_batch":micro*accumulation*world}
    if rank==0:atomic_json(output/"launch.json",{"status":"foundir_pairwise_safe_candidate_gate_started","run_id":run_id,"total_steps":total,"config_sha256":config_sha,"parameters":parameter_count(model),"hardware":hardware,**config["access"]})
    start_step=0; checkpoints=sorted(output.glob("step_*.pt"))
    if checkpoints:
        payload=torch.load(checkpoints[-1],map_location="cpu",weights_only=False)
        if payload.get("config_sha256")!=config_sha:raise RuntimeError("resume config changed")
        model.load_state_dict(payload["model"],strict=True); optimizer.load_state_dict(payload["optimizer"]); start_step=int(payload["step"])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
    terms=("gain_bce","harm_bce","delta_regression","ranking","gain_positive_fraction","harm_positive_fraction"); verified=set(); started=time.monotonic(); distributed.train(); global_batch=int(config["hardware"]["global_batch"])
    for step in range(start_step+1,total+1):
        lr=learning_rate(config,step)
        for group in optimizer.param_groups:group["lr"]=lr
        selected=sample_global_batch(grouped,task="pairwise_safe_candidate_gate_noise_v1",step=step,seed=int(config["schedule"]["seed"]),count=global_batch); local=selected[rank*(global_batch//world):(rank+1)*(global_batch//world)]; optimizer.zero_grad(set_to_none=True); reduced=torch.zeros(1+len(terms),device=device)
        for accumulation_index in range(accumulation):
            chunk=local[accumulation_index*micro:(accumulation_index+1)*micro]; batch_rows=[r for r,_ in chunk]; pairs=[load_training_patch(r,crop_size=crop_for_step(config,step),augmentation_seed=s,verified=verified) for r,s in chunk]
            observation=torch.stack([p[0] for p in pairs]).to(device); target=torch.stack([p[1] for p in pairs]).to(device); proposal,deployable=frozen_deployable(proposal_model,basis,observation); fidelity,alternatives=alternative_candidates(deployable); severity=severity_from_rows(batch_rows,device)
            sync=nullcontext() if accumulation_index+1==accumulation else distributed.no_sync()
            with sync:
                with torch.autocast("cuda",dtype=torch.bfloat16):outputs=distributed(observation,proposal,fidelity,alternatives,severity)
                loss,values=objective(outputs,fidelity,alternatives,target,spatial_lpips,config); (loss/accumulation).backward()
            reduced[0]+=loss.detach()/accumulation
            for i,name in enumerate(terms,1):reduced[i]+=values[name].detach()/accumulation
        norm=torch.nn.utils.clip_grad_norm_(distributed.parameters(),float(config["schedule"]["gradient_clip_norm"]));
        if not bool(torch.isfinite(norm)) or not bool(torch.isfinite(reduced).all()):raise RuntimeError("non-finite pairwise gate state")
        optimizer.step()
        if step%5==0:
            dist.all_reduce(reduced)
            if rank==0:
                numbers=(reduced/world).tolist(); record={"step":step,"lr":lr,"loss":numbers[0],"elapsed_seconds":time.monotonic()-started}; record.update({name:numbers[i] for i,name in enumerate(terms,1)}); print(json.dumps(record),flush=True)
        if step==total or step%int(config["schedule"]["steps_per_effective_pass"])==0:
            dist.barrier(); validation=validate(model,proposal_model,basis,calibration,evaluation,config,spatial_lpips,global_lpips,device,rank,world); dist.barrier()
            if rank==0:
                elapsed=time.monotonic()-started; save_checkpoint(output/f"step_{step:06d}.pt",model,optimizer,step=step,config_sha256=config_sha,validation=validation,hardware=hardware,elapsed_seconds=elapsed); atomic_json(output/f"validation_step_{step:06d}.json",{"status":"foundir_pairwise_safe_candidate_gate_validation","step":step,"validation":validation,"seconds_per_optimizer_step":elapsed/max(1,step-start_step),"projected_primary_wall_hours":elapsed/max(1,step-start_step)*int(config["schedule"]["primary_steps"])/3600,"projected_primary_gpu_hours":elapsed/max(1,step-start_step)*int(config["schedule"]["primary_steps"])/3600*world,"hardware":hardware,**config["access"]})
    if rank==0:
        elapsed=time.monotonic()-started; atomic_json(output/"terminal_result.json",{"status":"foundir_pairwise_safe_candidate_gate_training_complete","completed_optimizer_steps":total,"wall_clock_hours":elapsed/3600,"allocated_gpu_hours":elapsed/3600*world,"hardware":hardware,**config["access"]})
    dist.destroy_process_group()


if __name__=="__main__":main()
