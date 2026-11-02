#!/usr/bin/env python3
"""Native50 Rain warmstart: matched 200 updates, reusable parent cache, separate CPU verification."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"third_party/flux2/src"))
from smoke import existing_flux_replay_contract_v1 as c
from smoke import rain_native50_warmstart_contract_v1 as p
from smoke import rain_flow_formal_stage1_state_v1 as state_io
from smoke import replay_foundir_existing_flux_experts_v1 as replay
from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as metric_impl
from smoke import existing_flux_replay_cluster_v1 as cluster


def identity(output,row,index):
    return {"receipt_sha256":c.sha(output/"receipt.json"),"index":index,"row":row,"usage":row["usage"]}


def runtime(output,receipt,node,idle):
    actual=metric_impl.hardware(idle)
    actual["metrics"]=metric_impl.metric_runtime({"metric":receipt["metric"]})
    marker=output/f"runtime_node{node}.json"
    parent_runtime=c.read(c.path(ROOT,p.PARENT["path"])/f"runtime_node{node}.json")
    if runtime_compatibility(actual)!=runtime_compatibility(parent_runtime):
        raise ValueError("cached teacher/latent runtime differs from current runtime")
    if idle:c.write_new(marker,actual)
    elif c.read(marker)!=actual:raise ValueError("resume runtime changed")
    return actual


def runtime_compatibility(value):
    """Compare driver release, not its GCC build description; retain raw receipts."""
    description=value.get("nvidia_driver")
    if not isinstance(description,str):
        raise ValueError("missing NVIDIA driver description")
    matches=re.findall(r"^NVRM version:\s+NVIDIA UNIX x86_64 Kernel Module\s+(\d+\.\d+(?:\.\d+)?)\s+[^\n]+$",
                       description,re.MULTILINE)
    if len(matches)!=1:
        raise ValueError("unrecognized NVIDIA driver description")
    return {**value,"nvidia_driver":matches[0]}


def check_runtimes(output):
    values=[c.read(output/f"runtime_node{n}.json") for n in range(4)]
    signatures=[runtime_compatibility(v) for v in values]
    if any(v!=signatures[0] for v in signatures[1:]):
        raise ValueError("all four training nodes must share GPU family, NVIDIA driver release and exact runtime/metric versions")
    return metric_impl.check_metric_runtimes(output)


def load_pair(row,device):
    import numpy as np
    from PIL import Image
    import torch
    p.validate_row(row)
    x,y=row["crop_xy"];w,h=int(row["width"]),int(row["height"]);tensors=[]
    for side in ("lq","gt"):
        file=c.path(ROOT,row[side+"_path"])
        if c.sha(file)!=row[side+"_sha256"]:raise ValueError("image hash changed")
        with Image.open(file) as im:
            if im.size!=(w,h):raise ValueError("pair dimensions changed")
            array=np.array(im.convert("RGB"),copy=True)
        if min(w,h)<512:
            array=np.pad(array,((0,max(0,512-h)),(0,max(0,512-w)),(0,0)),mode="reflect")
        array=np.array(array[y:y+512,x:x+512],copy=True)
        tensor=torch.from_numpy(array).permute(2,0,1).float().div(127.5).sub(1)
        code=row["augmentation_code"]
        if code%4:tensor=torch.rot90(tensor,code%4,(-2,-1))
        if code>=4:tensor=torch.flip(tensor,(-1,))
        tensors.append(tensor.contiguous().unsqueeze(0).to(device))
    return tensors


def load_prompt(receipt,device):
    import torch
    from safetensors.torch import load_file
    info=receipt["rain"]["prompt_context_cache"];file=c.path(ROOT,info["path"])
    if c.sha(file)!=info["sha256"]:raise ValueError("original Rain prompt changed")
    values=load_file(str(file),device=str(device))
    expected={"context":((1,512,7680),torch.bfloat16),"context_ids":((1,512,4),torch.int64),
              "cfg_context":((2,512,7680),torch.bfloat16),"cfg_ids":((2,512,4),torch.int64)}
    if set(values)!=set(expected):raise ValueError("prompt schema changed")
    for k,(shape,dtype) in expected.items():
        if values[k].shape!=shape or values[k].dtype!=dtype or not torch.isfinite(values[k]).all():
            raise ValueError("invalid Rain prompt tensor")
    return values


def new_model(receipt,device):
    import torch
    from flux2.util import load_flow_model
    from lowlight_restoration.lora import inject_lora,load_lora_checkpoint
    from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    model=load_flow_model("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    names=inject_lora(model,receipt["lora"]["target_patterns"],32,32.,0.)
    if set(names)!=set(expected_modules()):raise ValueError("LoRA sites changed")
    parameters=promote_lora_to_fp32(model)
    file=c.path(ROOT,receipt["native_checkpoint"]["path"])
    if c.sha(file)!=receipt["native_checkpoint"]["sha256"]:raise ValueError("native initialization hash changed")
    load_lora_checkpoint(model,file)
    if any(not torch.isfinite(t).all() for t in parameters):raise ValueError("nonfinite native LoRA")
    if sum(t.numel() for t in parameters)!=32440320:raise ValueError("unexpected trainable parameter count")
    return model


def cached(output,receipt,index,device="cpu"):
    import torch
    from safetensors.torch import load_file
    # Validate original producer identity. The large immutable parent receipt is
    # hashed once by cache_receipt(), not thousands of times per worker.
    donor=c.path(ROOT,p.PARENT["path"])
    parent=p.cache_receipt(ROOT)
    if parent["rows"][index]!=receipt["rows"][index]:
        raise ValueError("cache row mismatch")
    marker=donor/"cache"/f"image_{index:03d}.json";entry=c.read(marker)
    row=parent["rows"][index]
    expected={"receipt_sha256":p.PARENT["receipt_sha256"],"index":index,"row":row,"usage":row["usage"]}
    c.validate_record(entry,expected,donor)
    if entry["teacher"]!="frozen_original_Rain_FLUX_step4000":
        raise ValueError("cache teacher changed")
    if entry["tensor"] not in entry["files"]:raise ValueError("missing cache binding")
    data=load_file(str(c.path(donor,entry["tensor"])),device=str(device))
    if set(data)!={"lq","gt","teacher_rgb","clean_latent","reference_latent"}:raise ValueError("cache keys changed")
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in data.values()):raise ValueError("cache nonfinite or nonFP32")
    for key in ("lq","gt","teacher_rgb"):
        if data[key].shape!=(1,3,512,512) or (data[key].abs()>1).any():raise ValueError("invalid RGB cache")
    if data["clean_latent"].shape!=data["reference_latent"].shape or data["clean_latent"].ndim!=4:
        raise ValueError("latent shape changed")
    if row["usage"]!="train_update" and torch.count_nonzero(data["clean_latent"]):
        raise ValueError("evaluation GT must not be encoded as a training latent")
    return data,marker


def prepare_cache(output,receipt,rank,device):
    # No FLUX teacher inference and no new 6,532-image cache. Audit shard in CPU.
    p.validate_parent(p.cache_receipt(ROOT),receipt)
    count=0
    for index in range(rank,len(receipt["rows"]),8):
        cached(output,receipt,index)
        count+=1
    print(f"RAIN_NATIVE50_WARMSTART_CACHE_VERIFIED rank={rank} count={count} reused=true",flush=True)


def quantize(value):
    return ((value.float()+1)*127.5).round().clamp(0,255).to(__import__("torch").uint8)


def scores(metric,u8,gt):
    import torch
    with torch.inference_mode():
        # Match the frozen Rain development metric arithmetic, including the uint8 round trip.
        prediction=u8.float()/127.5-1;target=(gt+1)*.5
        mse=float(((gt.float()-prediction.float())*.5).square().mean())
        perceptual=float(metric((prediction+1)*.5,target));metric.reset()
    result={"psnr":100. if mse<=1e-12 else -10*math.log10(mse),"lpips":perceptual}
    if not all(math.isfinite(x) for x in result.values()):raise ValueError("nonfinite probe metrics")
    return result


def probe_one(output,receipt,model,ae,prompt,metric,index,device,step):
    import torch
    from safetensors.torch import load_file
    from flux2.sampling import batched_prc_img
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    row=receipt["rows"][index];data,cache_marker=cached(output,receipt,index,device)
    checkpoint=output/"steps"/f"step_{step:06d}.json"
    ident={**identity(output,row,index),"step":step,"checkpoint_receipt_sha256":c.sha(checkpoint),
           "cache_receipt_sha256":c.sha(cache_marker)}
    marker=output/"probes"/f"step_{step:06d}_image_{index:03d}.json"
    if marker.exists():
        entry=c.read(marker);c.validate_record(entry,ident,output)
        if entry["training_fit_only"]!=(row["usage"]=="train_probe"):raise ValueError("wrong probe scope")
        return
    model.eval()
    raw=restore_batch(model,ae,data["lq"],prompt["cfg_context"],prompt["cfg_ids"],batched_prc_img,
        seeds=[receipt["config"]["probe_seed"]+index-6400],steps=50,reference_time=10)
    if not torch.isfinite(raw).all():raise ValueError("nonfinite real 50-step output")
    image=quantize(raw)
    name,digest=replay.save_tensors(output,f"probe_{step:06d}_{index:03d}",{"decoded":raw,"uint8":image})
    metrics={"teacher_one_step":scores(metric,quantize(data["teacher_rgb"]),data["gt"]),
             "student_50_step":scores(metric,image,data["gt"])}
    c.write_new(marker,{"identity":ident,"tensor":name,"files":{name:digest},"metrics":metrics,
        "training_fit_only":row["usage"]=="train_probe","peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3})
    print(f"RAIN_NATIVE50_WARMSTART_PROBE index={index} step={step} metrics={json.dumps(metrics,sort_keys=True)}",flush=True)


def probe_indices(rank,count=132,world=8):
    if world!=8 or count!=132 or not 0<=rank<world:
        raise ValueError("formal probes require 32 train and 100 development images on eight ranks")
    return range(6400+rank,6400+count,world)


def training_batch(output,receipt,position,device):
    if not 0<=position<6400 or receipt["rows"][position]["usage"]!="train_update":
        raise ValueError("evaluation record cannot enter training")
    data,_=cached(output,receipt,position)
    return {k:v.to(device) for k,v in data.items()}


def probe(output,receipt,model,ae,prompt,metric,rank,device,step):
    rng=state_io.rng_state(device)
    try:
        for index in probe_indices(rank):
            probe_one(output,receipt,model,ae,prompt,metric,index,device,step)
    finally:
        state_io.restore_rng(rng,device)


def summarize_entries(entries):
    groups={"train_probe":[],"development":[]}
    for entry in entries:
        row=entry["identity"]["row"];usage=row["usage"]
        if usage not in groups:raise ValueError("training update appeared in evaluation")
        if entry["training_fit_only"]!=(usage=="train_probe"):raise ValueError("wrong evaluation scope")
        if set(entry["metrics"])!={"student_50_step","teacher_one_step"}:raise ValueError("metric models changed")
        for metrics in entry["metrics"].values():
            if set(metrics)!={"psnr","lpips"} or any(not math.isfinite(x) for x in metrics.values()):
                raise ValueError("invalid milestone scores")
        groups[usage].append(entry)
    summaries={}
    for usage,count in (("train_probe",32),("development",100)):
        rows=groups[usage]
        if len(rows)!=count or len({e["identity"]["row"]["record_id"] for e in rows})!=count:
            raise ValueError("missing/duplicate evaluation records")
        means={model:{key:sum(e["metrics"][model][key] for e in rows)/count for key in ("psnr","lpips")}
               for model in ("student_50_step","teacher_one_step")}
        summaries[usage]={"count":count,"means":means}
    m=summaries["development"]["means"]["student_50_step"]
    return {"splits":summaries,"development_gate_pass":m["psnr"]>=34.2 and m["lpips"]<=.08}


def milestone_summary(output,receipt,step):
    entries=[]
    for index in range(6400,len(receipt["rows"])):
        entry=c.read(output/"probes"/f"step_{step:06d}_image_{index:03d}.json")
        expected={**identity(output,receipt["rows"][index],index),"step":step,
                  "checkpoint_receipt_sha256":c.sha(output/"steps"/f"step_{step:06d}.json"),
                  "cache_receipt_sha256":c.sha(c.path(ROOT,p.PARENT["path"])/"cache"/f"image_{index:03d}.json")}
        if entry["identity"]!=expected:raise ValueError("milestone identity mismatch")
        entries.append(entry)
    result={"step":step,"count":132,"quality_gate_evaluated":True,"official_test_accessed":False,
            "receipt_sha256":c.sha(output/"receipt.json"),**summarize_entries(entries)}
    previous=[s for s in receipt["config"]["probe_steps"] if s<step]
    result["review_required"]=False
    if previous:
        prior=c.read(output/"milestones"/f"step_{previous[-1]:06d}.json")
        before=prior["splits"]["development"]["means"]["student_50_step"]
        after=result["splits"]["development"]["means"]["student_50_step"]
        result["development_change_from_step"]=previous[-1]
        result["development_delta"]={k:after[k]-before[k] for k in ("psnr","lpips")}
        result["review_required"]=after["psnr"]<=before["psnr"] and after["lpips"]>=before["lpips"]
    c.write_new(output/"milestones"/f"step_{step:06d}.json",result)
    print("RAIN_NATIVE50_WARMSTART_MILESTONE "+json.dumps(result,sort_keys=True),flush=True)
    return result


def train(output,receipt,rank,local):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from flux2.util import load_ae
    from flux2.sampling import batched_prc_img
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.rain_flow_adaptation import enable_native_checkpointing,training_objective
    from lowlight_restoration.flux2_endpoint_precision import assert_fp32_optimizer
    device=torch.device("cuda",local);cfg=receipt["config"];receipt_hash=c.sha(output/"receipt.json")
    random.seed(cfg["seed"]+rank);torch.manual_seed(cfg["seed"]+rank);torch.cuda.manual_seed(cfg["seed"]+rank)
    model=new_model(receipt,device);enable_native_checkpointing(model)
    ae=load_ae("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    prompt=load_prompt(receipt,device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(parameters,lr=cfg["learning_rate"],weight_decay=0.,fused=False)
    metric=LearnedPerceptualImagePatchSimilarity(net_type="alex",normalize=True).to(device).eval().requires_grad_(False)
    saved,_=state_io.load_latest(output,receipt_hash,8)
    fresh=saved is None
    progress=[None]*8;dist.all_gather_object(progress,None if fresh else saved["step"])
    if len(set(progress))!=1:raise ValueError("ranks saw different committed progress")
    history=[];start=0
    if saved is not None:
        state_io.restore(model,optimizer,saved,rank,device)
        start=saved["step"];history=saved["history"]
        if optimizer.param_groups[0]["lr"]!=cfg["learning_rate"]:raise ValueError("resumed LR changed")
        print(f"RAIN_NATIVE50_WARMSTART_RESUMED rank={rank} step={start} next_global_slot={start*32}",flush=True)
    del saved
    def save(step):
        rngs=[None]*8;dist.all_gather_object(rngs,state_io.rng_state(device))
        if step in cfg["probe_steps"]:
            hashes=[None]*8;dist.all_gather_object(hashes,state_io.parameter_hash(state_io.trainable_state(model)))
            if len(set(hashes))!=1:raise ValueError("ranks have different LoRA weights")
        if rank==0:
            record=state_io.save_step(output,model,optimizer,step=step,world=8,rank_rng=rngs,
                receipt_hash=receipt_hash,history=history)
            print(f"RAIN_NATIVE50_WARMSTART_CHECKPOINT step={step} sha256={record['sha256']}",flush=True)
        dist.barrier()
    def evaluate(step):
        model.eval();gc.collect();torch.cuda.empty_cache()
        probe(output,receipt,model,ae,prompt,metric,rank,device,step);dist.barrier()
        if rank==0:milestone_summary(output,receipt,step)
        dist.barrier()
    if fresh:save(0)
    if start in cfg["probe_steps"]:evaluate(start)
    # Do not load the whole full-data cache into every rank; read one exposure at a time.
    for step in range(start,cfg["steps"]):
        begun=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
        total={k:0. for k in ["total",*cfg["loss_weights"]]}
        for micro in range(4):
            position=step*32+micro*8+rank
            batch=training_batch(output,receipt,position,device)
            generator=torch.Generator(device=device).manual_seed(cfg["seed"]+1000003*(position+1))
            noise=torch.randn(batch["clean_latent"].shape,device=device,generator=generator,dtype=torch.float32)
            timestep=torch.rand((1,),device=device,generator=generator)*.9998+.0001
            with wrapped.no_sync() if micro<3 else nullcontext():
                loss,terms=training_objective(wrapped,ae,batch,prompt["context"],prompt["context_ids"],
                    batched_prc_img,metric,cfg["loss_weights"],noise=noise,timesteps=timestep)
                if not torch.isfinite(loss):raise ValueError("nonfinite training loss")
                (loss/4).backward()
            metric.reset()
            for k,v in {"total":loss,**terms}.items():total[k]+=float(v.detach())/4
            del batch,loss,terms,noise
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
        if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in parameters):raise ValueError("no training gradient")
        optimizer.step();assert_fp32_optimizer(optimizer);optimizer.zero_grad(set_to_none=True)
        if any(not torch.isfinite(p).all() for p in parameters):raise ValueError("nonfinite LoRA")
        vector=torch.tensor(list(total.values()),device=device,dtype=torch.float64);dist.all_reduce(vector);vector/=8
        diagnostics=torch.tensor([float(norm),time.monotonic()-begun,torch.cuda.max_memory_allocated(device)/1024**3],device=device)
        dist.all_reduce(diagnostics,op=dist.ReduceOp.MAX)
        log={"step":step+1,"losses":dict(zip(total,vector.tolist())),"max_rank_grad_norm":float(diagnostics[0]),
             "max_rank_step_seconds":float(diagnostics[1]),"peak_allocated_gib":float(diagnostics[2]),"next_global_slot":(step+1)*32}
        history.append(log)
        if rank==0:print("RAIN_NATIVE50_WARMSTART_TRAIN "+json.dumps(log,sort_keys=True),flush=True)
        save(step+1)
        if step+1 in cfg["probe_steps"]:evaluate(step+1)
    model.eval();gc.collect();torch.cuda.empty_cache()
    if rank==0:
        final=state_io.trainable_state(model)
        marker=output/"final_lora.json"
        if marker.exists():
            exported=c.read(marker)
            if exported["receipt_hash"]!=receipt_hash or exported["parameter_hash"]!=state_io.parameter_hash(final):
                raise ValueError("existing final export differs")
            for name,digest in exported["files"].items():
                if c.sha(c.path(output,name))!=digest:raise ValueError("final export corrupt")
        else:
            name,digest=replay.save_tensors(output,"rain_native50_warmstart_step200",final)
            c.write_new(marker,{"tensor":name,"files":{name:digest},"step":cfg["steps"],
                "receipt_hash":receipt_hash,"parameter_hash":state_io.parameter_hash(final),"may_seed_formal":True})
    dist.barrier()


def verify(output,receipt):
    import torch
    from safetensors.torch import load_file
    check_runtimes(output)
    cfg=receipt["config"]
    checkpoint_hashes={}
    for step in range(cfg["steps"]+1):
        marker=output/"steps"/f"step_{step:06d}.json";record=c.read(marker)
        if record["step"]!=step or record["receipt_hash"]!=c.sha(output/"receipt.json") or c.sha(c.path(output,record["file"]))!=record["sha256"]:
            raise ValueError("checkpoint history corrupted")
        checkpoint_hashes[str(step)]={"receipt_sha256":c.sha(marker),"state_sha256":record["sha256"]}
    saved,record=state_io.load_latest(output,c.sha(output/"receipt.json"),8)
    if saved["step"]!=cfg["steps"] or len(saved["history"])!=cfg["steps"]:raise ValueError("training incomplete")
    if record["parameter_hash"]==c.read(output/"steps"/"step_000000.json")["parameter_hash"]:
        raise ValueError("LoRA never changed")
    for index in range(len(receipt["rows"])):cached(output,receipt,index)
    metrics={};expected=[]
    for step in cfg["probe_steps"]:
        entries=[]
        for index in range(6400,len(receipt["rows"])):
            marker=output/"probes"/f"step_{step:06d}_image_{index:03d}.json";entry=c.read(marker)
            expected.append(marker.name)
            ident={**identity(output,receipt["rows"][index],index),"step":step,
                "checkpoint_receipt_sha256":c.sha(output/"steps"/f"step_{step:06d}.json"),
                "cache_receipt_sha256":c.sha(c.path(ROOT,p.PARENT["path"])/"cache"/f"image_{index:03d}.json")}
            c.validate_record(entry,ident,output)
            tensor=load_file(str(c.path(output,entry["tensor"])),device="cpu")
            if (set(tensor)!={"decoded","uint8"} or tensor["decoded"].shape!=(1,3,512,512)
                or not torch.isfinite(tensor["decoded"]).all() or not torch.equal(quantize(tensor["decoded"]),tensor["uint8"])):
                raise ValueError("invalid saved probe output")
            if entry["training_fit_only"]!=(receipt["rows"][index]["usage"]=="train_probe") or set(entry["metrics"])!={"teacher_one_step","student_50_step"}:
                raise ValueError("probe scope changed")
            if any(set(v)!={"psnr","lpips"} or any(not math.isfinite(x) for x in v.values()) for v in entry["metrics"].values()):
                raise ValueError("invalid probe scores")
            entries.append(entry)
        summary=summarize_entries(entries)
        metrics[str(step)]={**summary,"rows":entries}
        milestone=milestone_summary(output,receipt,step)
        if any(milestone[key]!=value for key,value in summary.items()):raise ValueError("milestone mismatch")
    if {f.name for f in (output/"probes").glob("*.json")}!=set(expected):raise ValueError("unexpected probe set")
    for step in cfg["probe_steps"][1:]:
        for before,after in zip(metrics["0"]["rows"],metrics[str(step)]["rows"]):
            if before["metrics"]["teacher_one_step"]!=after["metrics"]["teacher_one_step"]:
                raise ValueError("teacher denominator changed")
    exported=c.read(output/"final_lora.json")
    for name,digest in exported["files"].items():
        if c.sha(c.path(output,name))!=digest:raise ValueError("export hash mismatch")
    weights=load_file(str(c.path(output,exported["tensor"])),device="cpu")
    if state_io.parameter_hash(weights)!=saved["parameter_hash"] or exported["may_seed_formal"] is not True:
        raise ValueError("export differs from final checkpoint")
    result={"status":"rain_native50_warmstart_complete","steps":cfg["steps"],"sample_presentations":cfg["steps"]*32,
        "receipt_sha256":c.sha(output/"receipt.json"),"checkpoint_hashes":checkpoint_hashes,"final_lora":exported,
        "metrics":metrics,"history":saved["history"],"training_fit_only":False,"quality_gate_evaluated":True,
        "source_disjoint_generalization_proven":False,"qualification_may_seed_formal":False,
        "qualification_weights_reused":False,"formal_checkpoint_continuable_with_authorization":True,
        "development_gate_pass":metrics["200"]["development_gate_pass"],"official_foundir_test_accessed":False,
        "four_original_experts_updated":False,"native_checkpoint":receipt["native_checkpoint"],
        "cache_parent":receipt["cache_parent"],"historical_optimizer_reused":False,"teacher":"frozen_Rain_FLUX_step4000",
        "protected_pixels_read":False,"lpips_independently_reexecuted":False,
        "node_runtimes":{str(n):c.read(output/f"runtime_node{n}.json") for n in range(4)}}
    c.write_new(output/"result.json",result)
    print("RAIN_NATIVE50_WARMSTART_RESULT_VERIFIED steps=200 execution_pass=true development_gate_evaluated=true official_test_accessed=false",flush=True)
    print(json.dumps({step:v["splits"] for step,v in metrics.items()},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["preflight","hardware","prepare","runtime","run","verify"])
    parser.add_argument("--run-id",default="native50_stage1_4x2_v1");parser.add_argument("--launch-id")
    parser.add_argument("--node-rank",type=int,choices=range(4),default=0)
    parser.add_argument("--asset-root",type=Path);parser.add_argument("--report",type=Path)
    parser.add_argument("--require-cache",action="store_true")
    args=parser.parse_args()
    if args.phase=="preflight":
        receipt=p.make_receipt(ROOT,args.asset_root or ROOT,require_cache=args.require_cache)
        report={"status":"RAIN_NATIVE50_WARMSTART_CPU_PREFLIGHT_PASS","selection_digest":receipt["selection_digest"],
            "sources":receipt["sources"],"training_records":38334,"training_sources":5780,"probe_sources":132,
            "scheduled_exposures":6400,"sampled_unique_sources":receipt["sampled_unique_sources"],
            "sampled_unique_records":receipt["sampled_unique_records"],"pixels_read":False,
            "gpu_used":False,"parent_cache_metadata_verified":args.require_cache,"destination_assets":(args.asset_root or ROOT).resolve()==Path(receipt["config"]["active_root"])}
        if args.report:c.write_new(args.report,report)
        print(json.dumps(report,sort_keys=True),flush=True);return
    cfg=c.read(ROOT/p.CONFIG);p.validate_config(cfg)
    if str(ROOT.resolve())!=cfg["active_root"] or args.asset_root or args.report or args.require_cache:raise ValueError("GPU only in migrated root")
    output=p.run_path(ROOT,args.run_id)
    if args.phase in ("prepare","verify") and args.node_rank!=0:raise ValueError("node0 only")
    if args.phase=="hardware":print(json.dumps(metric_impl.hardware(True)),flush=True);return
    if args.phase=="prepare":
        import shutil
        if shutil.disk_usage(ROOT).free<cfg["minimum_free_disk_gib"]*1024**3:raise ValueError("need 100GiB free, do not delete old assets")
        receipt=p.make_receipt(ROOT,ROOT,require_cache=True);c.write_new(output/"receipt.json",receipt);return
    receipt=p.load_receipt(ROOT,output)
    if args.phase=="runtime":runtime(output,receipt,args.node_rank,True);return
    if args.phase=="run":
        import torch
        import torch.distributed as dist
        rank,local=c.worker_rank(args.node_rank,int(os.environ["LOCAL_RANK"]),int(os.environ["WORLD_SIZE"]),
                                 int(os.environ["LOCAL_WORLD_SIZE"]),int(os.environ["RANK"]))
        if int(os.environ["REPLAY_NODE_RANK"])!=args.node_rank:raise ValueError("node rank mismatch")
        runtime(output,receipt,args.node_rank,False);check_runtimes(output)
        torch.cuda.set_device(local);device=torch.device("cuda",local)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,receipt["model"]["flow_weights"])),
            AE_MODEL_PATH=str(c.path(ROOT,receipt["model"]["autoencoder_weights"])),TORCH_HOME=str(ROOT/"models/torch"),
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        dist.init_process_group("nccl",device_id=device,timeout=timedelta(minutes=20))
        try:
            prepare_cache(output,receipt,rank,device);gc.collect();torch.cuda.empty_cache();dist.barrier()
            train(output,receipt,rank,local)
        finally:dist.destroy_process_group()
    else:
        if not args.launch_id or not cluster.check(output,args.launch_id,"done"):raise ValueError("four-node done required")
        verify(output,receipt)


if __name__=="__main__":main()
