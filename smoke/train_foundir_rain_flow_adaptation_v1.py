#!/usr/bin/env python3
"""25-step, four-node Rain native-flow qualification. No automatic formal training."""
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
from smoke import rain_flow_adaptation_contract_v1 as p
from smoke import rain_flow_adaptation_state_v1 as state_io
from smoke import replay_foundir_existing_flux_experts_v1 as replay
from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as metric_impl
from smoke import existing_flux_replay_cluster_v1 as cluster


def identity(output,row,index):
    return {"receipt_sha256":c.sha(output/"receipt.json"),"index":index,"row":row,"training_fit_only":True}


def runtime(output,receipt,node,idle):
    actual=metric_impl.hardware(idle)
    actual["metrics"]=metric_impl.metric_runtime({"metric":receipt["metric"]})
    marker=output/f"runtime_node{node}.json"
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
    if row["split"]!="train" or row["role"]!="direct_endpoint_train" or row["category"]!="10Rain":
        raise ValueError("only selected Rain training pixels may be read")
    x,y=row["crop_xy"];w,h=int(row["width"]),int(row["height"])
    if min(x,y)<0 or x+512>w or y+512>h:raise ValueError("invalid training crop")
    tensors=[]
    for side in ("lq","gt"):
        prefix=f"data/foundir/MillionIRData/Train/10Rain/{side.upper()}/"
        if not row[side+"_path"].startswith(prefix):raise ValueError("forbidden image path")
        file=c.path(ROOT,row[side+"_path"])
        if c.sha(file)!=row[side+"_sha256"]:raise ValueError("training image hash changed")
        with Image.open(file) as im:
            if im.size!=(w,h):raise ValueError("pair dimensions changed")
            array=np.array(im.convert("RGB").crop((x,y,x+512,y+512)),copy=True)
        tensors.append(torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().div(127.5).sub(1).to(device))
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
    from flux2.util import load_flow_model
    from lowlight_restoration.lora import inject_lora,load_lora_checkpoint
    from lowlight_restoration.flux2_endpoint_precision import promote_lora_to_fp32
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    model=load_flow_model("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    names=inject_lora(model,receipt["lora"]["target_patterns"],32,32.,0.)
    if set(names)!=set(expected_modules()):raise ValueError("LoRA sites changed")
    parameters=promote_lora_to_fp32(model)
    load_lora_checkpoint(model,c.path(ROOT,receipt["rain"]["checkpoint"]["path"]))
    if sum(t.numel() for t in parameters)!=32440320:raise ValueError("unexpected trainable parameter count")
    return model


def cached(output,receipt,index,device="cpu"):
    import torch
    from safetensors.torch import load_file
    marker=output/"cache"/f"image_{index:03d}.json";entry=c.read(marker)
    c.validate_record(entry,identity(output,receipt["rows"][index],index),output)
    if entry["tensor"] not in entry["files"]:raise ValueError("missing cache binding")
    data=load_file(str(c.path(output,entry["tensor"])),device=str(device))
    if set(data)!={"lq","gt","teacher_rgb","clean_latent","reference_latent"}:raise ValueError("cache keys changed")
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in data.values()):raise ValueError("cache nonfinite or nonFP32")
    for key in ("lq","gt","teacher_rgb"):
        if data[key].shape!=(1,3,512,512) or (data[key].abs()>1).any():raise ValueError("invalid RGB cache")
    if data["clean_latent"].shape!=data["reference_latent"].shape or data["clean_latent"].ndim!=4:
        raise ValueError("latent shape changed")
    return data,marker


def prepare_cache(output,receipt,rank,device):
    import torch
    from flux2.util import load_ae
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.flux2_endpoint_precision import restore_endpoint
    pending=[]
    for index in range(rank,32,8):
        if (output/"cache"/f"image_{index:03d}.json").exists():cached(output,receipt,index)
        else:pending.append(index)
    if not pending:return
    prompt=load_prompt(receipt,device);model=new_model(receipt,device).requires_grad_(False)
    ae=load_ae("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    for index in pending:
        row=receipt["rows"][index];lq,gt=load_pair(row,device)
        with torch.inference_mode():
            teacher=restore_endpoint(model,ae,lq,prompt["context"],prompt["context_ids"],batched_prc_img)
            with torch.autocast(device.type,enabled=False):clean=ae.encode(gt.float()).float()
        data={"lq":lq,"gt":gt,"teacher_rgb":teacher["decoded"],
              "clean_latent":clean,"reference_latent":teacher["degraded"]}
        if any(not torch.isfinite(v).all() for v in data.values()):raise ValueError("nonfinite teacher/cache")
        name,digest=replay.save_tensors(output,f"cache_{index:03d}",data)
        c.write_new(output/"cache"/f"image_{index:03d}.json",{"identity":identity(output,row,index),
            "tensor":name,"files":{name:digest},"teacher":"frozen_original_Rain_FLUX_step4000",
            "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3})
        print(f"RAIN_FLOW_CACHE_SAVED rank={rank} index={index}",flush=True)


def quantize(value):
    return ((value.float()+1)*127.5).round().clamp(0,255).to(__import__("torch").uint8)


def scores(metric,u8,gt):
    import torch
    with torch.inference_mode():
        rgb=u8.float()/255;target=(gt+1)*.5
        mse=float((rgb-target).square().mean());perceptual=float(metric(rgb,target));metric.reset()
    result={"psnr":100. if mse<=1e-12 else -10*math.log10(mse),"lpips":perceptual}
    if not all(math.isfinite(x) for x in result.values()):raise ValueError("nonfinite probe metrics")
    return result


def probe(output,receipt,model,ae,prompt,metric,rank,device,step):
    import torch
    from safetensors.torch import load_file
    from flux2.sampling import batched_prc_img
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    row=receipt["rows"][rank];data,cache_marker=cached(output,receipt,rank,device)
    checkpoint=output/"steps"/f"step_{step:06d}.json"
    ident={**identity(output,row,rank),"step":step,"checkpoint_receipt_sha256":c.sha(checkpoint),
           "cache_receipt_sha256":c.sha(cache_marker)}
    marker=output/"probes"/f"step_{step:06d}_image_{rank:03d}.json"
    if marker.exists():
        entry=c.read(marker);c.validate_record(entry,ident,output)
        if not entry["training_fit_only"]:raise ValueError("wrong probe scope")
        return
    model.eval()
    raw=restore_batch(model,ae,data["lq"],prompt["cfg_context"],prompt["cfg_ids"],batched_prc_img,
        seeds=[receipt["config"]["probe_seed"]+rank],steps=50,reference_time=10)
    if not torch.isfinite(raw).all():raise ValueError("nonfinite real 50-step output")
    image=quantize(raw)
    name,digest=replay.save_tensors(output,f"probe_{step:06d}_{rank:03d}",{"decoded":raw,"uint8":image})
    metrics={"teacher_one_step":scores(metric,quantize(data["teacher_rgb"]),data["gt"]),
             "student_50_step":scores(metric,image,data["gt"])}
    c.write_new(marker,{"identity":ident,"tensor":name,"files":{name:digest},"metrics":metrics,
        "training_fit_only":True,"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3})
    print(f"RAIN_FLOW_PROBE rank={rank} step={step} metrics={json.dumps(metrics,sort_keys=True)}",flush=True)


def train(output,receipt,rank,local):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from flux2.util import load_ae
    from flux2.sampling import batched_prc_img
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from lowlight_restoration.rain_flow_adaptation import enable_native_checkpointing,sample_index,training_objective
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
        print(f"RAIN_FLOW_RESUMED rank={rank} step={start} next_global_slot={start*32}",flush=True)
    del saved
    def save(step):
        rngs=[None]*8;dist.all_gather_object(rngs,state_io.rng_state(device))
        if step in (0,25):
            hashes=[None]*8;dist.all_gather_object(hashes,state_io.parameter_hash(state_io.trainable_state(model)))
            if len(set(hashes))!=1:raise ValueError("ranks have different LoRA weights")
        if rank==0:
            record=state_io.save_step(output,model,optimizer,step=step,world=8,rank_rng=rngs,
                receipt_hash=receipt_hash,history=history)
            print(f"RAIN_FLOW_CHECKPOINT step={step} sha256={record['sha256']}",flush=True)
        dist.barrier()
    if fresh:save(0)
    if start==0:
        probe(output,receipt,model,ae,prompt,metric,rank,device,0);dist.barrier()
    # Small qualification cache fits CPU memory. No new crop or teacher mismatch during reuse.
    cached_rows=[cached(output,receipt,i)[0] for i in range(32)] if start<25 else []
    for step in range(start,25):
        begun=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
        total={k:0. for k in ["total",*cfg["loss_weights"]]}
        for micro in range(4):
            index=sample_index(step,micro,rank,cfg["seed"])
            batch={k:v.to(device) for k,v in cached_rows[index].items()}
            position=step*32+micro*8+rank
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
        if rank==0:print("RAIN_FLOW_TRAIN "+json.dumps(log,sort_keys=True),flush=True)
        save(step+1)
    model.eval();gc.collect();torch.cuda.empty_cache()
    probe(output,receipt,model,ae,prompt,metric,rank,device,25);dist.barrier()
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
            name,digest=replay.save_tensors(output,"rain_flow_qualification_step25",final)
            c.write_new(marker,{"tensor":name,"files":{name:digest},"step":25,
                "receipt_hash":receipt_hash,"parameter_hash":state_io.parameter_hash(final),"may_seed_formal":False})
    dist.barrier()


def verify(output,receipt):
    import torch
    from safetensors.torch import load_file
    check_runtimes(output)
    checkpoint_hashes={}
    for step in range(26):
        marker=output/"steps"/f"step_{step:06d}.json";record=c.read(marker)
        if record["step"]!=step or record["receipt_hash"]!=c.sha(output/"receipt.json") or c.sha(c.path(output,record["file"]))!=record["sha256"]:
            raise ValueError("checkpoint history corrupted")
        checkpoint_hashes[str(step)]={"receipt_sha256":c.sha(marker),"state_sha256":record["sha256"]}
    saved,record=state_io.load_latest(output,c.sha(output/"receipt.json"),8)
    if saved["step"]!=25 or len(saved["history"])!=25:raise ValueError("training incomplete")
    if record["parameter_hash"]==c.read(output/"steps"/"step_000000.json")["parameter_hash"]:
        raise ValueError("LoRA never changed")
    for index in range(32):cached(output,receipt,index)
    metrics={};expected=[]
    for step in (0,25):
        entries=[]
        for index in range(8):
            marker=output/"probes"/f"step_{step:06d}_image_{index:03d}.json";entry=c.read(marker)
            expected.append(marker.name)
            ident={**identity(output,receipt["rows"][index],index),"step":step,
                "checkpoint_receipt_sha256":c.sha(output/"steps"/f"step_{step:06d}.json"),
                "cache_receipt_sha256":c.sha(output/"cache"/f"image_{index:03d}.json")}
            c.validate_record(entry,ident,output)
            tensor=load_file(str(c.path(output,entry["tensor"])),device="cpu")
            if (set(tensor)!={"decoded","uint8"} or tensor["decoded"].shape!=(1,3,512,512)
                or not torch.isfinite(tensor["decoded"]).all() or not torch.equal(quantize(tensor["decoded"]),tensor["uint8"])):
                raise ValueError("invalid saved probe output")
            if not entry["training_fit_only"] or set(entry["metrics"])!={"teacher_one_step","student_50_step"}:
                raise ValueError("probe scope changed")
            if any(set(v)!={"psnr","lpips"} or any(not math.isfinite(x) for x in v.values()) for v in entry["metrics"].values()):
                raise ValueError("invalid probe scores")
            entries.append(entry)
        means={a:{m:sum(e["metrics"][a][m] for e in entries)/8 for m in ("psnr","lpips")}
               for a in ("teacher_one_step","student_50_step")}
        metrics[str(step)]={"means":means,"rows":entries}
    if {f.name for f in (output/"probes").glob("*.json")}!=set(expected):raise ValueError("unexpected probe set")
    for before,after in zip(metrics["0"]["rows"],metrics["25"]["rows"]):
        if before["metrics"]["teacher_one_step"]!=after["metrics"]["teacher_one_step"]:raise ValueError("teacher denominator changed")
    exported=c.read(output/"final_lora.json")
    for name,digest in exported["files"].items():
        if c.sha(c.path(output,name))!=digest:raise ValueError("export hash mismatch")
    weights=load_file(str(c.path(output,exported["tensor"])),device="cpu")
    if state_io.parameter_hash(weights)!=saved["parameter_hash"] or exported["may_seed_formal"]:
        raise ValueError("export differs from final checkpoint")
    result={"status":"rain_flow_engineering_pass","steps":25,"sample_presentations":800,
        "receipt_sha256":c.sha(output/"receipt.json"),"checkpoint_hashes":checkpoint_hashes,"final_lora":exported,
        "metrics":metrics,"history":saved["history"],"training_fit_only":True,"quality_gate_evaluated":False,
        "source_disjoint_generalization_proven":False,"qualification_may_seed_formal":False,
        "four_original_experts_updated":False,"teacher":"frozen_Rain_FLUX_step4000",
        "protected_pixels_read":False,"lpips_independently_reexecuted":False,
        "node_runtimes":{str(n):c.read(output/f"runtime_node{n}.json") for n in range(4)}}
    c.write_new(output/"result.json",result)
    print("RAIN_FLOW_RESULT_VERIFIED steps=25 engineering_pass=true formal_gate_not_evaluated=true",flush=True)
    print(json.dumps({step:v["means"] for step,v in metrics.items()},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["preflight","hardware","prepare","runtime","run","verify"])
    parser.add_argument("--run-id",default="qualification_4x2_v1");parser.add_argument("--launch-id")
    parser.add_argument("--node-rank",type=int,choices=range(4),default=0)
    parser.add_argument("--asset-root",type=Path);parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    if args.phase=="preflight":
        receipt=p.make_receipt(ROOT,args.asset_root or ROOT)
        report={"status":"RAIN_FLOW_CPU_PREFLIGHT_PASS","selection_digest":receipt["selection_digest"],
            "sources":receipt["sources"],"training_sources":32,"probe_sources":8,"pixels_read":False,
            "gpu_used":False,"destination_assets":(args.asset_root or ROOT).resolve()==Path(receipt["config"]["active_root"])}
        if args.report:c.write_new(args.report,report)
        print(json.dumps(report,sort_keys=True),flush=True);return
    cfg=c.read(ROOT/p.CONFIG);p.validate_config(cfg)
    if str(ROOT.resolve())!=cfg["active_root"] or args.asset_root or args.report:raise ValueError("GPU only in migrated root")
    output=p.run_path(ROOT,args.run_id)
    if args.phase in ("prepare","verify") and args.node_rank!=0:raise ValueError("node0 only")
    if args.phase=="hardware":print(json.dumps(metric_impl.hardware(True)),flush=True);return
    if args.phase=="prepare":
        import shutil
        if shutil.disk_usage(ROOT).free<cfg["minimum_free_disk_gib"]*1024**3:raise ValueError("need 20GiB free, do not delete old assets")
        receipt=p.make_receipt(ROOT,ROOT);c.write_new(output/"receipt.json",receipt);return
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
