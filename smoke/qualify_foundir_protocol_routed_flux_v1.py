#!/usr/bin/env python3
"""Eight training images: explicit one-hot equivalence and fixed soft-route execution."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"third_party/flux2/src"))
from smoke import existing_flux_replay_contract_v1 as c
from smoke import protocol_routed_flux_contract_v1 as p
from smoke import replay_foundir_existing_flux_experts_v1 as replay
from smoke import diagnose_foundir_existing_flux_common_protocol_v1 as previous
from smoke import existing_flux_replay_cluster_v1 as cluster


def parent_prompts(receipt):
    parent=c.run_path(ROOT,receipt["config"]["parent_run_id"])
    info=replay.prompt_info(parent,c.read(parent/"receipt.json"))
    if info["prompts"]!={t:receipt["experts"][t]["prompt"] for t in receipt["config"]["tasks"]}:
        raise ValueError("inherited prompt text changed")
    return parent,info


def identity(output,receipt,task,row,rank):
    parent=c.run_path(ROOT,receipt["config"]["parent_run_id"])
    return replay.identity_for(task,row,c.sha(output/"receipt.json"),c.sha(parent/"prompts.json"),
                               rank,c.sha(output/f"runtime_node{rank//2}.json"))


def fixture(receipt,task,arm,device="cpu"):
    import torch
    if arm not in receipt["config"]["arms"]:
        raise ValueError("unknown engineering route")
    tasks=receipt["config"]["tasks"]
    w=torch.zeros(1,len(tasks),device=device)
    if arm=="near_one_hot":
        w.fill_(receipt["config"]["near_one_hot_other_weight"])
    w[0,tasks.index(task)]=1. if arm=="one_hot" else receipt["config"]["near_one_hot_primary_weight"]
    return w


def protocols(receipt):
    from lowlight_restoration.protocol_routed_flux import ExpertProtocol,ProtocolRouter
    return ProtocolRouter([ExpertProtocol(t,receipt["experts"][t]["prompt"],
        receipt["experts"][t]["inference"].get("frozen_alpha",1.),t!="haze") for t in receipt["config"]["tasks"]])


def expected_plan(receipt,task,arm):
    weights=fixture(receipt,task,arm)
    protocol=protocols(receipt).anchor(weights)
    return {"anchor":protocol.name,"prompt":protocol.prompt,"alpha":protocol.alpha,
            "second_quantization":protocol.second_quantization,"weights":weights.tolist()}


def check_tensors(tensors,size=512):
    import torch
    if (set(tensors)!={"decoded","uint8"} or tensors["decoded"].shape!=(1,3,size,size)
            or tensors["uint8"].shape!=(1,3,size,size) or tensors["decoded"].dtype!=torch.float32
            or tensors["uint8"].dtype!=torch.uint8 or not torch.isfinite(tensors["decoded"]).all()
            or (tensors["decoded"].abs()>1).any()):
        raise ValueError("invalid saved image tensors")


def parity(original,current,size=512):
    import torch
    check_tensors(original,size);check_tensors(current,size)
    difference=float((original["decoded"]-current["decoded"]).abs().max())
    equal=torch.equal(original["uint8"],current["uint8"])
    return {"max_abs_rgb":difference,"uint8_equal":equal,"pass":difference<=1e-6 and equal}


def arm_record(output,receipt,ident,task,arm,old_marker):
    marker=output/arm/f"{ident['key']}.json"
    entry=c.read(marker)
    c.validate_record(entry,ident,output)
    if (entry["plan"]!=expected_plan(receipt,task,arm)
            or entry["original_receipt_sha256"]!=c.sha(old_marker)
            or entry["tensor"] not in entry["files"]):
        raise ValueError("saved route/protocol/denominator changed")
    return marker,entry


def generate(output,receipt,rank,local):
    import torch
    from safetensors.torch import load_file
    from flux2.util import load_flow_model,load_ae
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.protocol_routed_flux import install_protocol_bank,restore_protocol_routed
    from lowlight_restoration.existing_flux_experts import FLOW_CONTRACT
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    task,rows=c.assigned(receipt["config"]["tasks"],receipt["rows"],rank,8)
    if len(rows)!=1:
        raise ValueError("exactly one image per rank required")
    row=rows[0];ident=identity(output,receipt,task,row,rank)
    parent,old_marker,old=previous.original(receipt,task,row,rank)
    original=load_file(str(c.path(parent,old["tensor"])),device="cpu")
    check_tensors(original)
    pending=[]
    for arm in receipt["config"]["arms"]:
        if (output/arm/f"{ident['key']}.json").exists():
            _,entry=arm_record(output,receipt,ident,task,arm,old_marker)
            saved=load_file(str(c.path(output,entry["tensor"])),device="cpu")
            check_tensors(saved)
            if arm=="one_hot" and (parity(original,saved)!=entry["parity"] or not entry["parity"]["pass"]):
                raise ValueError("saved one-hot did not preserve original; do not silently rerun")
        else:
            pending.append(arm)
    if not pending:
        print(f"PROTOCOL_ROUTED_RANK_REUSED rank={rank}",flush=True);return
    device=torch.device("cuda",local)
    prompt_root,info=parent_prompts(receipt)
    # Both fixed fixtures select this task's protocol, but restore receives the explicit map.
    prompt={task:load_file(str(c.path(prompt_root,info["tensors"][task]["path"])),device=str(device))}
    ae=load_ae("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    model=load_flow_model("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    states=[(t,load_file(str(c.path(ROOT,receipt["experts"][t]["checkpoint"]["path"])),device="cpu"),32.,FLOW_CONTRACT)
            for t in receipt["config"]["tasks"]]
    names=install_protocol_bank(model,modules=expected_modules(),expert_states=states)
    del states
    model.eval().requires_grad_(False)
    rules=protocols(receipt);lq=replay.lq_tensor(row,512,device)
    for arm in pending:
        start=time.monotonic()
        raw,image,plan=restore_protocol_routed(model,names,rules,fixture(receipt,task,arm,device),
            ae,lq,prompt,batched_prc_img,restore_batch,seed=row["inference_seed"],steps=50,reference_time=10)
        tensors={"decoded":raw.cpu(),"uint8":image.cpu()}
        check_tensors(tensors)
        if plan!=expected_plan(receipt,task,arm):
            raise ValueError("sampler selected wrong prompt/output protocol")
        comparison=parity(original,tensors) if arm=="one_hot" else None
        name,digest=replay.save_tensors(output,f"{arm}_{ident['key']}",tensors)
        c.write_new(output/arm/f"{ident['key']}.json",{"identity":ident,"tensor":name,"files":{name:digest},
            "plan":plan,"parity":comparison,"original_receipt_sha256":c.sha(old_marker),
            "seconds":time.monotonic()-start,"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3})
        print(f"PROTOCOL_ROUTED_IMAGE_SAVED rank={rank} arm={arm} parity={comparison}",flush=True)
        if comparison is not None and not comparison["pass"]:
            raise ValueError("one-hot parity failed; saved failure for diagnosis")
        del raw,image,tensors


def metric_binding(old_marker,arm_markers,row):
    return {"original_receipt_sha256":c.sha(old_marker),"arm_receipt_sha256":{a:c.sha(m) for a,m in arm_markers.items()},
            "gt_sha256":row["gt_sha256"]}


def score(output,receipt,rank,local):
    import torch
    from safetensors.torch import load_file
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    task,rows=c.assigned(receipt["config"]["tasks"],receipt["rows"],rank,8)
    row=rows[0];ident=identity(output,receipt,task,row,rank)
    parent,old_marker,old=previous.original(receipt,task,row,rank)
    arms={a:arm_record(output,receipt,ident,task,a,old_marker) for a in receipt["config"]["arms"]}
    binding=metric_binding(old_marker,{a:v[0] for a,v in arms.items()},row)
    marker=output/"metrics"/f"{ident['key']}.json"
    if marker.exists():
        saved=c.read(marker)
        if saved["identity"]!=ident or saved["binding"]!=binding or not saved["training_fit_only"]:
            raise ValueError("saved metric identity changed")
        return
    device=torch.device("cuda",local)
    metric=LearnedPerceptualImagePatchSimilarity(net_type="alex",normalize=True).to(device).eval().requires_grad_(False)
    gt=previous.load_gt(receipt,row,device)  # Only after all outputs for this image were committed.
    values={}
    items={"original":(parent,old),**{a:(output,v[1]) for a,v in arms.items()}}
    with torch.inference_mode():
        for arm,(folder,entry) in items.items():
            saved=load_file(str(c.path(folder,entry["tensor"])),device=str(device));check_tensors(saved)
            rgb=saved["uint8"].float()/255.
            mse=float((rgb-gt).square().mean());lpips=float(metric(rgb,gt));metric.reset()
            values[arm]={"psnr":100. if mse<=1e-12 else -10*math.log10(mse),"lpips":lpips}
    if not all(math.isfinite(v) for arm in values.values() for v in arm.values()):
        raise ValueError("nonfinite metrics")
    c.write_new(marker,{"identity":ident,"binding":binding,"metrics":values,"training_fit_only":True})
    print(f"PROTOCOL_ROUTED_SCORED rank={rank} metrics={json.dumps(values,sort_keys=True)}",flush=True)


def verify(output,receipt):
    from safetensors.torch import load_file
    previous.check_metric_runtimes(output);parent_prompts(receipt)
    records=[];expected=[];comparisons=[]
    for rank in range(8):
        task,rows=c.assigned(receipt["config"]["tasks"],receipt["rows"],rank,8)
        if len(rows)!=1:raise ValueError("one image per rank required")
        row=rows[0];ident=identity(output,receipt,task,row,rank)
        parent,old_marker,old=previous.original(receipt,task,row,rank)
        original=load_file(str(c.path(parent,old["tensor"])),device="cpu")
        markers={}
        for arm in receipt["config"]["arms"]:
            markers[arm],entry=arm_record(output,receipt,ident,task,arm,old_marker)
            tensors=load_file(str(c.path(output,entry["tensor"])),device="cpu");check_tensors(tensors)
            if arm=="one_hot":
                actual=parity(original,tensors)
                if actual!=entry["parity"] or not actual["pass"]:
                    raise ValueError("independent saved-tensor parity failed")
                comparisons.append({"identity":ident,**actual})
        saved=c.read(output/"metrics"/f"{ident['key']}.json")
        if (saved["identity"]!=ident or saved["binding"]!=metric_binding(old_marker,markers,row)
                or not saved["training_fit_only"]):
            raise ValueError("metric identity changed")
        records.append(saved);expected.append(ident["key"])
    for folder in receipt["config"]["arms"]+["metrics"]:
        if {f.stem for f in (output/folder).glob("*.json")}!=set(expected):
            raise ValueError("missing or extra records")
    summary=p.aggregate(records,expected)
    result={"status":"protocol_routed_engineering_pass","receipt_sha256":c.sha(output/"receipt.json"),
        "records":8,"arms":receipt["config"]["arms"],"one_hot_comparisons":comparisons,"metrics":records,
        "by_task":summary,"training_fit_only":True,"routes_are_manual_engineering_fixtures":True,
        "training":False,"learned_router":False,"vlm_contribution_proven":False,"rain_adapted":False,
        "soft_route_quality_gate":False,"protected_pixels_read":False,
        "gt_pixels_read":"only_selected_eight_training_pairs_after_generation",
        "node_runtimes":{str(n):c.read(output/f"runtime_node{n}.json") for n in range(4)},
        "verification":"tensor_SHA_shape_finiteness_and_independent_one_hot_parity_plus_metric_aggregate_not_LPIPS_reexecution"}
    c.write_new(output/"result.json",result)
    print("PROTOCOL_ROUTED_RESULT_VERIFIED records=8 one_hot_parity=true soft_execution=true scientific_pass_not_claimed=true",flush=True)
    print(json.dumps(summary,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["preflight","hardware","prepare","runtime","run","verify"])
    parser.add_argument("--run-id",default="protocol_routed_4x2_v1")
    parser.add_argument("--launch-id")
    parser.add_argument("--node-rank",type=int,choices=range(4),default=0)
    parser.add_argument("--asset-root",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--require-parent",action="store_true")
    args=parser.parse_args()
    if args.phase=="preflight":
        receipt=p.make_receipt(ROOT,args.asset_root or ROOT,require_parent=args.require_parent)
        result={"status":"PROTOCOL_ROUTED_CPU_PREFLIGHT_PASS","topology":c.TOPOLOGY,
            "selection_digest":receipt["selection_digest"],"parent_selection_digest":receipt["parent_selection_digest"],
            "source_bindings":receipt["sources"],"records":8,"gt_metadata_records":len(receipt["gt"]),
            "pixels_decoded":False,"parent_runtime_artifacts_verified":args.require_parent,"gpu_used":False}
        if args.report:c.write_new(args.report,result)
        print(json.dumps(result,sort_keys=True),flush=True);return
    cfg=c.read(ROOT/p.CONFIG);p.validate_config(cfg)
    if str(ROOT.resolve())!=cfg["active_root"] or args.asset_root or args.report:
        raise ValueError("GPU phases only in migrated root, without asset/report overrides")
    output=p.run_path(ROOT,args.run_id)
    if args.phase in ("prepare","verify") and args.node_rank!=0:raise ValueError("node0 only")
    if args.phase=="hardware":
        print(json.dumps(previous.hardware(True)),flush=True);return
    if args.phase=="prepare":
        import shutil
        if shutil.disk_usage(ROOT).free<4*1024**3:raise ValueError("need 4GiB free; preserve old outputs")
        receipt=p.make_receipt(ROOT,ROOT,require_parent=True);parent_prompts(receipt)
        c.write_new(output/"receipt.json",receipt);return
    receipt=p.load_receipt(ROOT,output)
    if args.phase=="runtime":
        previous.runtime(output,receipt,args.node_rank,idle=True);return
    if args.phase=="run":
        import torch
        previous.runtime(output,receipt,args.node_rank,idle=False);previous.check_metric_runtimes(output)
        if int(os.environ["REPLAY_NODE_RANK"])!=args.node_rank:raise ValueError("node rank mismatch")
        rank,local=c.worker_rank(args.node_rank,int(os.environ["LOCAL_RANK"]),int(os.environ["WORLD_SIZE"]),
                                 int(os.environ["LOCAL_WORLD_SIZE"]),int(os.environ["RANK"]))
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,"models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors")),
            AE_MODEL_PATH=str(c.path(ROOT,"models/flux2-klein-base-4b/ae.safetensors")),
            HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TORCH_HOME=str(ROOT/"models/torch"))
        generate(output,receipt,rank,local)
        gc.collect();torch.cuda.empty_cache()  # Release FLUX before loading metric-only AlexNet.
        score(output,receipt,rank,local)
    else:
        if not args.launch_id or not cluster.check(output,args.launch_id,"done"):
            raise ValueError("four-node completion required")
        verify(output,receipt)


if __name__=="__main__":main()
