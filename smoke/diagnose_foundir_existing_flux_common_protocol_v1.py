#!/usr/bin/env python3
"""One additional common-prompt rollout; fixed four-arm scoring on parent training sources."""
import argparse
import gc
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/flux2/src"))
from smoke import existing_flux_replay_contract_v1 as c
from smoke import existing_flux_common_protocol_contract_v1 as p
from smoke import replay_foundir_existing_flux_experts_v1 as replay
from smoke import existing_flux_replay_cluster_v1 as cluster


def prompt_view(receipt):
    return {"config":{"tasks":["common"]}, "experts":{"common":{"prompt":receipt["config"]["common_prompt"]}}}


def hardware(require_idle):
    import torch
    runtime = replay.hardware(8, require_idle=require_idle)
    if "H200" not in runtime["gpu_name"] or any(torch.cuda.get_device_properties(i).total_memory < 139*1024**3 for i in range(2)):
        raise ValueError("this stage is qualified for two >=139GiB H200 per node, not H100")
    return runtime


def metric_runtime(cfg):
    spec = importlib.util.find_spec("torchmetrics")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("torchmetrics unavailable")
    package = Path(next(iter(spec.submodule_search_locations)))
    head = package / "functional/image/lpips_models/alex.pth"
    if c.sha(head) != cfg["metric"]["lpips_head_sha256"]:
        raise ValueError("LPIPS v0.1 head changed")
    return {"packages":{name:importlib.metadata.version(name) for name in ("torchmetrics","torchvision","numpy","Pillow")},
            "lpips_head_sha256":c.sha(head),
            "lpips_implementation_sha256":c.sha(package / "functional/image/lpips.py")}


def runtime(output, receipt, node, *, idle):
    actual = hardware(idle)
    if actual != receipt["parent_runtimes"][str(node)]["runtime"]:
        raise ValueError("same-node parent runtime required to reuse its original outputs")
    actual = {**actual, "metrics":metric_runtime(receipt["config"])}
    marker = output / f"runtime_node{node}.json"
    if idle:
        c.write_new(marker, actual)
    elif actual != c.read(marker):
        raise ValueError("protocol runtime changed")
    return actual


def check_metric_runtimes(output):
    signatures=[c.read(output/f"runtime_node{n}.json")["metrics"] for n in range(4)]
    if any(value!=signatures[0] for value in signatures[1:]):
        raise ValueError("all four nodes must use the same LPIPS implementation and metric dependencies")
    return signatures[0]


def identity(output, task, row, rank):
    return replay.identity_for(task, row, c.sha(output / "receipt.json"), c.sha(output / "prompts.json"),
                               rank, c.sha(output / f"runtime_node{rank//2}.json"))


def original(receipt, task, row, rank):
    parent = c.run_path(ROOT, receipt["config"]["parent_run_id"])
    ident = replay.identity_for(task, row, c.sha(parent / "receipt.json"), c.sha(parent / "prompts.json"),
                                rank, c.sha(parent / f"runtime_node{rank//2}.json"))
    marker = parent / "original" / f"{ident['key']}.json"
    entry = c.read(marker)
    c.validate_record(entry, ident, parent)
    bank = c.read(parent / "bank" / f"{ident['key']}.json")
    c.validate_record(bank, ident, parent)
    if bank["original_receipt_sha256"] != c.sha(marker) or not bank["pass"] or bank["max_abs_rgb"] != 0 or not bank["uint8_equal"]:
        raise ValueError("parent per-image parity changed")
    return parent, marker, entry


def load_gt(receipt, row, device):
    import torch
    import numpy as np
    from PIL import Image
    entry = receipt["gt"][row["record_id"]]
    prefix = f"data/foundir/MillionIRData/Train/{row['category']}/GT/"
    if (row["split"] != "train" or not entry["path"].startswith(prefix)
            or entry["sha256"] != row["gt_sha256"]):
        raise ValueError("GT outside fixed training permission")
    file = c.path(ROOT, entry["path"])
    if c.sha(file) != entry["sha256"]:
        raise ValueError("training GT changed")
    x,y = row["crop_xy"]
    with Image.open(file) as im:
        if im.size != (int(row["width"]), int(row["height"])):
            raise ValueError("unaligned GT dimensions")
        array = np.asarray(im.convert("RGB").crop((x,y,x+512,y+512))).copy()
    return torch.from_numpy(array).permute(2,0,1).unsqueeze(0).float().div(255).to(device)


def output_arms(task, lq, original_tensors, common_tensors, alpha):
    import torch
    a = replay.quantize(task, lq, original_tensors["decoded"], alpha)
    b = replay.quantize(task, lq, common_tensors["decoded"], alpha)
    if not torch.equal(a, original_tensors["uint8"]) or not torch.equal(b, common_tensors["uint8"]):
        raise ValueError("original output quantization no longer reproduces saved pixels")
    return dict(zip(p.ARMS, (a, b,
        replay.quantize("haze", lq, original_tensors["decoded"], 1.),
        replay.quantize("haze", lq, common_tensors["decoded"], 1.))))


def generate(output, receipt, rank, local):
    import torch
    from safetensors.torch import load_file
    from flux2.util import load_flow_model, load_ae
    from flux2.sampling import batched_prc_img
    from lowlight_restoration.existing_flux_experts import install_replay_bank, FLOW_CONTRACT
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    from smoke.evaluate_foundir_blur_only_flux2_full_development_ddp import restore_batch
    task, rows = c.assigned(receipt["config"]["tasks"], receipt["rows"], rank, 8)
    pending = []
    for row in rows:
        ident = identity(output, task, row, rank)
        marker = output / "common" / f"{ident['key']}.json"
        if marker.exists():
            c.validate_record(c.read(marker), ident, output)
        else:
            pending.append(row)
    if not pending:
        return
    device = torch.device("cuda",local)
    info = replay.prompt_info(output, prompt_view(receipt))["tensors"]["common"]
    prompt = load_file(str(c.path(output,info["path"])),device=str(device))
    ae = load_ae("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    model = load_flow_model("flux.2-klein-base-4b",device=device).eval().requires_grad_(False)
    states = [(t,load_file(str(c.path(ROOT,receipt["experts"][t]["checkpoint"]["path"])),device="cpu"),32.,FLOW_CONTRACT)
              for t in receipt["config"]["tasks"]]
    install_replay_bank(model,modules=expected_modules(),expert_states=states,expert_index=receipt["config"]["tasks"].index(task))
    del states
    model.eval().requires_grad_(False)
    alpha = receipt["experts"][task]["inference"].get("frozen_alpha",1.)
    for row in pending:
        start = time.monotonic()
        ident = identity(output,task,row,rank)
        original(receipt,task,row,rank)  # Verify saved denominator before spending GPU time.
        lq = replay.lq_tensor(row,512,device)
        raw = restore_batch(model,ae,lq,prompt["cfg_context"],prompt["cfg_ids"],batched_prc_img,
                            seeds=[row["inference_seed"]],steps=50,reference_time=10)
        if not torch.isfinite(raw).all():
            raise ValueError("nonfinite common-prompt output")
        image = replay.quantize(task,lq,raw,alpha)
        name,digest = replay.save_tensors(output,f"common_{ident['key']}",{"decoded":raw,"uint8":image})
        c.write_new(output / "common" / f"{ident['key']}.json",{"identity":ident,"tensor":name,"files":{name:digest},
            "seconds":time.monotonic()-start,"peak_allocated_gib":torch.cuda.max_memory_allocated(device)/1024**3})
        print(f"COMMON_PROTOCOL_IMAGE_SAVED rank={rank} key={ident['key']}",flush=True)


def score(output, receipt, rank, local):
    import torch
    from safetensors.torch import load_file
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    device = torch.device("cuda",local)
    task,rows = c.assigned(receipt["config"]["tasks"],receipt["rows"],rank,8)
    metric = LearnedPerceptualImagePatchSimilarity(net_type="alex",normalize=True).to(device).eval().requires_grad_(False)
    alpha = receipt["experts"][task]["inference"].get("frozen_alpha",1.)
    for row in rows:
        ident = identity(output,task,row,rank)
        marker = output / "metrics" / f"{ident['key']}.json"
        parent, old_marker, old = original(receipt,task,row,rank)
        new_marker = output / "common" / f"{ident['key']}.json"
        new = c.read(new_marker)
        c.validate_record(new,ident,output)
        binding = {"original_receipt_sha256":c.sha(old_marker),"common_receipt_sha256":c.sha(new_marker),
                   "gt_sha256":row["gt_sha256"]}
        if marker.exists():
            saved = c.read(marker)
            if saved["identity"] != ident or saved["binding"] != binding:
                raise ValueError("saved metric identity changed")
            continue
        tensors = [load_file(str(c.path(folder,item["tensor"])),device=str(device))
                   for folder,item in ((parent,old),(output,new))]
        lq = replay.lq_tensor(row,512,device)
        gt = load_gt(receipt,row,device)
        arms = output_arms(task,lq,*tensors,alpha)
        values = {}
        with torch.inference_mode():
            for name,u8 in arms.items():
                rgb = u8.float()/255.
                mse = float((gt-rgb).square().mean())
                perceptual = float(metric(rgb,gt))
                metric.reset()
                values[name] = {"psnr":100. if mse<=1e-12 else -10*math.log10(mse),"lpips":perceptual}
        if not all(math.isfinite(v) for arm in values.values() for v in arm.values()):
            raise ValueError("nonfinite scores")
        c.write_new(marker,{"identity":ident,"binding":binding,"metrics":values,"training_fit_only":True})
        print(f"COMMON_PROTOCOL_SCORED rank={rank} key={ident['key']}",flush=True)


def verify(output, receipt):
    check_metric_runtimes(output)
    rows,expected = [],[]
    for rank in range(8):
        task,selected = c.assigned(receipt["config"]["tasks"],receipt["rows"],rank,8)
        for row in selected:
            ident = identity(output,task,row,rank)
            _,old_marker,_ = original(receipt,task,row,rank)
            new_marker = output / "common" / f"{ident['key']}.json"
            c.validate_record(c.read(new_marker),ident,output)
            record = c.read(output / "metrics" / f"{ident['key']}.json")
            binding = {"original_receipt_sha256":c.sha(old_marker),"common_receipt_sha256":c.sha(new_marker),
                       "gt_sha256":row["gt_sha256"]}
            if record["identity"] != ident or record["binding"] != binding or not record["training_fit_only"]:
                raise ValueError("metric receipt mismatch")
            rows.append(record); expected.append(ident["key"])
    for folder in ("common","metrics"):
        if {f.stem for f in (output/folder).glob("*.json")} != set(expected):
            raise ValueError("extra/missing records")
    summary = p.aggregate(rows,expected,receipt["config"]["mean_nonregression_tolerance"])
    result = {"status":"common_protocol_diagnosis_complete", "receipt_sha256":c.sha(output/"receipt.json"),
              "records":128,"by_task":summary,"comparisons":rows,"training_fit_only":True,
              "training":False,"rain_adapted":False,"router_qualified":False,"protocol_selected":False,
              "node_runtimes":{str(n):c.read(output/f"runtime_node{n}.json") for n in range(4)},
              "gt_pixels_read":"only_frozen_128_training_pairs", "protected_pixels_read":False,
              "verification":"saved_tensor_SHA_and_metric_identity_plus_aggregate_recomputation_not_independent_LPIPS_reexecution"}
    c.write_new(output/"result.json",result)
    print("COMMON_PROTOCOL_RESULT_VERIFIED records=128 engineering_complete=true scientific_pass_not_claimed=true",flush=True)
    print(json.dumps(summary,sort_keys=True),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["preflight","hardware","prepare","runtime","prompts","run","verify"])
    parser.add_argument("--run-id",default="common_protocol_4x2_v1")
    parser.add_argument("--launch-id")
    parser.add_argument("--node-rank",type=int,choices=range(4),default=0)
    parser.add_argument("--asset-root",type=Path)
    parser.add_argument("--report",type=Path)
    parser.add_argument("--require-parent",action="store_true")
    args = parser.parse_args()
    if args.phase=="preflight":
        receipt = p.make_receipt(ROOT,args.asset_root or ROOT,require_parent=args.require_parent)
        result = {"status":"COMMON_PROTOCOL_CPU_PREFLIGHT_PASS","topology":c.TOPOLOGY,
                  "selection_digest":receipt["selection_digest"],"source_bindings":receipt["sources"],
                  "records":128,"gt_metadata_records":len(receipt["gt"]),"pixels_decoded":False,
                  "parent_runtime_artifacts_verified":args.require_parent,"gpu_used":False}
        if args.report:
            c.write_new(args.report,result)
        print(json.dumps(result,sort_keys=True),flush=True)
        return
    cfg = c.read(ROOT/p.CONFIG)
    if str(ROOT.resolve())!=cfg["active_root"] or args.asset_root or args.report:
        raise ValueError("execute GPU phases only in migrated root, without asset/report overrides")
    output = p.run_path(ROOT,args.run_id)
    if args.phase in ("prepare","prompts","verify") and args.node_rank!=0:
        raise ValueError("only node0 may prepare prompts or verify cluster")
    if args.phase=="hardware":
        print(json.dumps(hardware(True)),flush=True); return
    if args.phase=="prepare":
        import shutil
        if shutil.disk_usage(ROOT).free<4*1024**3:
            raise ValueError("need 4GiB free; preserve old outputs")
        receipt=p.make_receipt(ROOT,ROOT,require_parent=True)
        c.write_new(output/"receipt.json",receipt); return
    receipt=p.load_receipt(ROOT,output)
    if args.phase=="runtime":
        runtime(output,receipt,args.node_rank,idle=True); return
    if args.phase in ("prompts","run"):
        runtime(output,receipt,args.node_rank,idle=False)
        check_metric_runtimes(output)
        os.environ.update(KLEIN_4B_BASE_MODEL_PATH=str(c.path(ROOT,"models/flux2-klein-base-4b/flux-2-klein-base-4b.safetensors")),
                          AE_MODEL_PATH=str(c.path(ROOT,"models/flux2-klein-base-4b/ae.safetensors")),
                          HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",TORCH_HOME=str(ROOT/"models/torch"))
    if args.phase=="prompts":
        replay.prompts(output,prompt_view(receipt)); return
    if args.phase=="run":
        import torch
        if int(os.environ["REPLAY_NODE_RANK"])!=args.node_rank:
            raise ValueError("node rank mismatch")
        rank,local=c.worker_rank(args.node_rank,int(os.environ["LOCAL_RANK"]),int(os.environ["WORLD_SIZE"]),
                                 int(os.environ["LOCAL_WORLD_SIZE"]),int(os.environ["RANK"]))
        torch.cuda.set_device(local)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        generate(output,receipt,rank,local)
        gc.collect(); torch.cuda.empty_cache()
        score(output,receipt,rank,local)
    else:
        if not args.launch_id or not cluster.check(output,args.launch_id,"done"):
            raise ValueError("all four nodes must complete")
        verify(output,receipt)


if __name__=="__main__":
    main()
