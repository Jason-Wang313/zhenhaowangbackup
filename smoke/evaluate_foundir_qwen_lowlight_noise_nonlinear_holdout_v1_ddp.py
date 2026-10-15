#!/usr/bin/env python3
"""Extract fresh features and evaluate the frozen targeted nonlinear router."""

from __future__ import annotations

import argparse,csv,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lowlight_restoration.qwen_compositional_router import COMPONENTS,compositional_route
from lowlight_restoration.qwen_eleven_class_gate import consensus_metrics
from smoke.run_foundir_qwen_compositional_router_v1 import predict_probabilities
from smoke.run_foundir_qwen_compositional_router_v3 import apply_gate
from smoke.run_foundir_qwen_five_class_gate_ddp import extract,feature_metadata,feature_path,load_feature,sha256,workspace_path
from smoke.run_foundir_qwen_lowlight_noise_nonlinear_v1 import TinyHead,apply_nonlinear


def load_contract(path):
    config=json.loads(path.read_text()); config_sha=sha256(path)
    if config.get("schema_version")!=1: raise ValueError("bad holdout config")
    artifacts=((workspace_path(config["data"]["manifest"]),config["data"]["manifest_sha256"]),(workspace_path(config["qwen"]["training_config"]),config["qwen"]["training_config_sha256"]),(workspace_path(config["qwen"]["adapter"]),config["qwen"]["adapter_sha256"]),(workspace_path(config["router"]["base_head"]),config["router"]["base_head_sha256"]),(workspace_path(config["router"]["nonlinear_head"]),config["router"]["nonlinear_head_sha256"]))
    for artifact,expected in artifacts:
        if sha256(artifact)!=expected: raise RuntimeError(f"frozen artifact changed: {artifact}")
    with artifacts[0][0].open(newline="") as handle: rows=list(csv.DictReader(handle))
    if len(rows)!=240 or {label:sum(row["label"]==label for row in rows) for label in ("low_light","noise+low_light")}!={"low_light":120,"noise+low_light":120}: raise RuntimeError("holdout cardinality changed")
    if config["data"].get("development_internal_or_official_test_allowed") is not False: raise RuntimeError("protected data must remain forbidden")
    return config,config_sha,rows


def load_tensors(path):
    with safe_open(path,framework="pt",device="cpu") as handle: return {name:handle.get_tensor(name) for name in handle.keys()}


def evaluate(config,config_sha,rows):
    result_path=workspace_path(config["outputs"]["result"]); predictions_path=workspace_path(config["outputs"]["predictions"])
    if result_path.exists() or predictions_path.exists(): raise FileExistsError("refusing to replace holdout result")
    dim=int(config["features"]["feature_dim"]); realized=[load_feature(feature_path(config,row["record_id"]),feature_metadata(config,config_sha,row),dim) for row in rows]
    if any(value is None for value in realized): raise RuntimeError("feature extraction incomplete")
    features=np.stack([value.numpy() for value in realized if value is not None]); base_t=load_tensors(workspace_path(config["router"]["base_head"])); base={name:value.numpy() for name,value in base_t.items()}; standardized=((features-base["feature_mean"])/base["feature_scale"]).astype(np.float32); base_probabilities=predict_probabilities(standardized,base["coefficient"],base["intercept"]); linear=predict_probabilities(standardized,base["conditional_noise_coefficient"][None,:],base["conditional_noise_intercept"])[:,0]
    nonlinear_t=load_tensors(workspace_path(config["router"]["nonlinear_head"])); model=TinyHead(dim,nonlinear_t["linear1_weight"].shape[0],0.0)
    with torch.no_grad(): model.linear1.weight.copy_(nonlinear_t["linear1_weight"]); model.linear1.bias.copy_(nonlinear_t["linear1_bias"]); model.linear2.weight.copy_(nonlinear_t["linear2_weight"]); model.linear2.bias.copy_(nonlinear_t["linear2_bias"])
    model.eval()
    with torch.inference_mode(): nonlinear=torch.sigmoid(model(torch.from_numpy(standardized))).numpy()
    shim={"base":{"route_margin":config["router"]["route_margin"],"conditional_positive_margin":config["router"]["conditional_positive_margin"],"conditional_negative_margin":config["router"]["conditional_negative_margin"]}}
    values,target=apply_nonlinear(base_probabilities,linear,nonlinear,config=shim,nonlinear_margin=float(config["router"]["nonlinear_margin"])); routes=[compositional_route(row,margin=float(config["router"]["route_margin"])) for row in values]; truths=[row["label"] for row in rows]; metrics=consensus_metrics(routes,truths)
    present={name:value for name,value in metrics["per_class"].items() if value["records"]}; overall=metrics["overall"]; gates=config["gates"]; gate_pass=overall["wrong_rate"]<=float(gates["maximum_wrong_rate"]) and overall["coverage"]>=float(gates["minimum_coverage"]) and all(value["wrong_rate"]<=float(gates["maximum_per_class_wrong_rate"]) and value["coverage"]>=float(gates["minimum_per_class_coverage"]) for value in present.values())
    predictions_path.parent.mkdir(parents=True,exist_ok=True)
    with predictions_path.open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=("record_id","truth","route","accepted","targeted","nonlinear_probability")); writer.writeheader()
        for row,truth,route,is_target,p in zip(rows,truths,routes,target,nonlinear,strict=True): writer.writerow({"record_id":row["record_id"],"truth":truth,"route":route or "abstain","accepted":route is not None,"targeted":bool(is_target),"nonlinear_probability":f"{p:.9f}"})
    payload={"schema_version":1,"status":"fresh_training_only_holdout_complete","gate_pass":gate_pass,"metrics":{"overall":overall,"per_class":present},"gates":gates,"records":len(rows),"source_groups_per_class":config["data"]["source_groups_per_class"],"limited_lowlight_source_diversity":True,"config_sha256":config_sha,"predictions":predictions_path.relative_to(ROOT).as_posix(),"predictions_sha256":sha256(predictions_path),"development_internal_or_official_test_used":False}
    result_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print("FOUNDIR_QWEN_LOWLIGHT_NOISE_FRESH_HOLDOUT_RESULT "+json.dumps(payload,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--check-only",action="store_true"); args=parser.parse_args(); config,config_sha,rows=load_contract(args.config.resolve())
    if args.check_only: print(f"FOUNDIR_QWEN_LOWLIGHT_NOISE_HOLDOUT_PREFLIGHT_PASS records={len(rows)} config_sha256={config_sha}"); return
    if any(name not in os.environ for name in ("RANK","WORLD_SIZE","LOCAL_RANK")): raise RuntimeError("launch with torchrun")
    rank,world,local_rank=int(os.environ["RANK"]),int(os.environ["WORLD_SIZE"]),int(os.environ["LOCAL_RANK"])
    if world!=8 or not torch.cuda.is_available() or torch.cuda.device_count()!=8: raise RuntimeError("holdout requires exactly 8 visible GPUs")
    torch.cuda.set_device(local_rank); device=torch.device(f"cuda:{local_rank}"); dist.init_process_group("nccl",device_id=device); started=time.monotonic(); extract(config,config_sha,rows,rank,world,device); dist.barrier()
    if rank==0: evaluate(config,config_sha,rows); print(f"FOUNDIR_QWEN_LOWLIGHT_NOISE_HOLDOUT_COMPLETE elapsed_seconds={time.monotonic()-started:.2f}",flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__=="__main__": main()
