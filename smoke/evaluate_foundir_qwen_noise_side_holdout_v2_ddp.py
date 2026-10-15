#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,sys,time
from pathlib import Path
import numpy as np,torch,torch.distributed as dist
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lowlight_restoration.qwen_compositional_router import compositional_route
from lowlight_restoration.qwen_eleven_class_gate import consensus_metrics
from smoke.run_foundir_qwen_compositional_router_v1 import predict_probabilities
from smoke.run_foundir_qwen_five_class_gate_ddp import extract,feature_metadata,feature_path,load_feature,sha256,workspace_path
from smoke.run_foundir_qwen_lowlight_noise_dual_nonlinear_v2 import apply_dual,model_from,tensors
def contract(path):
 c=json.loads(path.read_text());cs=sha256(path); arts=[(workspace_path(c["data"]["manifest"]),c["data"]["manifest_sha256"]),(workspace_path(c["qwen"]["training_config"]),c["qwen"]["training_config_sha256"]),(workspace_path(c["qwen"]["adapter"]),c["qwen"]["adapter_sha256"]),(workspace_path(c["router"]["base_head"]),c["router"]["base_head_sha256"]),(workspace_path(c["router"]["lowlight_head"]),c["router"]["lowlight_head_sha256"]),(workspace_path(c["router"]["noise_head"]),c["router"]["noise_head_sha256"])]
 for p,h in arts:
  if sha256(p)!=h:raise RuntimeError(f"artifact changed: {p}")
 with arts[0][0].open(newline="") as h:rows=list(csv.DictReader(h))
 if len(rows)!=192 or any(sum(r["label"]==l for r in rows)!=96 for l in ("noise","noise+low_light")):raise RuntimeError("cardinality changed")
 if c["data"].get("development_internal_or_official_test_allowed") is not False:raise RuntimeError("protected data forbidden")
 return c,cs,rows
def evaluate(c,cs,rows):
 rp=workspace_path(c["outputs"]["result"]);pp=workspace_path(c["outputs"]["predictions"])
 if rp.exists() or pp.exists():raise FileExistsError("refusing overwrite")
 f=[load_feature(feature_path(c,r["record_id"]),feature_metadata(c,cs,r),c["features"]["feature_dim"]) for r in rows]
 if any(v is None for v in f):raise RuntimeError("features incomplete")
 f=np.stack([v.numpy() for v in f]);bt={n:v.numpy() for n,v in tensors(workspace_path(c["router"]["base_head"])).items()};z=((f-bt["feature_mean"])/bt["feature_scale"]).astype(np.float32);bp=predict_probabilities(z,bt["coefficient"],bt["intercept"]);linear=predict_probabilities(z,bt["conditional_noise_coefficient"][None,:],bt["conditional_noise_intercept"])[:,0];lm=model_from(workspace_path(c["router"]["lowlight_head"]),z.shape[1]);nm=model_from(workspace_path(c["router"]["noise_head"]),z.shape[1])
 with torch.inference_mode():lq=torch.sigmoid(lm(torch.from_numpy(z))).numpy();nq=torch.sigmoid(nm(torch.from_numpy(z))).numpy()
 shim={"base":{"route_margin":c["router"]["route_margin"],"conditional_positive_margin":c["router"]["conditional_positive_margin"],"conditional_negative_margin":c["router"]["conditional_negative_margin"]},"lowlight_side":{"margin":c["router"]["lowlight_margin"]}}
 vals,lt,nt=apply_dual(bp,linear,lq,nq,shim,c["router"]["noise_margin"]);routes=[compositional_route(v,margin=c["router"]["route_margin"]) for v in vals];truth=[r["label"] for r in rows];m=consensus_metrics(routes,truth);present={k:v for k,v in m["per_class"].items() if v["records"]};g=c["gates"];o=m["overall"];passed=o["wrong_rate"]<=g["maximum_wrong_rate"] and o["coverage"]>=g["minimum_coverage"] and all(v["wrong_rate"]<=g["maximum_per_class_wrong_rate"] and v["coverage"]>=g["minimum_per_class_coverage"] for v in present.values());pp.parent.mkdir(parents=True,exist_ok=True)
 with pp.open("x",newline="") as h:w=csv.DictWriter(h,fieldnames=("record_id","truth","route","accepted","lowlight_side","noise_side"));w.writeheader();[w.writerow({"record_id":r["record_id"],"truth":t,"route":x or "abstain","accepted":x is not None,"lowlight_side":bool(a),"noise_side":bool(b)}) for r,t,x,a,b in zip(rows,truth,routes,lt,nt,strict=True)]
 out={"schema_version":1,"status":"second_fresh_training_only_holdout_complete","gate_pass":passed,"metrics":{"overall":o,"per_class":present},"records":192,"source_groups_per_class":{"noise":96,"noise+low_light":96},"all_holdout_sources_unique":True,"predictions":pp.relative_to(ROOT).as_posix(),"predictions_sha256":sha256(pp),"config_sha256":cs,"development_internal_or_official_test_used":False};rp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print("FOUNDIR_QWEN_NOISE_SIDE_HOLDOUT_RESULT "+json.dumps(out,sort_keys=True),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--check-only",action="store_true");a=p.parse_args();c,cs,rows=contract(a.config.resolve())
 if a.check_only:print(f"FOUNDIR_QWEN_NOISE_SIDE_HOLDOUT_PREFLIGHT_PASS records={len(rows)} config_sha256={cs}");return
 rank,world,local=map(int,(os.environ["RANK"],os.environ["WORLD_SIZE"],os.environ["LOCAL_RANK"]));
 if world!=8 or not torch.cuda.is_available() or torch.cuda.device_count()!=8:raise RuntimeError("requires 8 GPUs")
 torch.cuda.set_device(local);device=torch.device(f"cuda:{local}");dist.init_process_group("nccl",device_id=device);start=time.monotonic();extract(c,cs,rows,rank,world,device);dist.barrier()
 if rank==0:evaluate(c,cs,rows);print(f"FOUNDIR_QWEN_NOISE_SIDE_HOLDOUT_COMPLETE elapsed_seconds={time.monotonic()-start:.2f}",flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()
