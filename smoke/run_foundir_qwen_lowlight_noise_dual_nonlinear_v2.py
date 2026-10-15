#!/usr/bin/env python3
"""Train the symmetric Noise-side nonlinear guard and evaluate both guards."""
from __future__ import annotations
import argparse,copy,csv,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from lowlight_restoration.qwen_compositional_router import COMPONENTS,compositional_metrics,compositional_route
from smoke.run_foundir_qwen_compositional_router_v1 import load_contract,load_feature,predict_probabilities
from smoke.run_foundir_qwen_compositional_router_v3 import apply_gate
from smoke.run_foundir_qwen_five_class_gate_ddp import candidate_key,sha256,workspace_path
from smoke.run_foundir_qwen_lowlight_noise_nonlinear_v1 import TinyHead,targeted_metrics

def tensors(path):
 with safe_open(path,framework="pt",device="cpu") as h:return {n:h.get_tensor(n) for n in h.keys()}
def model_from(path,input_dim):
 t=tensors(path); m=TinyHead(input_dim,t["linear1_weight"].shape[0],0.0)
 with torch.no_grad():m.linear1.weight.copy_(t["linear1_weight"]);m.linear1.bias.copy_(t["linear1_bias"]);m.linear2.weight.copy_(t["linear2_weight"]);m.linear2.bias.copy_(t["linear2_bias"])
 return m.eval()
def apply_dual(base_p,linear_q,low_q,noise_q,config,noise_margin):
 v=apply_gate(base_p,linear_q,route_margin=config["base"]["route_margin"],positive_margin=config["base"]["conditional_positive_margin"],negative_margin=config["base"]["conditional_negative_margin"]); ix={n:COMPONENTS.index(n) for n in COMPONENTS}; absent=lambda n:base_p[:,ix[n]]<=1-config["base"]["route_margin"]
 shared=absent("blur")&absent("rain")&absent("haze"); lt=shared&(base_p[:,ix["low_light"]]>=config["base"]["route_margin"]); nt=shared&(base_p[:,ix["noise"]]>=config["base"]["route_margin"])&absent("low_light")
 lm=config["lowlight_side"]["margin"]; v[lt&(low_q>=lm),ix["noise"]]=1;v[lt&(low_q<=1-lm),ix["noise"]]=0;v[lt&(low_q>1-lm)&(low_q<lm),ix["noise"]]=.5
 v[nt&(noise_q>=noise_margin),ix["low_light"]]=1;v[nt&(noise_q<=1-noise_margin),ix["low_light"]]=0;v[nt&(noise_q>1-noise_margin)&(noise_q<noise_margin),ix["low_light"]]=.5
 return v,lt,nt
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--check-only",action="store_true");a=p.parse_args();c,cs,rows,srows,mrows=load_contract(a.config.resolve())
 for section,key in (("base","head"),("lowlight_side","head")):
  if sha256(workspace_path(c[section][key]))!=c[section]["head_sha256"]:raise RuntimeError(f"{section} changed")
 if a.check_only:print(f"FOUNDIR_QWEN_DUAL_NONLINEAR_PREFLIGHT_PASS config_sha256={cs}");return
 hp=workspace_path(c["outputs"]["head"]);pp=workspace_path(c["outputs"]["predictions"]);rp=workspace_path(c["outputs"]["result"])
 if any(x.exists() for x in (hp,pp,rp)):raise FileExistsError("refusing overwrite")
 start=time.monotonic();seed=c["noise_side"]["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);d=workspace_path(c["features"]["fit_calibration_dir"]);f=np.stack([load_feature(d,r,c) for r in rows]);lab=np.asarray([r["label"] for r in rows]);sp=np.asarray([r["head_split"] for r in rows]);bt={n:v.numpy() for n,v in tensors(workspace_path(c["base"]["head"])).items()};x=torch.from_numpy(((f-bt["feature_mean"])/bt["feature_scale"]).astype(np.float32));sel=np.isin(lab,c["noise_side"]["classes"]);fit=sel&(sp=="fit");cal=sel&(sp=="calibration");y=torch.from_numpy((lab=="noise+low_light").astype(np.float32));model=TinyHead(x.shape[1],c["noise_side"]["hidden_dim"],c["noise_side"]["dropout"]);opt=torch.optim.AdamW(model.parameters(),lr=c["noise_side"]["learning_rate"],weight_decay=c["noise_side"]["weight_decay"]);lossfn=torch.nn.BCEWithLogitsLoss();fi=torch.where(torch.from_numpy(fit))[0];ci=torch.where(torch.from_numpy(cal))[0];g=torch.Generator().manual_seed(seed);best=1e9;state=None;stale=0;epoch_best=0
 for epoch in range(c["noise_side"]["maximum_epochs"]):
  model.train();order=fi[torch.randperm(len(fi),generator=g)]
  for s in range(0,len(order),c["noise_side"]["batch_size"]):idx=order[s:s+c["noise_side"]["batch_size"]];opt.zero_grad();loss=lossfn(model(x[idx]),y[idx]);loss.backward();opt.step()
  model.eval()
  with torch.inference_mode():vl=float(lossfn(model(x[ci]),y[ci]))
  if vl<best-1e-5:best=vl;state=copy.deepcopy(model.state_dict());stale=0;epoch_best=epoch+1
  else:stale+=1
  if stale>=c["noise_side"]["early_stopping_patience"]:break
 model.load_state_dict(state);model.eval()
 with torch.inference_mode():q=torch.sigmoid(model(x[ci])).numpy()
 labs=["low_light" if label=="noise" else label for label in lab[cal].tolist()];pick=None
 for margin in c["noise_side"]["candidate_margins"]:
  met=targeted_metrics(q,labs,margin);qual=all(v["wrong_rate"]<=.01 and v["coverage"]>=.5 for v in met["per_class"].values());key=(float(qual),-met["overall"]["wrong_rate"],met["overall"]["coverage"]);item={"margin":margin,"metrics":met,"key":key}
  if pick is None or key>pick["key"]:pick=item
 print("FOUNDIR_QWEN_NOISE_SIDE_CALIBRATION "+json.dumps({"best_epoch":epoch_best,"bce":best,"selected":pick},sort_keys=True),flush=True)
 allrows=srows+mrows;sd=workspace_path(c["features"]["singleton_development_dir"]);md=workspace_path(c["features"]["mixed_development_dir"]);af=np.stack([load_feature(sd,r,c) for r in srows]+[load_feature(md,r,c) for r in mrows]);z=((af-bt["feature_mean"])/bt["feature_scale"]).astype(np.float32);bp=predict_probabilities(z,bt["coefficient"],bt["intercept"]);lq=predict_probabilities(z,bt["conditional_noise_coefficient"][None,:],bt["conditional_noise_intercept"])[:,0];lm=model_from(workspace_path(c["lowlight_side"]["head"]),z.shape[1])
 with torch.inference_mode():lowq=torch.sigmoid(lm(torch.from_numpy(z))).numpy();noiseq=torch.sigmoid(model(torch.from_numpy(z))).numpy()
 vals,lt,nt=apply_dual(bp,lq,lowq,noiseq,c,pick["margin"]);truth=[r["label"] for r in allrows];routes=[compositional_route(v,margin=c["base"]["route_margin"]) for v in vals];dev=compositional_metrics(vals.tolist(),truth,margin=c["base"]["route_margin"]);n=len(srows);single=compositional_metrics(vals[:n].tolist(),truth[:n],margin=c["base"]["route_margin"]);mixed=compositional_metrics(vals[n:].tolist(),truth[n:],margin=c["base"]["route_margin"]);passed=candidate_key(dev,c["gates"])[0]==1
 hp.parent.mkdir(parents=True,exist_ok=True);save_file({"linear1_weight":model.linear1.weight.detach(),"linear1_bias":model.linear1.bias.detach(),"linear2_weight":model.linear2.weight.detach(),"linear2_bias":model.linear2.bias.detach()},hp,metadata={"config_sha256":cs,"margin":str(pick["margin"]),"best_epoch":str(epoch_best)})
 with pp.open("x",newline="") as h:w=csv.DictWriter(h,fieldnames=("record_id","truth","route","accepted","lowlight_side","noise_side"));w.writeheader();[w.writerow({"record_id":r["record_id"],"truth":t,"route":o or "abstain","accepted":o is not None,"lowlight_side":bool(a),"noise_side":bool(b)}) for r,t,o,a,b in zip(allrows,truth,routes,lt,nt,strict=True)]
 out={"schema_version":1,"status":"dual_nonlinear_development_diagnostic_complete","gate_pass":passed,"training":{"best_epoch":epoch_best,"validation_bce":best},"margin_selection":pick,"development":dev,"singleton":single,"mixed":mixed,"head":hp.relative_to(ROOT).as_posix(),"head_sha256":sha256(hp),"predictions":pp.relative_to(ROOT).as_posix(),"predictions_sha256":sha256(pp),"elapsed_seconds":time.monotonic()-start,"gpu_hours":0,"internal_or_official_test_used":False,"development_reused":True};rp.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print("FOUNDIR_QWEN_DUAL_NONLINEAR_RESULT "+json.dumps(out,sort_keys=True),flush=True)
if __name__=="__main__":main()
