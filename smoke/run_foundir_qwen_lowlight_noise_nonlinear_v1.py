#!/usr/bin/env python3
"""Train one tiny nonlinear head for the remaining Low-light/Noise+Low-light ambiguity."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_compositional_router import COMPONENTS, compositional_metrics, compositional_route
from smoke.run_foundir_qwen_compositional_router_v1 import load_contract, load_feature, predict_probabilities
from smoke.run_foundir_qwen_compositional_router_v3 import apply_gate
from smoke.run_foundir_qwen_five_class_gate_ddp import candidate_key, sha256, workspace_path


class TinyHead(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, values):
        return self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(values)))).squeeze(-1)


def load_base(config):
    path = workspace_path(config["base"]["head"])
    if sha256(path) != config["base"]["head_sha256"]:
        raise RuntimeError("frozen base head changed")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).numpy() for name in handle.keys()}


def targeted_metrics(probabilities, labels, margin):
    totals = Counter(); per_class = {name: Counter() for name in ("low_light", "noise+low_light")}
    for probability, truth in zip(probabilities, labels, strict=True):
        route = "noise+low_light" if probability >= margin else "low_light" if probability <= 1.0-margin else None
        outcome = "refusal" if route is None else "correct" if route == truth else "wrong"
        totals[outcome] += 1; per_class[truth][outcome] += 1
    def rates(counter, count):
        accepted=counter["correct"]+counter["wrong"]
        return {"records":count,"correct":counter["correct"],"wrong":counter["wrong"],"refusal":counter["refusal"],"wrong_rate":counter["wrong"]/count,"coverage":accepted/count,"accepted_accuracy":counter["correct"]/accepted if accepted else 1.0}
    return {"overall":rates(totals,len(labels)),"per_class":{name:rates(per_class[name],sum(per_class[name].values())) for name in per_class},"margin":margin}


def apply_nonlinear(base_probabilities, linear_conditional, nonlinear, *, config, nonlinear_margin):
    values = apply_gate(
        base_probabilities, linear_conditional,
        route_margin=float(config["base"]["route_margin"]),
        positive_margin=float(config["base"]["conditional_positive_margin"]),
        negative_margin=float(config["base"]["conditional_negative_margin"]),
    )
    indexes = {name: COMPONENTS.index(name) for name in COMPONENTS}
    target = base_probabilities[:, indexes["low_light"]] >= float(config["base"]["route_margin"])
    for component in ("blur", "rain", "haze"):
        target &= base_probabilities[:, indexes[component]] <= 1.0-float(config["base"]["route_margin"])
    values[target & (nonlinear >= nonlinear_margin), indexes["noise"]] = 1.0
    values[target & (nonlinear <= 1.0-nonlinear_margin), indexes["noise"]] = 0.0
    uncertain = target & (nonlinear > 1.0-nonlinear_margin) & (nonlinear < nonlinear_margin)
    values[uncertain, indexes["noise"]] = 0.5
    return values, target


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--check-only",action="store_true"); args=parser.parse_args()
    config,config_sha,train_rows,singleton_rows,mixed_rows=load_contract(args.config.resolve()); base=load_base(config)
    if args.check_only:
        print(f"FOUNDIR_QWEN_LOWLIGHT_NOISE_NONLINEAR_PREFLIGHT_PASS config_sha256={config_sha}"); return
    head_path=workspace_path(config["outputs"]["head"]); predictions_path=workspace_path(config["outputs"]["predictions"]); result_path=workspace_path(config["outputs"]["result"])
    if any(path.exists() for path in (head_path,predictions_path,result_path)): raise FileExistsError("refusing to replace nonlinear artifacts")
    started=time.monotonic(); seed=int(config["nonlinear_head"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(max(1,min(16,torch.get_num_threads())))
    train_dir=workspace_path(config["features"]["fit_calibration_dir"]); features=np.stack([load_feature(train_dir,row,config) for row in train_rows]); labels=np.asarray([row["label"] for row in train_rows]); splits=np.asarray([row["head_split"] for row in train_rows])
    target_classes=set(config["nonlinear_head"]["classes"]); selected=np.asarray([label in target_classes for label in labels]); fit=selected&(splits=="fit"); calibration=selected&(splits=="calibration")
    x=torch.from_numpy(((features-base["feature_mean"])/base["feature_scale"]).astype(np.float32)); y=torch.from_numpy((labels=="noise+low_light").astype(np.float32))
    model=TinyHead(x.shape[1],int(config["nonlinear_head"]["hidden_dim"]),float(config["nonlinear_head"]["dropout"])); optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["nonlinear_head"]["learning_rate"]),weight_decay=float(config["nonlinear_head"]["weight_decay"])); loss_fn=torch.nn.BCEWithLogitsLoss()
    fit_indexes=torch.where(torch.from_numpy(fit))[0]; calibration_indexes=torch.where(torch.from_numpy(calibration))[0]; generator=torch.Generator().manual_seed(seed); best_loss=float("inf"); best_state=None; stale=0; best_epoch=0
    for epoch in range(int(config["nonlinear_head"]["maximum_epochs"])):
        model.train(); order=fit_indexes[torch.randperm(len(fit_indexes),generator=generator)]
        for start in range(0,len(order),int(config["nonlinear_head"]["batch_size"])):
            indexes=order[start:start+int(config["nonlinear_head"]["batch_size"])]; optimizer.zero_grad(set_to_none=True); loss=loss_fn(model(x[indexes]),y[indexes]); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode(): validation_loss=float(loss_fn(model(x[calibration_indexes]),y[calibration_indexes]))
        if validation_loss < best_loss-1e-5: best_loss=validation_loss; best_state=copy.deepcopy(model.state_dict()); stale=0; best_epoch=epoch+1
        else: stale+=1
        if stale>=int(config["nonlinear_head"]["early_stopping_patience"]): break
    assert best_state is not None; model.load_state_dict(best_state); model.eval()
    with torch.inference_mode(): calibration_probabilities=torch.sigmoid(model(x[calibration_indexes])).numpy()
    calibration_labels=labels[calibration].tolist(); margin_selection=None
    for margin in config["nonlinear_head"]["candidate_margins"]:
        metrics=targeted_metrics(calibration_probabilities,calibration_labels,float(margin)); qualifies=all(value["wrong_rate"]<=0.01 and value["coverage"]>=0.5 for value in metrics["per_class"].values()); key=(float(qualifies),-metrics["overall"]["wrong_rate"],metrics["overall"]["coverage"])
        item={"margin":float(margin),"metrics":metrics,"key":key}
        if margin_selection is None or key>margin_selection["key"]: margin_selection=item
    assert margin_selection is not None
    print("FOUNDIR_QWEN_LOWLIGHT_NOISE_NONLINEAR_CALIBRATION "+json.dumps({"best_epoch":best_epoch,"validation_bce":best_loss,"selected":margin_selection},sort_keys=True),flush=True)
    development_rows=singleton_rows+mixed_rows; singleton_dir=workspace_path(config["features"]["singleton_development_dir"]); mixed_dir=workspace_path(config["features"]["mixed_development_dir"]); development_features=np.stack([load_feature(singleton_dir,row,config) for row in singleton_rows]+[load_feature(mixed_dir,row,config) for row in mixed_rows]); standardized=((development_features-base["feature_mean"])/base["feature_scale"]).astype(np.float32)
    base_probabilities=predict_probabilities(standardized,base["coefficient"],base["intercept"]); linear_conditional=predict_probabilities(standardized,base["conditional_noise_coefficient"][None,:],base["conditional_noise_intercept"])[:,0]
    with torch.inference_mode(): nonlinear=torch.sigmoid(model(torch.from_numpy(standardized))).numpy()
    values,target=apply_nonlinear(base_probabilities,linear_conditional,nonlinear,config=config,nonlinear_margin=margin_selection["margin"]); development_labels=[row["label"] for row in development_rows]; routes=[compositional_route(row,margin=float(config["base"]["route_margin"])) for row in values]; development=compositional_metrics(values.tolist(),development_labels,margin=float(config["base"]["route_margin"])); n=len(singleton_rows); singleton=compositional_metrics(values[:n].tolist(),development_labels[:n],margin=float(config["base"]["route_margin"])); mixed=compositional_metrics(values[n:].tolist(),development_labels[n:],margin=float(config["base"]["route_margin"])); gate_pass=candidate_key(development,config["gates"])[0]==1.0
    head_path.parent.mkdir(parents=True,exist_ok=True); save_file({"linear1_weight":model.linear1.weight.detach(),"linear1_bias":model.linear1.bias.detach(),"linear2_weight":model.linear2.weight.detach(),"linear2_bias":model.linear2.bias.detach()},head_path,metadata={"config_sha256":config_sha,"base_head_sha256":config["base"]["head_sha256"],"margin":str(margin_selection["margin"]),"best_epoch":str(best_epoch)})
    with predictions_path.open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=("record_id","truth","route","accepted","targeted","nonlinear_probability")); writer.writeheader()
        for row,truth,route,is_target,p in zip(development_rows,development_labels,routes,target,nonlinear,strict=True): writer.writerow({"record_id":row["record_id"],"truth":truth,"route":route or "abstain","accepted":route is not None,"targeted":bool(is_target),"nonlinear_probability":f"{p:.9f}"})
    payload={"schema_version":1,"status":"lowlight_noise_nonlinear_development_diagnostic_complete","gate_pass":gate_pass,"training":{"best_epoch":best_epoch,"validation_bce":best_loss,"fit_records":int(fit.sum()),"calibration_records":int(calibration.sum())},"margin_selection":margin_selection,"development":development,"singleton":singleton,"mixed":mixed,"gates":config["gates"],"head":head_path.relative_to(ROOT).as_posix(),"head_sha256":sha256(head_path),"predictions":predictions_path.relative_to(ROOT).as_posix(),"predictions_sha256":sha256(predictions_path),"config_sha256":config_sha,"elapsed_seconds":time.monotonic()-started,"gpu_hours":0,"internal_or_official_test_used":False,"development_reused_after_prior_router_diagnosis":True}
    result_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print("FOUNDIR_QWEN_LOWLIGHT_NOISE_NONLINEAR_RESULT "+json.dumps(payload,sort_keys=True),flush=True)


if __name__=="__main__": main()
