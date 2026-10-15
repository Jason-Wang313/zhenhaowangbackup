#!/usr/bin/env python3
"""Final bounded asymmetric absence gate for conditional low-light noise routing."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lowlight_restoration.qwen_compositional_router import COMPONENTS, compositional_metrics, compositional_route
from smoke.run_foundir_qwen_compositional_router_v1 import load_contract, load_feature, predict_probabilities
from smoke.run_foundir_qwen_five_class_gate_ddp import candidate_key, sha256, workspace_path


def load_head(config):
    path = workspace_path(config["base"]["head"])
    if sha256(path) != config["base"]["head_sha256"]:
        raise RuntimeError("conditional head changed")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name).numpy() for name in handle.keys()}


def apply_gate(base_probabilities, conditional, *, route_margin, positive_margin, negative_margin):
    values = base_probabilities.copy()
    noise_index, lowlight_index = COMPONENTS.index("noise"), COMPONENTS.index("low_light")
    eligible = values[:, lowlight_index] >= route_margin
    values[eligible & (conditional >= positive_margin), noise_index] = 1.0
    values[eligible & (conditional <= 1.0-negative_margin), noise_index] = 0.0
    uncertain = eligible & (conditional > 1.0-negative_margin) & (conditional < positive_margin)
    values[uncertain, noise_index] = 0.5
    return values


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config, config_sha, train_rows, singleton_rows, mixed_rows = load_contract(args.config.resolve())
    head = load_head(config)
    if args.check_only:
        print(f"FOUNDIR_QWEN_COMPOSITIONAL_V3_PREFLIGHT_PASS config_sha256={config_sha}"); return
    predictions_path, result_path = (workspace_path(config["outputs"][key]) for key in ("predictions", "result"))
    if predictions_path.exists() or result_path.exists(): raise FileExistsError("refusing to replace v3 artifacts")
    started=time.monotonic(); train_dir=workspace_path(config["features"]["fit_calibration_dir"])
    train_features=np.stack([load_feature(train_dir,row,config) for row in train_rows])
    splits=np.asarray([row["head_split"] for row in train_rows]); calibration=splits=="calibration"
    x=(train_features[calibration]-head["feature_mean"])/head["feature_scale"]
    base=predict_probabilities(x,head["coefficient"],head["intercept"])
    conditional=predict_probabilities(x,head["conditional_noise_coefficient"][None,:],head["conditional_noise_intercept"])[:,0]
    labels=[row["label"] for row,keep in zip(train_rows,calibration,strict=True) if keep]
    selected=None
    for negative_margin in config["base"]["candidate_negative_noise_margins"]:
        values=apply_gate(base,conditional,route_margin=float(config["base"]["route_margin"]),positive_margin=float(config["base"]["positive_noise_margin"]),negative_margin=float(negative_margin))
        metrics=compositional_metrics(values.tolist(),labels,margin=float(config["base"]["route_margin"]))
        item={"negative_noise_margin":float(negative_margin),"metrics":metrics,"key":candidate_key(metrics,config["gates"])}
        if selected is None or item["key"]>selected["key"]: selected=item
    assert selected is not None
    print("FOUNDIR_QWEN_COMPOSITIONAL_V3_CALIBRATION "+json.dumps({"negative_margin":selected["negative_noise_margin"],"overall":selected["metrics"]["overall"]},sort_keys=True),flush=True)
    development_rows=singleton_rows+mixed_rows; singleton_dir=workspace_path(config["features"]["singleton_development_dir"]); mixed_dir=workspace_path(config["features"]["mixed_development_dir"])
    features=np.stack([load_feature(singleton_dir,row,config) for row in singleton_rows]+[load_feature(mixed_dir,row,config) for row in mixed_rows])
    x=(features-head["feature_mean"])/head["feature_scale"]
    base=predict_probabilities(x,head["coefficient"],head["intercept"]); conditional=predict_probabilities(x,head["conditional_noise_coefficient"][None,:],head["conditional_noise_intercept"])[:,0]
    values=apply_gate(base,conditional,route_margin=float(config["base"]["route_margin"]),positive_margin=float(config["base"]["positive_noise_margin"]),negative_margin=selected["negative_noise_margin"])
    labels=[row["label"] for row in development_rows]; routes=[compositional_route(row,margin=float(config["base"]["route_margin"])) for row in values]
    development=compositional_metrics(values.tolist(),labels,margin=float(config["base"]["route_margin"])); n=len(singleton_rows)
    singleton=compositional_metrics(values[:n].tolist(),labels[:n],margin=float(config["base"]["route_margin"])); mixed=compositional_metrics(values[n:].tolist(),labels[n:],margin=float(config["base"]["route_margin"]))
    gate_pass=candidate_key(development,config["gates"])[0]==1.0; predictions_path.parent.mkdir(parents=True,exist_ok=True)
    with predictions_path.open("x",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=("record_id","truth","route","accepted","conditional_noise_probability")); writer.writeheader()
        for row,truth,route,q in zip(development_rows,labels,routes,conditional,strict=True): writer.writerow({"record_id":row["record_id"],"truth":truth,"route":route or "abstain","accepted":route is not None,"conditional_noise_probability":f"{q:.9f}"})
    payload={"schema_version":1,"status":"asymmetric_compositional_development_diagnostic_complete","gate_pass":gate_pass,"selected":selected,"development":development,"singleton":singleton,"mixed":mixed,"gates":config["gates"],"base_head_sha256":config["base"]["head_sha256"],"predictions":predictions_path.relative_to(ROOT).as_posix(),"predictions_sha256":sha256(predictions_path),"config_sha256":config_sha,"elapsed_seconds":time.monotonic()-started,"gpu_hours":0,"internal_or_official_test_used":False,"development_reused_after_prior_router_diagnosis":True}
    result_path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); print("FOUNDIR_QWEN_COMPOSITIONAL_V3_RESULT "+json.dumps(payload,sort_keys=True),flush=True)


if __name__=="__main__": main()
