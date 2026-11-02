"""CPU-only real checkpoint load through production initializer, base linears on meta."""
import argparse
import json
from pathlib import Path
import sys
from unittest import mock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"third_party/flux2/src"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root",type=Path,required=True)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from lowlight_restoration.lora import inject_lora,lora_modules
    from smoke import existing_flux_replay_contract_v1 as c
    from smoke import rain_native50_warmstart_contract_v1 as p
    from smoke import train_foundir_rain_native50_warmstart_v1 as impl
    from smoke.audit_foundir_existing_flux_experts_v1 import expected_modules
    file=c.path(args.asset_root,p.NATIVE["path"])
    if c.sha(file)!=p.NATIVE["sha256"]:raise ValueError("native checkpoint changed")
    weights=load_file(str(file),device="cpu")
    expected={f"{name}.lora_{part}.weight" for name in expected_modules() for part in ("A","B")}
    if set(weights)!=expected or any(t.dtype!=torch.bfloat16 or not torch.isfinite(t).all() for t in weights.values()):
        raise ValueError("invalid native LoRA tensors")
    model=torch.nn.Module()
    for name in expected_modules():
        parent=model;parts=name.split(".")
        for part in parts[:-1]:
            if not hasattr(parent,part):parent.add_module(part,torch.nn.Module())
            parent=getattr(parent,part)
        a=weights[name+".lora_A.weight"];b=weights[name+".lora_B.weight"]
        if a.shape[0]!=32 or b.shape[1]!=32:raise ValueError("rank mismatch")
        parent.add_module(parts[-1],torch.nn.Linear(a.shape[1],b.shape[0],bias=False,device="meta",dtype=torch.bfloat16))
    def materialize_lora(*args,**kwargs):
        names=inject_lora(*args,**kwargs)
        for module in lora_modules(args[0]).values():
            module.lora_A.to_empty(device="cpu");module.lora_B.to_empty(device="cpu")
        return names
    receipt={"native_checkpoint":p.NATIVE,"lora":c.read(ROOT/p.NATIVE_CONFIG)["lora"]}
    with mock.patch("flux2.util.load_flow_model",return_value=model),mock.patch("lowlight_restoration.lora.inject_lora",side_effect=materialize_lora),mock.patch.object(impl,"ROOT",args.asset_root):
        loaded=impl.new_model(receipt,"cpu")
    trained={name:t for name,t in loaded.named_parameters() if t.requires_grad}
    if set(trained)!=set(weights) or any(t.dtype!=torch.float32 or not torch.equal(t,weights[n].float()) for n,t in trained.items()):
        raise ValueError("native BF16 to FP32 initialization changed values")
    if any(t.requires_grad or t.device.type!="meta" for n,t in loaded.named_parameters() if n not in trained):
        raise ValueError("base unexpectedly materialized/trainable")
    optimizer=torch.optim.AdamW(list(trained.values()),lr=1e-5,weight_decay=0.,fused=False)
    if optimizer.state:raise ValueError("Adam not fresh")
    if c.sha(file)!=p.NATIVE["sha256"]:raise ValueError("source checkpoint modified")
    result={"status":"RAIN_NATIVE50_REAL_LORA_CPU_LOAD_PASS","native_checkpoint":p.NATIVE,
        "modules":50,"tensors":len(trained),"parameters":sum(t.numel() for t in trained.values()),
        "input_dtype":"BF16","trainable_dtype":"FP32","every_value_equal":True,"adam_fresh":True,
        "base_linears":"meta_shape_stubs_not_actual_FLUX_forward","gpu_used":False,"pixels_read":False,
        "cuda_forward_or_quality_qualified":False}
    c.write_new(args.report,result);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()
