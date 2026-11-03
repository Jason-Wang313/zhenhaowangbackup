"""Continuation-only immutable checkpoints, global steps 200..2000; old schema unchanged."""
import hashlib
import os
from pathlib import Path
import random
import time
from smoke import existing_flux_replay_contract_v1 as c


def cpu_tree(value):
    import torch
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def trainable_state(model):
    values={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not values or any(".lora_A." not in n and ".lora_B." not in n for n in values):
        raise ValueError("only new Rain LoRA may train")
    return cpu_tree(values)


def parameter_hash(values):
    h=hashlib.sha256()
    for name,tensor in sorted(values.items()):
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state(device):
    import torch
    return {"python":random.getstate(),"torch_cpu":torch.get_rng_state(),
            "torch_cuda":torch.cuda.get_rng_state(device) if torch.device(device).type=="cuda" else None}


def restore_rng(state,device):
    import torch
    random.setstate(state["python"]);torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:torch.cuda.set_rng_state(state["torch_cuda"],device)


def save_step(output,model,optimizer,*,step,world,rank_rng,receipt_hash,history):
    import torch
    if len(history)!=step or len(rank_rng)!=world or not 200<=step<=2000:raise ValueError("incomplete step")
    directory=Path(output)/"steps";directory.mkdir(exist_ok=True)
    values=trainable_state(model)
    payload={"step":step,"world":world,"next_global_slot":step*32,"receipt_hash":receipt_hash,
        "model":values,"parameter_order":[n for n,p in model.named_parameters() if p.requires_grad],
        "optimizer":cpu_tree(optimizer.state_dict()),"rank_rng":cpu_tree(rank_rng),"history":history,
        "may_seed_formal":True,"parameter_hash":parameter_hash(values)}
    name=f"state_{step:06d}_{time.time_ns()}.pt";file=directory/name
    with file.open("xb") as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    fd=os.open(directory,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    record={"step":step,"file":f"steps/{name}","sha256":c.sha(file),"receipt_hash":receipt_hash,
            "parameter_hash":payload["parameter_hash"]}
    c.write_new(directory/f"step_{step:06d}.json",record)
    return record


def load_latest(output,receipt_hash,world):
    import torch
    directory=Path(output)/"steps";markers=sorted(directory.glob("step_*.json"))
    if not markers:return None,None
    # Missing committed intermediate steps are corruption, not an excuse to roll back.
    records=[c.read(m) for m in markers]
    if [r["step"] for r in records]!=list(range(200,200+len(records))):raise ValueError("checkpoint receipt gap")
    record=records[-1];file=c.path(output,record["file"])
    if file.parent!=directory.resolve() or c.sha(file)!=record["sha256"]:raise ValueError("checkpoint hash/path mismatch")
    state=torch.load(file,map_location="cpu",weights_only=True)
    step=state["step"]
    if (not 200<=step<=2000 or record["step"]!=step or state["world"]!=world
        or state["receipt_hash"]!=receipt_hash or record["receipt_hash"]!=receipt_hash
        or state["next_global_slot"]!=32*step or len(state["history"])!=step
        or len(state["rank_rng"])!=world or state["may_seed_formal"] is not True
        or parameter_hash(state["model"])!=record["parameter_hash"]
        or state["parameter_hash"]!=record["parameter_hash"]):raise ValueError("checkpoint identity/progress mismatch")
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in state["model"].values()):
        raise ValueError("LoRA must be finite FP32")
    entries=state["optimizer"]["state"]
    if step and len(entries)!=len(state["model"]):raise ValueError("missing Adam moments")
    for entry in entries.values():
        if float(entry["step"])!=step or any(entry[k].dtype!=torch.float32 or not torch.isfinite(entry[k]).all()
                    for k in ("exp_avg","exp_avg_sq")):raise ValueError("invalid Adam state")
    return state,record


def restore(model,optimizer,state,rank,device):
    import torch
    values={n:p for n,p in model.named_parameters() if p.requires_grad}
    if list(values)!=state["parameter_order"] or set(values)!=set(state["model"]):raise ValueError("parameter registration changed")
    with torch.no_grad():
        for n,p in values.items():
            value=state["model"][n]
            if p.shape!=value.shape or p.dtype!=value.dtype:raise ValueError("parameter shape/dtype changed")
            p.copy_(value.to(p.device))
    optimizer.load_state_dict(state["optimizer"])
    restored=cpu_tree(optimizer.state_dict())
    if restored["param_groups"]!=state["optimizer"]["param_groups"]:raise ValueError("Adam options changed")
    for index,entry in restored["state"].items():
        for k,value in entry.items():
            if not torch.equal(value,state["optimizer"]["state"][index][k]):raise ValueError("Adam restoration changed values")
    restore_rng(state["rank_rng"][rank],device)
