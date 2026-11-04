"""Immutable CPU VLM-only state; the image head is identified in the receipt."""
from pathlib import Path
import os
import time
from smoke import rain_native50_continue_state_v1 as io
c=io.c


def save(output,model,optimizer,step,history,identity):
    import torch
    if not 0<=step<=1600 or len(history)!=step: raise ValueError('invalid progress')
    folder=Path(output)/'steps'; folder.mkdir(parents=True,exist_ok=True)
    params=io.cpu_tree(model.vlm.state_dict())
    value={'step':step,'next_slot':step*32,'vlm':params,'optimizer':io.cpu_tree(optimizer.state_dict()),
           'rng':io.rng_state('cpu'),'history':history,'identity':identity,'parameter_hash':io.parameter_hash(params)}
    name=f'vlm_{step:06d}_{time.time_ns()}.pt'; file=folder/name
    with file.open('xb') as f: torch.save(value,f); f.flush(); os.fsync(f.fileno())
    fd=os.open(folder,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    c.write_new(folder/f'step_{step:06d}.json',{'step':step,'file':'steps/'+name,'sha256':c.sha(file),
                'identity':identity,'parameter_hash':value['parameter_hash']})


def load_step(output,step,identity):
    import torch
    if type(step) is not int or not 0<=step<=1600: raise ValueError('step outside fixed budget')
    output=Path(output); marker=c.read(output/f'steps/step_{step:06d}.json'); file=c.path(output,marker['file'])
    if marker['step']!=step or marker['identity']!=identity or file.parent!=(output/'steps').resolve() or c.sha(file)!=marker['sha256']:
        raise ValueError('checkpoint identity/hash mismatch')
    v=torch.load(file,map_location='cpu',weights_only=True)
    if (v['identity']!=identity or v['step']!=step or v['next_slot']!=step*32 or len(v['history'])!=step
        or io.parameter_hash(v['vlm'])!=marker['parameter_hash'] or v['parameter_hash']!=marker['parameter_hash']):
        raise ValueError('checkpoint progress/parameters mismatch')
    if any(t.dtype!=torch.float32 or not torch.isfinite(t).all() for t in v['vlm'].values()): raise ValueError('invalid VLM state')
    entries=v['optimizer']['state']
    if step and len(entries)!=len(v['vlm']): raise ValueError('Adam states missing')
    for e in entries.values():
        if float(e['step'])!=step or any(not torch.isfinite(e[k]).all() for k in ('exp_avg','exp_avg_sq')): raise ValueError('invalid Adam')
    return v


def load(output,identity):
    files=sorted((Path(output)/'steps').glob('step_*.json'))
    if not files: return None
    if [p.name for p in files]!=[f'step_{i:06d}.json' for i in range(len(files))]: raise ValueError('checkpoint gap; no rollback')
    return load_step(output,len(files)-1,identity)


def restore(model,optimizer,value):
    model.vlm.load_state_dict(value['vlm'],strict=True); optimizer.load_state_dict(value['optimizer']); io.restore_rng(value['rng'],'cpu')
