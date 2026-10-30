"""V3 execution only: 1x2, 1x4, 3x2 or 1x8 ranks, including mixed H100/H200.

The 32-pair stream is unchanged. DDP averages ranks, so EACH pair's loss
must be multiplied by world/32, even when rank sample counts differ.
"""
from contextlib import contextmanager
from datetime import timedelta
import os
from pathlib import Path
import platform
import socket
import uuid

_persistent = False
_identities = None


def topology(world):
    if world not in (2, 4, 6, 8):
        raise ValueError('supported layouts are 1x2, 1x4, 3x2 and 1x8')
    return (3, 2) if world == 6 else (1, world)


def assigned(indices, rank, world):
    topology(world)
    if not 0 <= rank < world:
        raise ValueError('invalid global rank')
    return list(indices)[rank::world]


def training_slots(step, rank, world):
    assigned([], rank, world)
    if step < 1:
        raise ValueError('one-based optimizer steps required')
    return list(range((step-1)*32+rank, step*32, world))


def pair_loss_weight(world):
    topology(world)
    return world / 32


def validate_identities(rows, world):
    nodes, local = topology(world)
    if len(rows) != world or [v['rank'] for v in rows] != list(range(world)):
        raise RuntimeError('missing/duplicate global ranks')
    if len({v['uuid'] for v in rows}) != world:
        raise RuntimeError('duplicate GPU UUIDs')
    hosts = {}
    for rank, row in enumerate(rows):
        if (row['node_rank'] != rank//local or row['local_rank'] != rank % local
                or row['local_world_size'] != local):
            raise RuntimeError('rank/local/node mapping changed')
        if (row['family'] not in ('H100', 'H200') or row['memory_bytes'] < 75*1024**3
                or row['capability'] != [9, 0]):
            raise RuntimeError('requires H100/H200 SM90 with >=75 GiB each')
        hosts.setdefault(row['node_rank'], set()).add(row['hostname'])
    if (len(hosts) != nodes or any(len(v) != 1 for v in hosts.values())
            or len(set.union(*hosts.values())) != nodes):
        raise RuntimeError('expected distinct hosts, one per node rank')
    # GPU families may differ; Python/CUDA/NCCL/package versions must match.
    if len({repr(row['runtime']) for row in rows}) != 1:
        raise RuntimeError('runtime versions differ across ranks')


def runtime_versions():
    import importlib.metadata
    import torch
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda, nccl=list(torch.cuda.nccl.version()),
                packages={name:importlib.metadata.version(name) for name in
                          ('torchvision','torchmetrics','safetensors','numpy','transformers','einops')})


def hardware(local):
    import torch
    if (local not in (2,4,8) or not torch.cuda.is_available()
            or torch.cuda.device_count() != local):
        raise RuntimeError(f'requires exactly {local} visible CUDA devices on this node')
    for i in range(local):
        props = torch.cuda.get_device_properties(i)
        if (not any(k in props.name for k in ('H100','H200'))
                or props.total_memory < 75*1024**3 or (props.major,props.minor) != (9,0)):
            raise RuntimeError('only H100/H200 SM90 >=75 GiB supported')
        free,total = torch.cuda.mem_get_info(i)
        if total-free > 2*1024**3:
            raise RuntimeError(f'GPU {i} is not idle; do not overlap/kill jobs')


def setup_ddp(world):
    import torch
    import torch.distributed as dist
    global _identities
    nodes,local = topology(world)
    if dist.is_initialized():
        if dist.get_world_size() != world:
            raise RuntimeError('world size changed')
        return dist.get_rank(), torch.device('cuda',int(os.environ['LOCAL_RANK']))
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    if (int(os.environ['WORLD_SIZE']) != world or int(os.environ['LOCAL_WORLD_SIZE']) != local
            or not 0 <= rank < world or local_rank != rank % local
            or int(os.environ.get('GROUP_RANK',rank//local)) != rank//local):
        raise RuntimeError('torchrun layout differs from declared topology')
    hardware(local)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda',local_rank)
    props = torch.cuda.get_device_properties(local_rank)
    identity = dict(rank=rank,node_rank=rank//local,local_rank=local_rank,local_world_size=local,
                    hostname=socket.gethostname(),uuid=str(props.uuid),name=props.name,
                    family='H200' if 'H200' in props.name else 'H100',
                    memory_bytes=props.total_memory,capability=[props.major,props.minor],
                    runtime=runtime_versions())
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=10))
    _identities = [None]*world
    dist.all_gather_object(_identities,identity)
    validate_identities(_identities,world)
    return rank,device


def identities():
    if _identities is None:
        raise RuntimeError('GPU topology has not been verified')
    return _identities


def finish_phase():
    import torch.distributed as dist
    if not _persistent and dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def persistent_group(world):
    import torch.distributed as dist
    global _persistent
    try:
        rank,device = setup_ddp(world)
        _persistent = True
        yield rank,device
    finally:
        _persistent = False
        if dist.is_initialized():
            dist.destroy_process_group()


def leader_call(function):
    """Broadcast rank-zero failures instead of leaving peers at a barrier."""
    import torch.distributed as dist
    message = [None]
    if dist.get_rank() == 0:
        try:
            function()
        except Exception as exc:
            message[0] = f'{type(exc).__name__}: {exc}'
    dist.broadcast_object_list(message,src=0)
    if message[0] is not None:
        raise RuntimeError('rank-zero operation failed: '+message[0])


def shared_storage_probe(directory):
    """Fresh per-rank tokens prove shared write/read visibility, without images."""
    import torch.distributed as dist
    from smoke.flux_tail_checkpoint_v2 import atomic_json
    import json
    rank=dist.get_rank();world=dist.get_world_size()
    token=uuid.uuid4().hex;tokens=[None]*world
    dist.all_gather_object(tokens,token)
    directory=Path(directory)/tokens[0]
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/f'rank{rank}.json',dict(rank=rank,token=token))
    dist.barrier()
    error=None
    try:
        for i,value in enumerate(tokens):
            if json.loads((directory/f'rank{i}.json').read_text()) != dict(rank=i,token=value):
                raise RuntimeError('shared storage token mismatch')
    except Exception as exc:
        error=str(exc)
    errors=[None]*world;dist.all_gather_object(errors,error)
    if any(errors):
        raise RuntimeError('shared storage visibility failed: '+repr(errors))
    return directory


def communication_probe(device):
    """Check actual scalar and 4-MiB gradient-sized reductions before data access."""
    import torch
    import torch.distributed as dist
    import time
    rank=dist.get_rank();world=dist.get_world_size();expected=world*(world+1)//2
    result=[]
    for count in (1,1024*1024):
        value=torch.full((count,),rank+1.,dtype=torch.float32,device=device)
        dist.barrier();torch.cuda.synchronize(device);tick=time.monotonic()
        dist.all_reduce(value);torch.cuda.synchronize(device)
        if not bool((value==expected).all()):
            raise RuntimeError('NCCL all-reduce correctness failure')
        elapsed=torch.tensor(time.monotonic()-tick,device=device,dtype=torch.float64)
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        result.append(dict(bytes=count*4,max_rank_seconds=float(elapsed)))
    return result
