"""Bounded-memory Hopper execution; no change to loss, crop or optimizer state."""
from contextlib import contextmanager
from types import MethodType
import os

WORKSPACE_MIB = '256'

def configure_environment():
    # Must be set before CUDA/cuDNN initialization. Not a measured total-memory cap.
    os.environ['CUDNN_CONV_WSCAP_DBG'] = WORKSPACE_MIB
    os.environ['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'] = '64'


def hardware(require_idle):
    from smoke import replay_foundir_existing_flux_experts_v1 as replay
    return replay.hardware(8, require_idle=require_idle)


def execution_signature(value):
    from smoke.train_foundir_rain_flow_adaptation_v1 import runtime_compatibility
    name=value.get('gpu_name','')
    if not any(f in name for f in ('H100','H200')):
        raise ValueError('only H100/H200 are supported')
    return {**runtime_compatibility(value), 'gpu_name':'Hopper_H100_or_H200'}


def configure_vae(ae, chunk_size=256):
    """Chunk only queries, retaining ALL keys/values and spatial context."""
    import torch
    from flux2.autoencoder import AttnBlock
    if chunk_size < 1:raise ValueError('positive attention query chunk required')
    def attention(self, x):
        x=self.norm(x);b,c,h,w=x.shape
        def project(layer):return layer(x).reshape(b,1,c,h*w).transpose(-2,-1).contiguous()
        q,k,v=project(self.q),project(self.k),project(self.v)
        parts=[torch.nn.functional.scaled_dot_product_attention(q[:,:,start:start+chunk_size],k,v)
               for start in range(0,h*w,chunk_size)]
        return torch.cat(parts,dim=2).transpose(-2,-1).reshape(b,c,h,w)
    for block in ae.modules():
        if isinstance(block,AttnBlock):block.attention=MethodType(attention,block)
    return ae


@contextmanager
def offload_activations():
    """Move saved non-leaf CUDA activations to host; keep leaf weights resident.

    Synchronous copies deliberately avoid racing host writes with backward reads.
    Values/dtypes are preserved, at a CPU RAM and PCIe throughput cost.
    """
    import torch
    def pack(tensor):
        if not tensor.is_cuda or tensor.is_leaf:return (None,tensor)
        return (tensor.device,tensor.detach().to('cpu',copy=True))
    def unpack(saved):
        device,tensor=saved
        return tensor if device is None else tensor.to(device)
    with torch.autograd.graph.saved_tensors_hooks(pack,unpack):yield
