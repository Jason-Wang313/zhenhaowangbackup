"""Native noise-to-clean Rain adaptation, with an endpoint teacher only as RGB advice."""
import random
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .flow import build_reference_flow_batch


def enable_native_checkpointing(model):
    """Native Flux2.forward calls forward_kv_extract, NOT block.forward."""
    def wrap(block):
        if getattr(block,"_rain_flow_checkpointed",False):
            raise ValueError("already checkpointed")
        original=block.forward_kv_extract
        def call(*args,**kwargs):
            if block.training and torch.is_grad_enabled():
                return checkpoint(original,*args,use_reentrant=False,**kwargs)
            return original(*args,**kwargs)
        block.forward_kv_extract=call
        block._rain_flow_checkpointed=True
    for block in list(model.double_blocks)+list(model.single_blocks):wrap(block)


def sample_index(step,micro,rank,seed,count=32,world=8):
    if not 0<=rank<world or count%world or not 0<=micro<count//world or step<0:
        raise ValueError("invalid absolute sample position")
    order=list(range(count));random.Random(seed+step).shuffle(order)
    return order[micro*world+rank]


def endpoint_estimate(noisy_tokens,prediction,timesteps,shape):
    # x_t - t*v, NOT degraded-v and not x_t-v except at t=1.
    t=timesteps.float().view(-1,1,1)
    clean_tokens=noisy_tokens.float()-t*prediction.float()
    if clean_tokens.shape!=(shape[0],shape[2]*shape[3],shape[1]):
        raise ValueError("native token shape changed")
    return clean_tokens.transpose(1,2).reshape(shape)


def training_objective(model,ae,batch,context,ids,pack,metric,weights,*,noise,timesteps):
    clean=batch["clean_latent"].float();reference=batch["reference_latent"].float()
    # Construct the path in FP32, cast model tokens only. GT defines the velocity.
    flow=build_reference_flow_batch(clean,reference,noise.float(),timesteps.float(),pack,10)
    with torch.autocast(clean.device.type,dtype=torch.bfloat16,enabled=clean.is_cuda):
        prediction=model(x=flow.model_tokens.to(torch.bfloat16),x_ids=flow.model_ids,
            timesteps=flow.timesteps,ctx=context,ctx_ids=ids,guidance=None)[:,:flow.target_token_count]
    estimate=endpoint_estimate(flow.model_tokens[:,:flow.target_token_count],prediction,timesteps,clean.shape)
    def decode(z):
        with torch.autocast(z.device.type,dtype=torch.bfloat16,enabled=z.is_cuda):
            return ae.decode(z).float().clamp(-1,1)
    decoded=checkpoint(decode,estimate,use_reentrant=False)
    terms={"flow_mse":F.mse_loss(prediction.float(),flow.target_velocity.float()),
           "gt_latent_l1":F.l1_loss(estimate,clean),
           "gt_rgb_mse":F.mse_loss(decoded,batch["gt"]),
           "gt_lpips":metric((decoded+1)*.5,(batch["gt"]+1)*.5),
           "teacher_rgb_l1":F.l1_loss(decoded,batch["teacher_rgb"])}
    if set(weights)!=set(terms):raise ValueError("loss terms changed")
    return sum(weights[k]*v for k,v in terms.items()),terms
